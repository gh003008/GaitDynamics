#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downstream 3 (project-integrated):
- Take a reference gait .mot (OpenSim Coordinates, radians), convert to model-state window
- Inpaint with the pretrained diffusion model, applying speed-scale soft guidance on pelvis_tx (AP velocity)
- DS3 preset:
    * Hard-fix: pelvis_ty, pelvis_tz (mask=1). pelvis_tx is hard-fixed unless --free-pelvis-tx, in which case it is guided.
    * Others (most kinematics): NOT hard-fixed; kept near the reference by a soft prior (hinge loss around ~30% of per-channel range).
- This yields physically plausible adaptation while allowing necessary changes to satisfy the target speed.
- Write the generated result back to .mot for OpenSim/GIF rendering
"""

import argparse
import os
import sys
import time
from pathlib import Path
import re
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from consts import OSIM_DOF_ALL, JOINTS_3D_ALL, KINETICS_ALL, MODEL_STATES_COLUMN_NAMES_NO_ARM
from args import parse_opt as parse_train_opt
from model.model import MotionModel
from model.utils import convert_addb_state_to_model_input, inverse_convert_addb_state_to_model_input, align_moving_direction, data_filter
from importlib.machinery import SourceFileLoader


def build_opt(window_len: int, target_hz: int = 100):
    opt = SimpleNamespace()
    opt.with_arm = False
    opt.with_kinematics_vel = True

    # Columns: OSIM 23 DoFs + kinetics
    opt.osim_dof_columns = list(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    opt.joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
    opt.model_states_column_names = list(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    for joint_name, _ in opt.joints_3d.items():
        opt.model_states_column_names += [f"{joint_name}_{axis}_angular_vel" for axis in ['x', 'y', 'z']]
    if opt.with_kinematics_vel:
        opt.model_states_column_names += [
            f"{col}_vel" for col in opt.model_states_column_names
            if not any(term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5'])
        ]

    opt.window_len = int(window_len)
    opt.target_sampling_rate = int(target_hz)
    opt.batch_size_inference = 1

    # Soft guidance schedule (as in speed-sweep)
    opt.guide_x_start_the_end_step = 950
    opt.guide_x_start_the_beginning_step = 100
    opt.n_guided_steps = 1
    opt.guidance_lr = 0.05

    # Pretrained diffusion checkpoint path (shipped with repo)
    opt.checkpoint = str(PROJECT_ROOT / 'example_usage' / 'GaitDynamicsDiffusion.pt')

    # Useful index groups
    opt.kinematic_diffusion_col_loc = [i for i, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    return opt


def read_opensim_mot(path: str) -> pd.DataFrame:
    """Read a Coordinates .mot written by our tools into a pandas DataFrame.
    Assumes 'endheader' line then tab-delimited 'time' + OSIM coords.
    """
    lines = Path(path).read_text().splitlines()
    # find endheader
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == 'endheader':
            start = i + 1
            break
    # header row
    header = lines[start].strip().split()
    rows = []
    for ln in lines[start + 1:]:
        if not ln.strip():
            continue
        rows.append([float(x) for x in ln.strip().split()])
    df = pd.DataFrame(rows, columns=header)
    return df


def make_out_dir(input_path: str, suffix: str = "") -> Path:
    """Create output directory: plots/DS3_<dataset>/<subject>/<condition>/"""
    # Extract dataset and subject from input path
    path_parts = Path(input_path).parts
    
    # Determine dataset (AB or LD)
    if 'ab_' in Path(input_path).stem.lower() or 'subj' in Path(input_path).stem.lower():
        dataset = 'AB'
        # Extract subject from filename (e.g., ab_Subj06_walk_09 -> Subj06)
        stem = Path(input_path).stem
        if 'Subj06' in stem:
            subject = 'Subj06'
        elif 'Subj08' in stem:
            subject = 'Subj08'
        else:
            subject = 'Unknown'
    else:
        # Assume LD dataset
        dataset = 'LD'
        # Try to find S00X pattern in path
        subject = 'Unknown'
        for part in path_parts:
            if part.startswith('S') and len(part) == 4 and part[1:].isdigit():
                subject = part
                break
    
    # Use suffix as condition name (e.g., "ds3__v1dot20ms__nolora")
    condition = suffix if suffix else 'default'
    
    out = PROJECT_ROOT / 'plots' / f'DS3_{dataset}' / subject / condition
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_pelvis_translation_velocities(poses23: np.ndarray, hz: int) -> np.ndarray:
    """Estimate pelvis translation velocities [tx, ty, tz] from raw positions, filtered then differenced."""
    p = poses23[:, [OSIM_DOF_ALL.index('pelvis_tx'), OSIM_DOF_ALL.index('pelvis_ty'), OSIM_DOF_ALL.index('pelvis_tz')]].astype(np.float64)
    p_f = data_filter(p, 15, hz, 4)
    v = np.zeros_like(p_f)
    v[1:] = (p_f[1:] - p_f[:-1]) * float(hz)
    v[0] = v[1]
    return v.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mot-in', type=str, default=str(PROJECT_ROOT / 'previews' / 'ab_Subj06_walk_09.mot'), help='Input .mot (Coordinates, radians)')
    ap.add_argument('--height-m', type=float, default=1.75, help='Subject height (m) for normalization')
    ap.add_argument('--scale', type=float, default=1.0, help='Speed scale factor (multiplies base forward speed)')
    ap.add_argument('--target-speed', type=float, default=None, help='Absolute target speed (m/s). If set, overrides --scale')
    ap.add_argument('--window-len', type=int, default=150, help='Window length (frames) to inpaint')
    ap.add_argument('--preset', type=str, default='custom', choices=['custom', 'ds3'],
                    help="'ds3' mimics figure/Downstream3 style: hard-fix pelvis tx/ty/tz (with tx speed-scaled), others free with soft prior")
    ap.add_argument('--total-seconds', type=float, default=10.0, help='Total duration to generate (seconds). Default 10s.')
    ap.add_argument('--overlap-frames', type=int, default=20, help='Overlap between consecutive windows when stitching')
    ap.add_argument('--unlock', type=str, default='none', choices=['none', 'lower', 'all'],
                    help='Which kinematic channels to free for adaptation: none (default), lower (hips/knees/ankles), all (all kinematics)')
    ap.add_argument('--prior-weight', type=float, default=0.25, help='Soft prior weight to keep unlocked joints near input')
    ap.add_argument('--prior-thd', type=float, default=0.10, help='Soft prior hinge threshold for unlocked joints')
    ap.add_argument('--pelvis-drift-weight', type=float, default=0.5, help='Soft prior weight for pelvis_ty/tz drift suppression')
    ap.add_argument('--pelvis-drift-thd', type=float, default=0.05, help='Soft prior hinge threshold for pelvis_ty/tz')
    ap.add_argument('--export-mot', action='store_true', help='Also export an OpenSim .mot of the generated sequence')
    ap.add_argument('--use-lora', action='store_true', help='Inject and load a LoRA adaptor before sampling')
    ap.add_argument('--lora-ckpt', type=str, default='', help='Path to LoRA weights (*.pt)')
    ap.add_argument('--free-pelvis-tx', action='store_true', help='Do NOT hard-fix pelvis_tx in DS3 preset; let it be guided (forward progression)')
    ap.add_argument('--lora-scale', type=float, default=1.0, help='Scale factor for LoRA delta at inference (1.0 = as trained)')
    args = ap.parse_args()

    # Read .mot
    df_mot = read_opensim_mot(args.mot_in)
    if not all(c in df_mot.columns for c in OSIM_DOF_ALL[:23]):
        missing = [c for c in OSIM_DOF_ALL[:23] if c not in df_mot.columns]
        raise SystemExit(f"Input .mot missing required OSIM columns: {missing}")
    time_s = df_mot['time'].values.astype(np.float32)
    if len(time_s) < 2:
        raise SystemExit('Not enough frames in input .mot')
    dt = float(time_s[1] - time_s[0])
    hz = int(round(1.0 / dt))

    # Assemble OSIM states: 23 DoFs + zeros for kinetics (GRF/CoP)
    poses23 = df_mot[OSIM_DOF_ALL[:23]].values.astype(np.float32)
    T_all = poses23.shape[0]
    kinetics_zeros = np.zeros((T_all, len(KINETICS_ALL)), dtype=np.float32)
    osim_states = np.concatenate([poses23, kinetics_zeros], axis=1)
    osim_cols = list(OSIM_DOF_ALL[:23] + KINETICS_ALL)

    # Align forward direction (rotate world so pelvis faces +X)
    aligned, rot_mat = align_moving_direction(osim_states, osim_cols)
    if aligned is False:
        raise SystemExit('Input orientation varies too much; unable to align moving direction')
    osim_states_aligned = aligned.detach().cpu().numpy().astype(np.float32)

    # Convert to model-state space (6v + angular v + kinematics v)
    pose_df = pd.DataFrame(osim_states_aligned, columns=osim_cols)
    converted_df, pos_vec = convert_addb_state_to_model_input(pose_df, {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}, hz)

    # Overwrite pelvis_tx/ty/tz in converted states with normalized translational velocities (as in training)
    p_vel = compute_pelvis_translation_velocities(poses23, hz)  # [T,3]
    p_vel_norm = p_vel / float(args.height_m)
    for i, name in enumerate(['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
        if name in converted_df.columns:
            converted_df[name] = p_vel_norm[:, i].astype(np.float32)

    # Reorder to the exact model-state column set
    model_cols = list(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    for joint_name in ['pelvis', 'hip_r', 'hip_l', 'lumbar']:
        for axis in ['x', 'y', 'z']:
            model_cols.append(f"{joint_name}_{axis}_angular_vel")
    # kinematics velocities for non-force, excluding pelvis_* and existing *_vel and 6v indices
    extra_vel_cols = [
        f"{col}_vel" for col in model_cols
        if not any(term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5'])
    ]
    model_cols_full = model_cols + extra_vel_cols
    # Some columns may be missing in converted_df (rare); fill zeros
    missing_any = [c for c in model_cols_full if c not in converted_df.columns]
    for m in missing_any:
        converted_df[m] = 0.0
    converted_df = converted_df[model_cols_full]

    # Determine generation plan (single window or stitched multi-window)
    T = int(args.window_len)
    N = len(converted_df)
    if N < T:
        raise SystemExit(f"Input sequence too short ({N} frames) for window_len={T}")

    # Build inference opt and model
    opt = build_opt(window_len=T, target_hz=hz)
    if args.preset == 'ds3':
        # Use hard constraint (mask=1) for pelvis_tx, so guidance steps can be minimal
        opt.n_guided_steps = 5  # Restored to original
        opt.guidance_lr = 0.02  # Restored to original
        opt.guide_x_start_the_beginning_step = 1000
        opt.guide_x_start_the_end_step = 0
    motion_model = MotionModel(opt)
    motion_model.eval()

    # Optional: inject and load LoRA
    lora_used = False
    lora_epoch = None
    lora_exp = None
    if args.use_lora:
        try:
            lora_mod = SourceFileLoader("lora_mod", str(PROJECT_ROOT / "lora" / "lora.py")).load_module()
            inject_lora = lora_mod.inject_lora
            load_lora_weights = lora_mod.load_lora_weights
            replaced, _ = inject_lora(motion_model.diffusion.model, r=16, alpha=32, dropout=0.05, include_mha_out_proj=True)
            print(f"[LoRA] Injected into {replaced} Linear layers (r=16, alpha=32, dropout=0.05).")
            if args.lora_ckpt and os.path.exists(args.lora_ckpt):
                load_lora_weights(motion_model.diffusion.model, args.lora_ckpt)
                print(f"[LoRA] Loaded weights: {args.lora_ckpt}")
                lora_used = True
                # Try to parse epoch and experiment name from path
                try:
                    ck = Path(args.lora_ckpt)
                    # Epoch from filename like epoch-170-lora.pt
                    m = re.search(r"epoch-(\d+)", ck.name)
                    if m:
                        lora_epoch = int(m.group(1))
                    # Experiment name if path contains .../adapter_train/<exp>/weights/...
                    if ck.parent.name == 'weights' and ck.parent.parent:
                        lora_exp = ck.parent.parent.name
                except Exception:
                    pass
            else:
                print("[LoRA] No weights loaded (path missing). Using injected adapters with random init.")

            # Apply LoRA scaling at inference (multiply delta by lora_scale)
            try:
                scaled = 0
                for mod in motion_model.diffusion.model.modules():
                    if isinstance(mod, lora_mod.LoRALinear) and mod.r > 0:
                        base_scale = (mod.alpha / mod.r) if mod.r > 0 else 0.0
                        mod.scaling = float(base_scale * args.lora_scale)
                        scaled += 1
                if scaled > 0:
                    print(f"[LoRA] Applied inference scale x{args.lora_scale:g} to {scaled} LoRA layers.")
            except Exception as e:
                print(f"[LoRA] Scaling apply failed: {e}")
        except Exception as e:
            print(f"[LoRA] Injection failed: {e}")

    # Column indices for pelvis tracking
    pelvis_tx_idx = model_cols_full.index('pelvis_tx') if 'pelvis_tx' in model_cols_full else None
    pelvis_ty_idx = model_cols_full.index('pelvis_ty') if 'pelvis_ty' in model_cols_full else None
    pelvis_tz_idx = model_cols_full.index('pelvis_tz') if 'pelvis_tz' in model_cols_full else None

    def build_mask_and_guidance_for_window(state_true_np: np.ndarray, previous_ptx_end=None):
        # Normalize state_true and prepare masks/guidance for a window slice
        state_true_local = torch.from_numpy(state_true_np).clone()  # [T,C]
        C_local = state_true_local.shape[1]
        mask_local = torch.zeros((1, T, C_local), dtype=torch.float32)
        # Kinematic/kinetic column groups
        kin_cols_local = [i for i, col in enumerate(model_cols_full) if 'force' not in col]

        def is_lower_body(col: str) -> bool:
            keys = ['hip_', 'knee', 'ankle']
            return any(k in col for k in keys)

        if args.preset == 'ds3':
            # In DS3 preset, hard-fix pelvis ty/tz/tx (tx now always constrained for stable speed control)
            if pelvis_tx_idx is not None:
                mask_local[0, :, pelvis_tx_idx] = 1.0  # Changed: always use hard constraint for pelvis_tx
            if pelvis_ty_idx is not None:
                mask_local[0, :, pelvis_ty_idx] = 1.0
            if pelvis_tz_idx is not None:
                mask_local[0, :, pelvis_tz_idx] = 1.0
        elif args.unlock == 'none':
            for idx in kin_cols_local:
                if idx in (pelvis_tx_idx, pelvis_ty_idx, pelvis_tz_idx):
                    continue
                mask_local[0, :, idx] = 1.0
        elif args.unlock == 'lower':
            for i, col in enumerate(model_cols_full):
                if 'force' in col:
                    mask_local[0, :, i] = 1.0
                    continue
                if i in (pelvis_tx_idx, pelvis_ty_idx, pelvis_tz_idx):
                    continue
                if is_lower_body(col):
                    continue
                mask_local[0, :, i] = 1.0
        elif args.unlock == 'all':
            for i, col in enumerate(model_cols_full):
                if 'force' in col:
                    mask_local[0, :, i] = 1.0

        # Prepare soft guidance for pelvis tx (and optional drift suppression)
        cur_norm_vx = state_true_local[:, pelvis_tx_idx].mean().item() if pelvis_tx_idx is not None else 0.0
        if args.target_speed is not None:
            target_norm_vx = float(args.target_speed) / float(args.height_m)
        else:
            target_norm_vx = float(args.scale) * float(cur_norm_vx)
        if pelvis_tx_idx is not None:
            # CRITICAL: pelvis_tx in model is VELOCITY (BH/frame), not position!
            # inverse_convert will do: position = cumsum(velocity * height_m) / sampling_fre
            
            if abs(cur_norm_vx) < 0.01:  # Treadmill walking (< 0.01 BH/s avg)
                print(f"[TREADMILL] Input is stationary (avg vx={cur_norm_vx:.4f} body-heights/s).")
                print(f"[TREADMILL] Setting constant velocity for target speed {args.target_speed} m/s...")
                
                # Model stores velocity (BH/s), same as training data preprocessing
                # inverse_convert does: position = cumsum(velocity * height_m) / Hz
                state_true_local[:, pelvis_tx_idx] = target_norm_vx
                
                # Expected travel over T frames at this velocity
                expected_travel = target_norm_vx * T / float(hz)  # BH
                
                if previous_ptx_end is not None:
                    target_ptx_end = previous_ptx_end + expected_travel
                else:
                    target_ptx_end = expected_travel
                
                print(f"[TREADMILL] Set velocity: {target_norm_vx:.4f} BH/s ({args.target_speed} m/s)")
                print(f"[TREADMILL] Expected travel: {expected_travel:.4f} BH over {T} frames")
                
            elif args.preset == 'ds3':
                # Overground data: scale existing velocity
                scale_factor = float(target_norm_vx) / float(cur_norm_vx)
                state_true_local[:, pelvis_tx_idx] = state_true_local[:, pelvis_tx_idx] * scale_factor
                target_ptx_end = None
            else:
                state_true_local[:, pelvis_tx_idx] = torch.tensor(target_norm_vx / float(hz), dtype=torch.float32).repeat(T)
                target_ptx_end = None
        if args.preset != 'ds3':
            if pelvis_ty_idx is not None:
                state_true_local[:, pelvis_ty_idx] = 0.0
            if pelvis_tz_idx is not None:
                state_true_local[:, pelvis_tz_idx] = 0.0

        state_true_norm_local = motion_model.normalizer.normalize(state_true_local).unsqueeze(0)

        value_diff_thd_local = torch.ones((C_local,), dtype=torch.float32) * 1e9
        value_diff_weight_local = torch.zeros((C_local,), dtype=torch.float32)
        if args.preset == 'ds3':
            # pelvis_tx now uses hard constraint (mask=1), no soft guidance needed
            for i, col in enumerate(model_cols_full):
                if ('_vel' in col) or ('force' in col):
                    value_diff_thd_local[i] = 999.0
                    value_diff_weight_local[i] = 0.0
                    continue
                if i == pelvis_tx_idx:
                    continue
                ref = state_true_norm_local[0, :, i]
                thd = (ref.max() - ref.min()) * 0.3
                value_diff_thd_local[i] = float(thd)
                value_diff_weight_local[i] = 1.0
        else:
            if pelvis_tx_idx is not None:
                value_diff_thd_local[pelvis_tx_idx] = 0.02
                value_diff_weight_local[pelvis_tx_idx] = 3.0
            if pelvis_ty_idx is not None:
                value_diff_thd_local[pelvis_ty_idx] = float(args.pelvis_drift_thd)
                value_diff_weight_local[pelvis_ty_idx] = float(args.pelvis_drift_weight)
            if pelvis_tz_idx is not None:
                value_diff_thd_local[pelvis_tz_idx] = float(args.pelvis_drift_thd)
                value_diff_weight_local[pelvis_tz_idx] = float(args.pelvis_drift_weight)
            if args.unlock in ('lower', 'all'):
                for i, col in enumerate(model_cols_full):
                    if 'force' in col:
                        continue
                    if args.unlock == 'lower' and not is_lower_body(col) and i not in (pelvis_tx_idx, pelvis_ty_idx, pelvis_tz_idx):
                        continue
                    if i == pelvis_tx_idx:
                        continue
                    value_diff_thd_local[i] = float(args.prior_thd)
                    value_diff_weight_local[i] = float(args.prior_weight)

        # Debug: Print mask and target values for pelvis_tx
        if args.preset == 'ds3' and pelvis_tx_idx is not None:
            print(f"[DEBUG] pelvis_tx mask: {mask_local[0, 0, pelvis_tx_idx]:.1f} (1.0 = hard constraint)")
            print(f"[DEBUG] pelvis_tx VELOCITY (normalized): {state_true_norm_local[0, 0, pelvis_tx_idx]:.6f}")
            # Denormalize to check actual velocity value
            denorm_vel = motion_model.normalizer.unnormalize(state_true_norm_local)[0, 0, pelvis_tx_idx].item()
            print(f"[DEBUG] pelvis_tx VELOCITY (denormalized): {denorm_vel:.4f} BH/s ({denorm_vel * args.height_m:.4f} m/s)")
            if target_ptx_end is not None:
                print(f"[DEBUG] Expected end position: {target_ptx_end:.4f} BH")

        return state_true_norm_local, mask_local, value_diff_thd_local, value_diff_weight_local, target_ptx_end

    # Note: per-window mask and guidance are constructed inside build_mask_and_guidance_for_window()

    # Prepare window plan
    if args.total_seconds is None:
        starts = [max(0, N - T)]  # last window by default
        target_len = T
    else:
        target_len = min(int(round(args.total_seconds * hz)), N)
        hop = max(1, T - int(args.overlap_frames))
        starts = list(range(0, max(1, target_len - T + 1), hop))
        # Ensure last window reaches target_len
        if starts and starts[-1] + T < target_len:
            starts.append(target_len - T)

    # Build naming tags (speed/preset/LoRA) for folder and files
    def tag_float(v: float) -> str:
        return f"{v:.2f}".replace('.', 'dot')

    in_base = Path(args.mot_in).stem
    if args.target_speed is not None:
        speed_tag = f"v{tag_float(args.target_speed)}ms"
    else:
        speed_tag = f"scale_{tag_float(args.scale)}"
    if args.preset == 'ds3':
        prior_tag = 'ds3' if not args.free_pelvis_tx else 'ds3_ptxFree'
    elif args.unlock == 'none':
        prior_tag = 'none'
    else:
        prior_tag = f"pw_{tag_float(args.prior_weight)}_th_{tag_float(args.prior_thd)}"
    if args.use_lora:
        if lora_epoch is not None:
            if args.lora_scale and abs(args.lora_scale - 1.0) > 1e-6:
                lora_tag = f"lora_e{lora_epoch}x{tag_float(args.lora_scale)}"
            else:
                lora_tag = f"lora_e{lora_epoch}"
        else:
            lora_tag = 'lora'
    else:
        lora_tag = 'nolora'

    folder_suffix = "__".join([prior_tag, speed_tag, lora_tag])
    out_dir = make_out_dir(args.mot_in, folder_suffix)
    out_len = (starts[-1] + T) if starts else T
    out_arr23 = np.zeros((out_len, 23), dtype=np.float32)
    out_weight = np.zeros((out_len,), dtype=np.float32)
    
    # Track pelvis_tx across windows for continuity
    last_pelvis_tx_end = None
    cumulative_pos_vec = pos_vec.copy()  # Start with initial position

    for i, s in enumerate(starts):
        e = s + T
        win_states = converted_df.iloc[s:e].to_numpy().astype(np.float32)
        win_time = time_s[s:e]

        state_true_norm, mask, value_diff_thd, value_diff_weight, target_ptx_end = build_mask_and_guidance_for_window(win_states, last_pelvis_tx_end)
        cond = torch.ones((1, 6), dtype=torch.float32)
        state_pred_list = motion_model.eval_loop(
            opt,
            state_true=state_true_norm,  # [1, T, C] normalized
            masks=mask,
            value_diff_thd=value_diff_thd,
            value_diff_weight=value_diff_weight,
            cond=cond,
            num_of_generation_per_window=1,
            mode='inpaint_ddim_guided'
        )
        
        model_states = state_pred_list[0]
        
        # Adjust pos_vec for multi-window continuity
        # For window N > 0, set pos_vec to where previous window ended
        current_pos_vec = cumulative_pos_vec if i == 0 else cumulative_pos_vec.copy()
        
        osim_states = inverse_convert_addb_state_to_model_input(
            model_states,
            model_cols_full,
            {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis', 'hip_r', 'hip_l', 'lumbar']},
            list(OSIM_DOF_ALL[:23] + KINETICS_ALL),
            pos_vec=current_pos_vec,
            height_m=torch.tensor([args.height_m], dtype=torch.float32),
            sampling_fre=hz,
        )
        arr23 = osim_states[0].detach().cpu().numpy()[:, :23]
        
        # DEBUG: Log pelvis_tx for this window
        pelvis_tx_col = OSIM_DOF_ALL[:23].index('pelvis_tx')
        ptx_generated = arr23[:, pelvis_tx_col]
        ptx_vel = np.diff(ptx_generated) / (1.0/hz) / args.height_m  # m/s
        print(f"[Window {i+1}/{len(starts)}] pelvis_tx: {ptx_generated[0]:.3f} → {ptx_generated[-1]:.3f} BH, "
              f"travel: {ptx_generated[-1]-ptx_generated[0]:.3f} BH ({(ptx_generated[-1]-ptx_generated[0])*args.height_m:.3f} m), "
              f"avg vel: {ptx_vel.mean()*args.height_m:.3f} m/s")
        
        # Update cumulative_pos_vec for next window to continue from where this window ended
        # This ensures position continuity across windows
        if i < len(starts) - 1:  # Not the last window
            travel_distance = ptx_generated[-1] - ptx_generated[0]  # BH
            cumulative_pos_vec[0] = ptx_generated[-1]  # Set next window's start to this window's end
            print(f"[Window {i+1}] Next window will start at: {cumulative_pos_vec[0]:.4f} BH")
        
        # Update last_pelvis_tx_end for next window (use TARGET, not generated)
        if target_ptx_end is not None:
            last_pelvis_tx_end = target_ptx_end
            print(f"[Window {i+1}] Target end position: {target_ptx_end:.4f} BH (for guidance)")



        # Overlap-add with linear crossfade
        if i == 0:
            out_arr23[s:e] = arr23
            out_weight[s:e] = 1.0
        else:
            ov = int(args.overlap_frames)
            ov = max(0, min(ov, T))
            non_ov_start = s + ov
            # overlap region
            if ov > 0:
                w = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
                w = w[:, None]
                out_arr23[s:s+ov] = (1.0 - w) * out_arr23[s:s+ov] + w * arr23[:ov]
                out_weight[s:s+ov] = 1.0
            # non-overlap tail
            out_arr23[non_ov_start:e] = arr23[ov:]
            out_weight[non_ov_start:e] = 1.0

    # Trim to target length (if total_seconds set)
    if args.total_seconds is not None:
        out_arr23 = out_arr23[:target_len]
        out_time = time_s[:target_len]
    else:
        out_time = time_s[starts[0]:starts[0]+T]

    out_df = pd.DataFrame(out_arr23, columns=OSIM_DOF_ALL[:23])
    out_df.insert(0, 'time', out_time)

    # File naming: simple trial name
    dur_tag = f"{int(round((len(out_df)-1)/hz))}s"
    base_name = f"{in_base}_{dur_tag}"

    csv_path = out_dir / f"{base_name}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    if args.export_mot:
        tools_dir = PROJECT_ROOT / 'tools'
        sys.path.insert(0, str(tools_dir))
        try:
            from export_to_opensim_mot import write_coordinates_mot
            mot_path = out_dir / f"{base_name}.mot"
            write_coordinates_mot(out_df, str(mot_path))
        finally:
            try:
                sys.path.remove(str(tools_dir))
            except Exception:
                pass
        print(f"[SAVE] {mot_path}")

    # Interactive comparison plot (input vs generated) and save image
    if _HAVE_MPL:
        try:
            # Use the yaw-aligned input states for fair comparison
            L = len(out_df)
            in_aligned_df = pd.DataFrame(osim_states_aligned[:L, :23], columns=OSIM_DOF_ALL[:23])

            # Define channels to plot: (right_name, left_name, title)
            plot_specs = [
                ('hip_flexion_r', 'hip_flexion_l', 'Hip (sagittal)'),
                ('knee_angle_r', 'knee_angle_l', 'Knee'),
                ('ankle_angle_r', 'ankle_angle_l', 'Ankle (sagittal)'),
                ('pelvis_tilt', None, 'Pelvis tilt (sagittal)'),
            ]

            plt.ion()
            fig, axes = plt.subplots(len(plot_specs), 1, sharex=True, figsize=(12, 10))
            if len(plot_specs) == 1:
                axes = [axes]
            t = out_df['time'].values
            for ax, (r_name, l_name, title) in zip(axes, plot_specs):
                # Right side
                if r_name in in_aligned_df.columns and r_name in out_df.columns:
                    ax.plot(t, in_aligned_df[r_name].values, color='C0', linestyle='-', linewidth=1.6, label='R input')
                    ax.plot(t, out_df[r_name].values, color='C0', linestyle='--', linewidth=1.4, label='R gen')
                # Left side
                if l_name is not None and l_name in in_aligned_df.columns and l_name in out_df.columns:
                    ax.plot(t, in_aligned_df[l_name].values, color='C1', linestyle='-', linewidth=1.6, label='L input')
                    ax.plot(t, out_df[l_name].values, color='C1', linestyle='--', linewidth=1.4, label='L gen')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', ncol=2, fontsize=8)
            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(f"Input vs Generated ({speed_tag}, {prior_tag}{' + LoRA' if args.use_lora else ''})", y=0.98)
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])

            img_path = out_dir / f"{base_name}_compare.png"
            fig.savefig(img_path, dpi=150)
            print(f"[SAVE] {img_path}")
            try:
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass
        except Exception as e:
            print(f"[plot] skipped: {e}")
    else:
        print("[plot] matplotlib not available; skipped interactive plot.")

    # Save run metadata for full reproducibility
    meta = {
        'input_mot': os.path.abspath(args.mot_in),
        'subject_height_m': float(args.height_m),
        'sampling_rate_hz': int(hz),
        'window_len': int(T),
        'total_seconds': None if args.total_seconds is None else float(args.total_seconds),
        'overlap_frames': int(args.overlap_frames),
        'preset': args.preset,
        'unlock': args.unlock,
        'prior_weight': float(args.prior_weight),
        'prior_thd': float(args.prior_thd),
        'pelvis_drift_weight': float(args.pelvis_drift_weight),
        'pelvis_drift_thd': float(args.pelvis_drift_thd),
        'target_speed_mps': None if args.target_speed is None else float(args.target_speed),
        'scale': float(args.scale),
        'guidance': {
            'n_guided_steps': int(opt.n_guided_steps),
            'guidance_lr': float(opt.guidance_lr),
            'guide_x_start_begin': int(opt.guide_x_start_the_beginning_step),
            'guide_x_start_end': int(opt.guide_x_start_the_end_step),
        },
        'lora': {
            'used': bool(lora_used or args.use_lora),
            'ckpt': None if not args.lora_ckpt else os.path.abspath(args.lora_ckpt),
            'epoch': None if lora_epoch is None else int(lora_epoch),
            'experiment': lora_exp,
            'injection': {'r': 16, 'alpha': 32, 'dropout': 0.05, 'include_mha_out_proj': True} if args.use_lora else None,
            'scale': float(args.lora_scale) if args.use_lora else None,
        },
        'output': {
            'dir': str(out_dir),
            'csv': str(csv_path),
            'mot': str(mot_path) if args.export_mot else None,
        },
        'tags': {
            'speed_tag': speed_tag,
            'prior_tag': prior_tag,
            'lora_tag': lora_tag,
            'duration_tag': dur_tag,
        },
    }
    try:
        with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        # Also write a short human-readable summary
        summary_lines = [
            f"Input: {meta['input_mot']}",
            f"Preset: {meta['preset']} | Unlock: {meta['unlock']}",
            f"Speed: {'target '+str(meta['target_speed_mps'])+' m/s' if meta['target_speed_mps'] is not None else 'scale '+str(meta['scale'])}",
            f"LoRA: {'ON' if meta['lora']['used'] else 'OFF'}{(' (exp='+str(lora_exp)+', epoch='+str(lora_epoch)+')') if meta['lora']['used'] and lora_epoch is not None else ''}",
            f"Sampling rate: {hz} Hz | Window: {T} | Overlap: {args.overlap_frames}",
            f"Guidance: steps={opt.n_guided_steps}, lr={opt.guidance_lr}, begin={opt.guide_x_start_the_beginning_step}, end={opt.guide_x_start_the_end_step}",
            f"Files: {csv_path.name}{(', '+Path(meta['output']['mot']).name) if meta['output']['mot'] else ''}",
        ]
        (out_dir / 'conditions.txt').write_text("\n".join(summary_lines), encoding='utf-8')
    except Exception as e:
        print(f"[meta] failed to write metadata: {e}")

    print(f"\n[DONE] Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
