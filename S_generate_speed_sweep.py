#!/usr/bin/env python3
"""
Gait generation across various target speeds with stronger, time-varying soft guidance.

- Loads the pretrained diffusion model checkpoint (example_usage/GaitDynamicsDiffusion.pt)
- Optional LoRA injection/weights
- For each target speed, constructs a time-varying guidance:
  * Forward pelvis_tx velocity set to target v/height
  * Reduce vertical/lateral drift (pelvis_ty, pelvis_tz ~ 0)
  * Encourage alternating hip flexion velocities (sinusoid, opposite phase L/R)
- Generates from noise and saves per-speed CSVs (OSIM 23-DoF) under previews/speed_sweep/<timestamp>/
- Optionally exports .mot files alongside CSVs

Usage examples:
  python3 S_generate_speed_sweep.py --speeds 0.8 1.2 1.6 2.0 --cycles 3 --export-mot
  python3 S_generate_speed_sweep.py --speeds 1.0 1.4 --use-lora \
    --lora-ckpt runs/adapter_train/lora_crouch_highq_1000steps_34656644/weights/epoch-170-lora.pt --export-mot
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from consts import OSIM_DOF_ALL, JOINTS_3D_ALL, KINETICS_ALL, MODEL_STATES_COLUMN_NAMES_NO_ARM
from model.model import MotionModel
from model.utils import inverse_convert_addb_state_to_model_input


def build_opt(window_len: int, target_hz: int = 100):
    opt = SimpleNamespace()
    opt.with_arm = False
    opt.with_kinematics_vel = True

    # Columns: OSIM 23 DoFs + kinetics
    opt.osim_dof_columns = list(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    # Joints with 3 DoF used during training
    opt.joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
    # Base model-state columns
    opt.model_states_column_names = list(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    # Add 3D joint angular velocities
    for joint_name, joints_with_3_dof in opt.joints_3d.items():
        opt.model_states_column_names += [f"{joint_name}_{axis}_angular_vel" for axis in ['x', 'y', 'z']]
    # Add kinematics velocities for non-force columns, excluding pelvis_* and existing *_vel and 6v indices
    if opt.with_kinematics_vel:
        opt.model_states_column_names += [
            f"{col}_vel" for col in opt.model_states_column_names
            if not any(term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5'])
        ]

    opt.window_len = int(window_len)
    opt.target_sampling_rate = int(target_hz)
    opt.batch_size_inference = 1
    opt.checkpoint = os.path.join(PROJECT_ROOT, 'example_usage', 'GaitDynamicsDiffusion.pt')

    # Guidance schedule
    opt.guide_x_start_the_end_step = 950
    opt.guide_x_start_the_beginning_step = 100
    opt.n_guided_steps = 1
    opt.guidance_lr = 0.05
    return opt


def ensure_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_out_dir() -> Path:
    out = Path(PROJECT_ROOT) / 'previews' / 'speed_sweep' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    return out


def inject_and_load_lora_if_needed(model_module, use_lora: bool, lora_ckpt: str = '', r: int = 16, alpha: int = 32, dropout: float = 0.05):
    if not use_lora:
        return False
    try:
        from lora.lora import inject_lora, load_lora_weights
    except Exception as e:
        print(f"[LoRA] Failed to import: {e}")
        return False

    replaced, _ = inject_lora(model_module, r=r, alpha=alpha, dropout=dropout, include_mha_out_proj=True)
    print(f"[LoRA] Injected into {replaced} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")
    if lora_ckpt and os.path.exists(lora_ckpt):
        load_lora_weights(model_module, lora_ckpt)
        print(f"[LoRA] Loaded weights: {lora_ckpt}")
        return True
    else:
        print("[LoRA] No weights loaded (path missing). Using injected adapters with random init.")
        return True


def estimate_cadence_hz(speed_mps: float) -> float:
    """Crude cadence estimate (steps per second) from speed. Tuned for walking ranges.
    Typical: ~1.6 Hz @ 0.8 m/s, ~1.8 Hz @ 1.2 m/s, ~2.0 Hz @ 1.6 m/s.
    """
    # Simple linear fit with clipping
    f = 1.4 + 0.35 * (speed_mps - 1.0)  # 1.05 @ 0.0, 1.75 @ 2.0
    return float(np.clip(f, 1.2, 2.4))


def build_time_varying_guidance(opt, T: int, C: int, col_names: List[str], speed_mps: float, height_m: float) -> Dict[str, torch.Tensor]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guide_states = torch.zeros((T, C), dtype=torch.float32, device=device)

    # Look up indices if present
    def idx(name: str):
        try:
            return col_names.index(name)
        except ValueError:
            return None

    pelvis_tx_idx = idx('pelvis_tx')  # forward translation (acts as forward velocity channel in training)
    pelvis_ty_idx = idx('pelvis_ty')
    pelvis_tz_idx = idx('pelvis_tz')
    hip_r_vel_idx = idx('hip_flexion_r_vel')
    hip_l_vel_idx = idx('hip_flexion_l_vel')

    # Base: forward speed guidance
    target_vel_norm = float(speed_mps) / float(height_m)
    if pelvis_tx_idx is not None:
        guide_states[:, pelvis_tx_idx] = target_vel_norm

    # Reduce vertical/lateral drift
    if pelvis_ty_idx is not None:
        guide_states[:, pelvis_ty_idx] = 0.0
    if pelvis_tz_idx is not None:
        guide_states[:, pelvis_tz_idx] = 0.0

    # Alternating hip flexion velocities (sinusoid, opposite phase)
    f = estimate_cadence_hz(speed_mps)
    t = torch.arange(T, device=device) / float(opt.target_sampling_rate)
    omega = 2 * np.pi * f
    amp = 1.5  # rad/s amplitude target in model-state velocity space
    s = torch.sin(omega * t)
    if hip_r_vel_idx is not None:
        guide_states[:, hip_r_vel_idx] = amp * s
    if hip_l_vel_idx is not None:
        guide_states[:, hip_l_vel_idx] = -amp * s

    # Normalize guidance to model's normalized space
    # Note: normalizer is applied in caller where MotionModel is available
    return {
        'guide_states': guide_states,
        'indices': {
            'pelvis_tx': pelvis_tx_idx,
            'pelvis_ty': pelvis_ty_idx,
            'pelvis_tz': pelvis_tz_idx,
            'hip_r_vel': hip_r_vel_idx,
            'hip_l_vel': hip_l_vel_idx,
        }
    }


@torch.no_grad()
def generate_one(opt, motion_model: MotionModel, speed_mps: float, height_m: float = 1.75) -> pd.DataFrame:
    device = motion_model.diffusion.betas.device
    T = opt.window_len
    C = motion_model.repr_dim

    col_names: List[str] = opt.model_states_column_names
    g = build_time_varying_guidance(opt, T, C, col_names, speed_mps, height_m)
    guide_states = g['guide_states']

    # Normalize guidance using the checkpoint's normalizer
    guide_states_norm = motion_model.normalizer.normalize(guide_states.clone().detach().cpu()).to(device)

    # Soft guidance masks and per-channel thresholds/weights
    mask = torch.zeros((1, T, C), dtype=torch.float32, device=device)
    value = guide_states_norm.unsqueeze(0)

    value_diff_thd = torch.ones((C,), dtype=torch.float32, device=device) * 1e9
    value_diff_weight = torch.zeros((C,), dtype=torch.float32, device=device)

    idxs = g['indices']
    if idxs['pelvis_tx'] is not None:
        value_diff_thd[idxs['pelvis_tx']] = 0.02
        value_diff_weight[idxs['pelvis_tx']] = 3.0
    if idxs['pelvis_ty'] is not None:
        value_diff_thd[idxs['pelvis_ty']] = 0.05
        value_diff_weight[idxs['pelvis_ty']] = 0.5
    if idxs['pelvis_tz'] is not None:
        value_diff_thd[idxs['pelvis_tz']] = 0.05
        value_diff_weight[idxs['pelvis_tz']] = 0.5
    if idxs['hip_r_vel'] is not None:
        value_diff_thd[idxs['hip_r_vel']] = 0.3
        value_diff_weight[idxs['hip_r_vel']] = 0.8
    if idxs['hip_l_vel'] is not None:
        value_diff_thd[idxs['hip_l_vel']] = 0.3
        value_diff_weight[idxs['hip_l_vel']] = 0.8

    cond = torch.ones((1, 6), dtype=torch.float32, device=device)
    constraint = {
        'mask': mask,
        'value': value,
        'value_diff_thd': value_diff_thd,
        'value_diff_weight': value_diff_weight,
        'cond': cond,
    }

    # Sample from noise with DDIM + soft guidance
    shape = (1, T, C)
    samples_norm = motion_model.diffusion.generate_samples(
        shape,
        motion_model.normalizer,
        opt,
        mode='inpaint_ddim_guided',
        constraint=constraint,
    )

    model_states = samples_norm
    osim_dofs = inverse_convert_addb_state_to_model_input(
        model_states,
        opt.model_states_column_names,
        opt.joints_3d,
        opt.osim_dof_columns,
        pos_vec=[0, 0, 0],
        height_m=torch.tensor([height_m], dtype=torch.float32),
    )
    osim_arr = osim_dofs[0].detach().cpu().numpy()
    df = pd.DataFrame(osim_arr[:, :23], columns=OSIM_DOF_ALL[:23])
    t = np.arange(T) / float(opt.target_sampling_rate)
    df.insert(0, 'time', t)
    return df


def export_to_mot_if_requested(df: pd.DataFrame, out_csv: Path, export_mot: bool):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")
    if export_mot:
        # Import the converter from tools
        tools_dir = os.path.join(PROJECT_ROOT, 'tools')
        sys.path.insert(0, tools_dir)
        try:
            from export_to_opensim_mot import write_coordinates_mot
            write_coordinates_mot(df, str(out_csv.with_suffix('.mot')))
        finally:
            try:
                sys.path.remove(tools_dir)
            except Exception:
                pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--speeds', type=float, nargs='+', default=[0.8, 1.2, 1.6, 2.0], help='Target treadmill speeds (m/s)')
    parser.add_argument('--cycles', type=int, default=3, help='Approximate gait cycles to cover (controls window length)')
    parser.add_argument('--height-m', type=float, default=1.75, help='Subject height (m) used for normalization')
    parser.add_argument('--use-lora', action='store_true', help='Inject and load LoRA adaptor before sampling')
    parser.add_argument('--lora-ckpt', type=str, default='', help='Path to LoRA weights (*.pt)')
    parser.add_argument('--export-mot', action='store_true', help='Also export each CSV to an OpenSim .mot')
    args = parser.parse_args()

    out_dir = make_out_dir()
    meta = {
        'speeds_mps': args.speeds,
        'height_m': args.height_m,
        'cycles': args.cycles,
        'used_lora': bool(args.use_lora),
        'lora_ckpt': args.lora_ckpt,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    # Build model once; we'll rebuild opt per-speed to adjust window length via cadence
    base_opt = build_opt(window_len=150, target_hz=100)
    motion_model = MotionModel(base_opt)
    motion_model.eval()

    if args.use_lora:
        inject_and_load_lora_if_needed(motion_model.diffusion.model, True, args.lora_ckpt)

    # Generate per speed
    for v in args.speeds:
        f = estimate_cadence_hz(v)
        period_frames = max(20, int(round(base_opt.target_sampling_rate / f)))
        T = int(args.cycles * period_frames)
        opt = build_opt(window_len=T, target_hz=base_opt.target_sampling_rate)
        # Reuse the same MotionModel instance; the diffusion uses opt for schedule, not shape-dependent params
        df = generate_one(opt, motion_model, speed_mps=v, height_m=args.height_m)
        name = f"gen_sweep_{v:.2f}ms_{T}f"
        export_to_mot_if_requested(df, out_dir / f"{name}.csv", export_mot=args.export_mot)

    print(f"\n[DONE] Saved {len(args.speeds)} samples to: {out_dir}")


if __name__ == '__main__':
    main()
