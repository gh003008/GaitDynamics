#!/usr/bin/env python3
"""
Unconditional gait generation with optional speed guidance and optional LoRA adaptor.

- Loads the pretrained diffusion model checkpoint (example_usage/GaitDynamicsDiffusion.pt)
- Optionally injects and loads a LoRA adaptor
- Generates from pure noise (no inpaint), using soft guidance on pelvis_tx velocity
- Saves outputs as CSV (OSIM 23-DoF) under previews/samples/<timestamp>/

Usage (examples):
  python3 S_generate_unconditional_speed.py --speeds 0.8 1.2 1.8
  python3 S_generate_unconditional_speed.py --speeds 1.2 --use-lora \
      --lora-ckpt runs/adapter_train/lora_crouch_highq_1000steps_34656644/weights/epoch-170-lora.pt
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List

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


def build_opt(window_len: int = 150, target_hz: int = 100):
    # Minimal opt object mimicking training-time column config (no CLI parsing)
    opt = SimpleNamespace()
    opt.with_arm = False
    opt.with_kinematics_vel = True

    # Columns: OSIM 23 DoFs + kinetics
    opt.osim_dof_columns = list(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    # Joints with 3 DoF used during training
    opt.joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
    # Base model-state columns
    opt.model_states_column_names = list(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    # Add 3D joint angular velocities (x,y,z) per training
    for joint_name, joints_with_3_dof in opt.joints_3d.items():
        opt.model_states_column_names += [f"{joint_name}_{axis}_angular_vel" for axis in ['x', 'y', 'z']]
    # Add kinematics velocities for non-force columns, excluding pelvis_* and existing *_vel and 6v indices
    if opt.with_kinematics_vel:
        opt.model_states_column_names += [
            f"{col}_vel" for col in opt.model_states_column_names
            if not any(term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5'])
        ]

    # Core inference options
    opt.window_len = int(window_len)
    opt.target_sampling_rate = int(target_hz)
    opt.batch_size_inference = 1
    # Paths
    opt.checkpoint = os.path.join(PROJECT_ROOT, 'example_usage', 'GaitDynamicsDiffusion.pt')
    # No baseline model needed here
    # Conditioning/guidance hyperparams
    # Enable soft guidance in the middle of the schedule
    opt.guide_x_start_the_end_step = 950
    opt.guide_x_start_the_beginning_step = 100
    opt.n_guided_steps = 1
    opt.guidance_lr = 0.05
    return opt


def ensure_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def make_out_dir() -> Path:
    out = Path(PROJECT_ROOT) / 'previews' / 'samples' / time.strftime('%Y%m%d_%H%M%S')
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

    # Inject LoRA into the diffusion's neural model (DanceDecoder)
    replaced, _ = inject_lora(model_module, r=r, alpha=alpha, dropout=dropout, include_mha_out_proj=True)
    print(f"[LoRA] Injected into {replaced} Linear layers (r={r}, alpha={alpha}, dropout={dropout}).")
    if lora_ckpt and os.path.exists(lora_ckpt):
        load_lora_weights(model_module, lora_ckpt)
        print(f"[LoRA] Loaded weights: {lora_ckpt}")
        return True
    else:
        print("[LoRA] No weights loaded (path missing). Using injected adapters with random init.")
        return True


@torch.no_grad()
def generate_one(opt, motion_model: MotionModel, speed_mps: float, height_m: float = 1.75) -> pd.DataFrame:
    """
    Generate a single window (T x 23 OSIM DoFs) from noise with soft guidance on pelvis_tx velocity.
    """
    device = motion_model.diffusion.betas.device
    T = opt.window_len
    C = motion_model.repr_dim

    # Build soft guidance targets in MODEL-STATE space (before normalization)
    guide_states = torch.zeros((T, C), dtype=torch.float32, device=device)
    col_names: List[str] = opt.model_states_column_names

    # Pelvis translational velocity columns exist in model-state space (normalized by height during preprocessing)
    try:
        pelvis_tx_idx = col_names.index('pelvis_tx')
    except ValueError:
        raise RuntimeError("'pelvis_tx' not found in model state columns; cannot guide speed.")

    target_vel_norm = float(speed_mps) / float(height_m)  # training used vel/height
    guide_states[:, pelvis_tx_idx] = target_vel_norm

    # Normalize the guidance to the model's normalized space
    guide_states_norm = motion_model.normalizer.normalize(guide_states.clone().detach().cpu()).to(device)

    # Unconditional mask (no hard inpaint): all zeros so the process is free-running
    mask = torch.zeros((1, T, C), dtype=torch.float32, device=device)
    value = guide_states_norm.unsqueeze(0)  # (1, T, C)

    # Soft guidance margins and weights (push pelvis_tx near target; allow others to float)
    value_diff_thd = torch.ones((C,), dtype=torch.float32, device=device) * 1e9  # large default = no pressure
    value_diff_weight = torch.zeros((C,), dtype=torch.float32, device=device)
    value_diff_thd[pelvis_tx_idx] = 0.02  # tight tolerance around target
    value_diff_weight[pelvis_tx_idx] = 3.0

    cond = torch.ones((1, 6), dtype=torch.float32, device=device)  # not currently used by the model

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
    )  # returns de-normalized MODEL-STATE space (T x C)

    # Convert to OSIM 23-DoF space for export (keep batch dim)
    # motion_model.normalizer.unnormalize already applied within generate_samples
    model_states = samples_norm  # (1, T, C)
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
    # add synthetic time for 100 Hz
    t = np.arange(T) / float(opt.target_sampling_rate)
    df.insert(0, 'time', t)
    return df


def save_csv(df: pd.DataFrame, out_dir: Path, name: str):
    fp = out_dir / f"{name}.csv"
    df.to_csv(fp, index=False)
    print(f"[SAVE] {fp}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--speeds', type=float, nargs='+', default=[1.2], help='Target treadmill speeds (m/s)')
    parser.add_argument('--height-m', type=float, default=1.75, help='Subject height (m) used for normalization')
    parser.add_argument('--window-len', type=int, default=150, help='Frames per sample (100Hz default)')
    parser.add_argument('--use-lora', action='store_true', help='Inject and load LoRA adaptor before sampling')
    parser.add_argument('--lora-ckpt', type=str, default='', help='Path to LoRA weights (*.pt)')
    args = parser.parse_args()

    opt = build_opt(window_len=args.window_len, target_hz=100)

    # Build model and load pretrained checkpoint (normalizer included)
    motion_model = MotionModel(opt)
    motion_model.eval()

    # Optionally inject/load LoRA
    if args.use_lora:
        inject_and_load_lora_if_needed(motion_model.diffusion.model, True, args.lora_ckpt)

    out_dir = make_out_dir()
    meta = {
        'speeds_mps': args.speeds,
        'height_m': args.height_m,
        'window_len': args.window_len,
        'used_lora': bool(args.use_lora),
        'lora_ckpt': args.lora_ckpt,
        'checkpoint': opt.checkpoint,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    for v in args.speeds:
        df = generate_one(opt, motion_model, speed_mps=v, height_m=args.height_m)
        save_csv(df, out_dir, f"gen_speed_{v:.2f}ms")

    print(f"\n[DONE] Saved {len(args.speeds)} samples to: {out_dir}")


if __name__ == '__main__':
    main()
