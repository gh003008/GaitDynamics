#!/usr/bin/env python3
"""
Inpaint missing variables with the pretrained diffusion model from a dataset window.

What it does:
- Loads the pretrained diffusion checkpoint (example_usage/GaitDynamicsDiffusion.pt)
- Loads one window from the dataset configured in machine_specific_config.json
- Treats kinematics as known (mask=1) and inpaints the rest (kinetics, velocities, etc.)
- Exports OSIM 23-DoF results as CSV and OpenSim .mot
- Optional: render a GIF using nimble FK (requires nimblephysics)

Usage (examples):
  # Simple run (fills kinetics from known kinematics on a random test window)
  python3 S_inpaint_from_dataset.py

  # Choose a specific trial by substring (e.g., "walking")
  python3 S_inpaint_from_dataset.py --trial-filter walking

  # Export .mot and render GIF
  python3 S_inpaint_from_dataset.py --export-mot --render \
    --osim example_usage/example_opensim_model.osim
"""

import os
import sys
import time
from pathlib import Path
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from consts import OSIM_DOF_ALL, JOINTS_3D_ALL, KINETICS_ALL, MODEL_STATES_COLUMN_NAMES_NO_ARM
from args import parse_opt as parse_train_opt
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from model.utils import inverse_convert_addb_state_to_model_input


def build_infer_opt(window_len: int = 150, target_hz: int = 100):
    # Minimal inference opt mirroring training-time column config
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

    # Guidance off for pure inpaint (hard mask enforcement)
    opt.guide_x_start_the_end_step = -1
    opt.guide_x_start_the_beginning_step = -1
    opt.n_guided_steps = 0
    opt.guidance_lr = 0.0

    # Pretrained diffusion checkpoint shipped with the repo
    opt.checkpoint = str(PROJECT_ROOT / 'example_usage' / 'GaitDynamicsDiffusion.pt')
    return opt


def make_out_dir() -> Path:
    out = PROJECT_ROOT / 'previews' / 'inpaint' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    return out


@torch.no_grad()
def inpaint_one(opt, motion_model: MotionModel, dataset: MotionDataset, trial_filter: str = None):
    # Choose windows: known kinematics (mask=1) so the model fills in the rest
    col_loc_to_unmask = dataset.opt.kinematic_diffusion_col_loc

    if trial_filter:
        windows = MotionDataset(
            data_path=dataset.data_path,
            train=False,
            normalizer=motion_model.normalizer,
            opt=dataset.opt,
            include_trials_shorter_than_window_len=True,
            specific_trial=trial_filter,
        ).get_one_win_from_the_end_of_each_trial(col_loc_to_unmask)
    else:
        windows = dataset.get_one_win_from_the_end_of_each_trial(col_loc_to_unmask)

    if not windows:
        raise RuntimeError('No windows found. Check dataset path and trial filter.')

    win = windows[0]
    state_true = win.pose.unsqueeze(0)   # [1, T, C], already normalized
    masks = win.mask.unsqueeze(0)        # [1, T, C]
    cond = torch.ones((1, 6), dtype=torch.float32)

    state_pred_list = motion_model.eval_loop(opt, state_true, masks, cond=cond, num_of_generation_per_window=1, mode='inpaint')
    state_pred = state_pred_list[0]      # [1, T, C] de-normalized (model-state space)

    # Convert to OSIM 23-DoF (batch-aware), add synthetic time
    T = state_pred.shape[1]
    osim = inverse_convert_addb_state_to_model_input(
        state_pred, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns,
        pos_vec=[0, 0, 0], height_m=torch.tensor([win.height_m], dtype=torch.float32)
    )
    arr = osim[0].detach().cpu().numpy()
    df = pd.DataFrame(arr[:, :23], columns=OSIM_DOF_ALL[:23])
    t = np.arange(T) / float(opt.target_sampling_rate)
    df.insert(0, 'time', t)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial-filter', type=str, default=None, help='Substring to match trial name (optional)')
    ap.add_argument('--export-mot', action='store_true', help='Also export OpenSim .mot')
    ap.add_argument('--render', action='store_true', help='Render a GIF from the .mot (requires nimblephysics)')
    ap.add_argument('--osim', type=str, default=str(PROJECT_ROOT / 'example_usage' / 'example_opensim_model.osim'), help='OpenSim model for rendering')
    args = ap.parse_args()

    # Use training args to resolve dataset root from machine_specific_config.json
    data_opt = parse_train_opt()
    data_root = data_opt.data_path_test  # use test split by default

    # Build inference opts and model
    opt = build_infer_opt(window_len=data_opt.window_len, target_hz=data_opt.target_sampling_rate)
    motion_model = MotionModel(opt)
    motion_model.eval()

    # Dataset (normalized using the model's normalizer)
    dataset = MotionDataset(
        data_path=data_root,
        train=False,
        normalizer=motion_model.normalizer,
        opt=data_opt,
        include_trials_shorter_than_window_len=True,
    )

    out_dir = make_out_dir()
    df = inpaint_one(data_opt, motion_model, dataset, trial_filter=args.trial_filter)

    # Save CSV and optionally .mot + GIF
    base = 'inpaint_kinematics_known'
    csv_path = out_dir / f'{base}.csv'
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    if args.export_mot:
        from tools.export_to_opensim_mot import write_coordinates_mot
        mot_path = out_dir / f'{base}.mot'
        write_coordinates_mot(df, str(mot_path))
        if args.render:
            from tools.render_mot_to_gif import main as render_main
            # Render CLI expects args; call via subprocess-like invocation
            import argparse as _argparse
            import sys as _sys
            _argv = [
                '--mot', str(mot_path),
                '--osim', args.osim,
                '--out', str(out_dir / f'{base}.gif'),
            ]
            _sys.argv = ['render_mot_to_gif.py'] + _argv
            render_main()

    print(f"\n[DONE] Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
