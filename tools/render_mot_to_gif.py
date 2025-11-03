#!/usr/bin/env python3
"""
Render an OpenSim Coordinates .mot (23-DoF lower-body) to a headless GIF using nimble FK + Matplotlib.

Example:
  python3 tools/render_mot_to_gif.py \
    --mot previews/samples/20251103_131600/gen_speed_1.20ms.mot \
    --osim example_usage/example_opensim_model.osim \
    --out previews/samples/20251103_131600/gen_speed_1.20ms.gif
"""

import os
import sys
from pathlib import Path
import argparse
import re
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from consts import OSIM_DOF_ALL
from data.osim_fk import forward_kinematics


def read_mot(filepath: str):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Find endheader
    end_idx = None
    in_degrees = None
    nrows = None
    for i, line in enumerate(lines):
        if line.lower().startswith('inDegrees'.lower()):
            in_degrees = ('yes' in line.lower()) and ('no' not in line.lower())
        if line.startswith('nRows='):
            try:
                nrows = int(line.split('=')[1].strip())
            except Exception:
                pass
        if 'endheader' in line:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError('endheader not found in .mot')
    header_line = lines[end_idx + 1]
    headers = re.split(r'[\t ,]+', header_line.strip())
    data_lines = lines[end_idx + 2:]
    data = []
    for dl in data_lines:
        if not dl.strip():
            continue
        data.append([float(x) for x in re.split(r'[\t ,]+', dl.strip())])
    data = np.array(data, dtype=np.float32)
    if nrows is not None and data.shape[0] != nrows:
        # not fatal, but warn
        pass
    return headers, data, bool(in_degrees)


def parse_time_and_coords(headers, data):
    # Expect 'time' + 23 OSIM DOFs (radians) in OSIM_DOF_ALL[:23]
    col_idx = {h: i for i, h in enumerate(headers)}
    if 'time' not in col_idx:
        raise ValueError(".mot missing 'time' column")
    must = list(OSIM_DOF_ALL[:23])
    missing = [c for c in must if c not in col_idx]
    if missing:
        raise ValueError(f".mot missing required columns: {missing}")
    t = data[:, col_idx['time']]
    coords = np.stack([data[:, col_idx[c]] for c in must], axis=1)  # [T,23]
    return t, coords


def parse_osim_to_offsets(osim_path: str):
    try:
        import nimblephysics as nimble
        from data.osim_fk import get_model_offsets
    except Exception as e:
        raise RuntimeError("nimblephysics is required for FK rendering; please install it") from e

    osim_path = os.path.abspath(osim_path)
    geom_dir = os.path.join(os.path.dirname(osim_path), 'Geometry')
    if not os.path.isdir(geom_dir):
        geom_dir = ''

    custom_osim = nimble.biomechanics.OpenSimParser.parseOsim(osim_path, geom_dir)
    skel = custom_osim.skeleton
    offsets = get_model_offsets(skel)
    return skel, offsets


def render_gif(t: np.ndarray, coords: np.ndarray, offsets: torch.Tensor, out_path: str, fps: int = None, dpi=120):
    # Prepare tensors
    pose = torch.from_numpy(coords).float()  # [T,23]
    # FK expects batch or [T,23]; we'll call per-frame for simplicity to keep memory small

    # Simple 2D sagittal view figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    lines = {
        'r_thigh': ax.plot([], [], 'o-', lw=2, c='#1f77b4')[0],
        'r_shank': ax.plot([], [], 'o-', lw=2, c='#1f77b4')[0],
        'r_foot': ax.plot([], [], 'o-', lw=2, c='#1f77b4')[0],
        'l_thigh': ax.plot([], [], 'o-', lw=2, c='#ff7f0e')[0],
        'l_shank': ax.plot([], [], 'o-', lw=2, c='#ff7f0e')[0],
        'l_foot': ax.plot([], [], 'o-', lw=2, c='#ff7f0e')[0],
        'pelvis': ax.plot([], [], 'ks', ms=4)[0],
    }

    # Determine bounds from a quick pass to keep axes stable
    with torch.no_grad():
        xs, ys = [], []
        for i in range(0, pose.shape[0], max(1, pose.shape[0] // 20)):
            _, joints, names, _ = forward_kinematics(pose[i:i+1, :], offsets)
            J = joints.detach().cpu().numpy()  # expected [11,1,3] or [11,3]
            J = np.squeeze(J)
            if J.ndim == 1:
                J = J.reshape(11, 3)
            xs.extend(J[:, 0]); ys.extend(J[:, 1])
        if xs and ys:
            pad = 0.2
            ax.set_xlim(min(xs)-pad, max(xs)+pad)
            ax.set_ylim(min(ys)-pad, max(ys)+pad)

    # Animation frames
    frames = []
    tmp_dir = Path(out_path).with_suffix('').as_posix() + '_frames'
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(pose.shape[0]):
        with torch.no_grad():
            _, joints, names, _ = forward_kinematics(pose[i:i+1, :], offsets)
            J = joints.detach().cpu().numpy()
            J = np.squeeze(J)
            if J.ndim == 1:
                J = J.reshape(11, 3)
            # joint indices per data.osim_fk: ['pelvis','hip_r','knee_r','ankle_r','calcn_r','mtp_r','hip_l','knee_l','ankle_l','calcn_l','mtp_l']
            P = {n: J[idx] for idx, n in enumerate(['pelvis','hip_r','knee_r','ankle_r','calcn_r','mtp_r','hip_l','knee_l','ankle_l','calcn_l','mtp_l'])}

            # Right
            lines['r_thigh'].set_data([P['hip_r'][0], P['knee_r'][0]], [P['hip_r'][1], P['knee_r'][1]])
            lines['r_shank'].set_data([P['knee_r'][0], P['ankle_r'][0]], [P['knee_r'][1], P['ankle_r'][1]])
            lines['r_foot'].set_data([P['ankle_r'][0], P['mtp_r'][0]], [P['ankle_r'][1], P['mtp_r'][1]])
            # Left
            lines['l_thigh'].set_data([P['hip_l'][0], P['knee_l'][0]], [P['hip_l'][1], P['knee_l'][1]])
            lines['l_shank'].set_data([P['knee_l'][0], P['ankle_l'][0]], [P['knee_l'][1], P['ankle_l'][1]])
            lines['l_foot'].set_data([P['ankle_l'][0], P['mtp_l'][0]], [P['ankle_l'][1], P['mtp_l'][1]])
            lines['pelvis'].set_data([P['pelvis'][0]], [P['pelvis'][1]])

            # Save frame
            frame_path = tmp_dir / f"frame_{i:05d}.png"
            fig.savefig(frame_path)
            frames.append(frame_path)

    # Save GIF via Pillow, estimate fps from time
    if fps is None:
        if len(t) >= 2:
            dt = float(np.median(np.diff(t)))
            fps = max(1, int(round(1.0 / dt)))
        else:
            fps = 30

    try:
        import imageio
        imgs = [imageio.v2.imread(fp) for fp in frames]
        imageio.mimsave(out_path, imgs, duration=1.0/float(fps))
    except Exception:
        # Fallback to PIL
        from PIL import Image
        imgs = [Image.open(fp) for fp in frames]
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000/float(fps)), loop=0)

    # Cleanup frames
    for fp in frames:
        try:
            os.remove(fp)
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    print(f"[GIF] {out_path} ({fps} fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mot', type=str, required=True)
    ap.add_argument('--osim', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    headers, data, in_deg = read_mot(args.mot)
    if in_deg:
        print('[WARN] Input .mot is in degrees; expected radians. Proceeding anyway.')
    t, coords = parse_time_and_coords(headers, data)
    _, offsets = parse_osim_to_offsets(args.osim)
    render_gif(t, coords, offsets, args.out)


if __name__ == '__main__':
    main()
