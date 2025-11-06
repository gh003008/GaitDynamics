#!/usr/bin/env python3
"""
Generate masked (missing) data files for inpainting experiments (B3D only).

Environment: nimble (별도 conda env 필요)
  - Python 3.8 or 3.9 recommended
  - nimblephysics (pip install nimblephysics, or build from source on Windows)
  - numpy, h5py

Features (automatic mode):
- Accept a path to a .b3d file or a directory (will pick first matching file).
- Read trial data from .b3d using nimblephysics SubjectOnDisk.
- Mask (replace with NaN) selected coordinate columns.
- Save masked trial as HDF5 file suitable for inpainting.

Notes:
- This script only supports .b3d files via nimblephysics.
- For .h5 files, use S01_generate_missed_data.py (requires gaitdyn2 environment).

Usage examples:
  python S02_generate_missed_data_b3d.py --path data/AB/vanderZee2022_Formatted_No_Arm/p9/p9.b3d --headers pelvis_tx,hip_flexion_r
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import datetime
from typing import List, Optional

import numpy as np

try:
    import nimblephysics as nimble
except ImportError as e:
    print('[ERROR] nimblephysics import failed. Install nimblephysics in the nimble environment.')
    print('  Recommended: conda create -n nimble python=3.8; conda activate nimble; pip install nimblephysics')
    print('  Error:', e)
    sys.exit(2)

# Add repo root to sys.path for consts import
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from consts import OSIM_DOF_ALL
except ImportError:
    # Fallback if consts not available
    OSIM_DOF_ALL = [
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
        'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
        'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
        'arm_flex_r', 'arm_add_r', 'arm_rot_r'
    ]


DEFAULT_HEADERS = [
    # Example joint angle channels (23 DOF). Replace these with your dataset's exact header names.
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
    'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'hip_flexion_l', 'hip_adduction_l',
    'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    # Example GRF / COP channels (3 each) — these won't be in .b3d pose data but included for reference
    'grf_r_x', 'grf_r_y', 'grf_r_z', 'cop_r_x', 'cop_r_y', 'cop_r_z',
    'grf_l_x', 'grf_l_y', 'grf_l_z', 'cop_l_x', 'cop_l_y', 'cop_l_z',
]


def find_first_b3d_file(path: Path) -> Optional[Path]:
    """Find the first .b3d file under path (if path is a dir)."""
    if path.is_file():
        return path
    for p in path.rglob('*.b3d'):
        return p
    return None


def cast_to_float_and_mask(arr: np.ndarray, mode: str = 'full', mask_length: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Cast array to float and mask according to mode.
    - full: entire array -> NaN
    - temporal: mask an interval of mask_length frames (random start)
    """
    a = np.array(arr, copy=True)
    try:
        a = a.astype(np.float64)
    except Exception:
        # fallback: convert via astype float
        a = a.astype(float)
    if mode == 'full':
        a[...] = np.nan
    elif mode == 'temporal':
        T = a.shape[0]
        if mask_length is None or mask_length <= 0 or mask_length >= T:
            a[...] = np.nan
        else:
            rng = np.random.RandomState(seed)
            start = rng.randint(0, T - mask_length + 1)
            if a.ndim == 1:
                a[start:start+mask_length] = np.nan
            else:
                a[start:start+mask_length, ...] = np.nan
    else:
        raise ValueError('Unknown mask mode')
    return a


def handle_b3d(path: Path, args):
    """Read .b3d trial, mask selected coordinate columns, save as HDF5."""
    import h5py

    subj = nimble.biomechanics.SubjectOnDisk(str(path))
    
    # pick trial index
    if args.trial_idx is not None:
        t_idx = int(args.trial_idx)
    else:
        # pick first trial
        t_idx = 0

    print(f'Reading trial index {t_idx} from {path.name}...')
    
    try:
        dt = subj.getTrialTimestep(t_idx)
        n = subj.getTrialLength(t_idx)
        trial_name = subj.getTrialName(t_idx)
        frames = subj.readFrames(t_idx, 0, n, False, True)
        
        if not frames or not frames[0].processingPasses:
            raise RuntimeError('No processing passes found in frames')
        
        pass0 = [fr.processingPasses[0] for fr in frames]
        poses = np.array([fr.pos for fr in pass0], dtype=np.float64)
        
        print(f'Trial: {trial_name}, frames: {n}, timestep: {dt}s')
        print(f'Pose shape: {poses.shape}')
        
    except Exception as e:
        print('[ERROR] Failed to read poses from .b3d:', e)
        return

    # Build an HDF5 to save masked trial (coordinates in OSIM order)
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    dst = path.with_name(path.stem + f'_trial{t_idx}_masked_{stamp}.h5')
    coords23 = list(OSIM_DOF_ALL[:23])
    
    # use headers list
    headers = args.headers if args.headers else DEFAULT_HEADERS

    with h5py.File(dst, 'w') as h5:
        grp = h5.create_group('trials')
        gtr = grp.create_group(trial_name)
        
        # write time
        T, D = poses.shape
        t = np.arange(T, dtype=np.float64) * float(dt)
        gtr.create_dataset('time', data=t)
        
        # write coords
        num_masked = 0
        for j, col in enumerate(coords23):
            if j >= D:
                break  # Don't go past available DOFs
            data = poses[:, j]
            if col in headers:
                print(f'  Masking coordinate column: {col}')
                data = cast_to_float_and_mask(data, mode=args.mask_mode, mask_length=args.mask_length)
                num_masked += 1
            gtr.create_dataset(col, data=data)

    print(f'Masked {num_masked} coordinate(s).')
    print(f'Saved masked file to: {dst}')


def parse_headers_arg(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser(description='Generate masked data for inpainting (B3D only)')
    ap.add_argument('--path', '-p', required=True, help='File or directory containing .b3d data')
    ap.add_argument('--trial-idx', type=int, default=None, help='Trial index to pick (default: 0)')
    ap.add_argument('--headers', type=str, default=None, help='Comma-separated coordinate names to mask (default: use built-in list)')
    ap.add_argument('--mask-mode', type=str, default='full', choices=['full', 'temporal'], help='Mask full signals or temporal window')
    ap.add_argument('--mask-length', type=int, default=None, help='If mask-mode=temporal, number of frames to mask')
    args = ap.parse_args()

    p = Path(args.path)
    f = find_first_b3d_file(p)
    if f is None:
        print('No .b3d file found at', p)
        sys.exit(2)

    args.headers = parse_headers_arg(args.headers)

    print('Selected file:', f)
    if f.suffix.lower() == '.b3d':
        handle_b3d(f, args)
    else:
        print('Unsupported file type:', f.suffix, '(only .b3d supported)')


if __name__ == '__main__':
    main()
