#!/usr/bin/env python3
"""
Generate masked (missing) data files for inpainting experiments (HDF5 only).

Environment: gaitdyn2

Features (automatic mode):
- Accept a path to a data file (.h5) or a directory (will pick first matching file).
- Print the top-level and second-level headers/groups to the terminal so you can inspect structure.
- Select a trial automatically (first matching trial); you can override with --trial-idx or --trial-name.
- Mask (replace with NaN) selected headers for that trial and save a masked HDF5 file suitable for inpainting.

Notes:
- This script only supports .h5 files via h5py.
- For .b3d files, use S02_generate_missed_data_b3d.py (requires nimble environment).

Usage examples (automatic):
  python S01_generate_missed_data.py --path data/LD/subject01

To run in automatic mode and accept defaults simply pass --path. Use --trial-idx or --trial-name to override.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import shutil
import datetime
from typing import List, Optional, Tuple

import numpy as np

# =========================
# Configuration - Customize headers to mask here
# =========================
IK_DATA_HEADERS = [
    # Pelvis (6 DOF)
    "pelvis_tx", "pelvis_ty", "pelvis_tz",
    "pelvis_list", "pelvis_rotation", "pelvis_tilt",
    # Lumbar (3 DOF)
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    # Hip left (3 DOF)
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    # Hip right (3 DOF)
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    # Knee (2 DOF)
    "knee_angle_l", "knee_angle_r",
    # Ankle (2 DOF)
    "ankle_angle_l", "ankle_angle_r",
    # Subtalar (2 DOF)
    "subtalar_angle_l", "subtalar_angle_r",
    # MTP (2 DOF)
    "mtp_angle_l", "mtp_angle_r",
]  # Total: 23 DOF

GRF_HEADERS = [
    # Left foot - cop and force (no moment)
    "grf_measured/left/cop/x", "grf_measured/left/cop/y", "grf_measured/left/cop/z",
    "grf_measured/left/force/Fx", "grf_measured/left/force/Fy", "grf_measured/left/force/Fz",
    # Right foot - cop and force (no moment)
    "grf_measured/right/cop/x", "grf_measured/right/cop/y", "grf_measured/right/cop/z",
    "grf_measured/right/force/Fx", "grf_measured/right/force/Fy", "grf_measured/right/force/Fz",
]  # Total: 12 GRF channels

# Combine all headers to mask
DEFAULT_HEADERS = IK_DATA_HEADERS + GRF_HEADERS


def find_first_data_file(path: Path) -> Optional[Path]:
    """Find the first .h5 file under path (if path is a dir)."""
    if path.is_file():
        return path
    for p in path.rglob('*.h5'):
        return p
    return None
def print_h5_structure(h5, max_children=10):
	"""Print two levels of the h5 structure to stdout."""
	print('HDF5 structure (top-level -> second-level):')
	for key in h5.keys():
		print(f' - {key}')
		try:
			node = h5[key]
			# list up to max_children
			children = list(node.keys())[:max_children]
			for c in children:
				print(f'    - {c}')
		except Exception:
			# not a group
			continue


def pick_h5_trial(h5) -> Tuple[str, object]:
	"""Heuristic: find a subgroup that looks like a trial (contains datasets). 
	Prioritize MoCap paths over IMU/sensor data. Returns path and group."""
	import h5py
	# Search for groups with dataset children and with names containing common trial substrings
	candidates = []
	
	def visit_func(name, obj):
		# if it's a group and has datasets
		if isinstance(obj, h5py.Group):
			has_dataset = any(isinstance(obj[k], h5py.Dataset) for k in obj.keys())
			if has_dataset:
				candidates.append(name)
	
	h5.visititems(visit_func)
	
	# First priority: paths containing 'MoCap' or 'ik_data'
	for c in candidates:
		if any(s in c for s in ('MoCap', 'ik_data')):
			return c, h5[c]
	
	# Second priority: paths with 'trial', 'walk', 'level', etc. but not 'imu'
	for c in candidates:
		if any(s in c.lower() for s in ('trial', 'walk', 'level', 'asym', 'cadence')) and 'imu' not in c.lower():
			return c, h5[c]
	
	# Fallback: any candidate not containing 'imu'
	for c in candidates:
		if 'imu' not in c.lower():
			return c, h5[c]
	
	# Last resort: first candidate or root
	if candidates:
		return candidates[0], h5[candidates[0]]
	return '/', h5


# Note: DEFAULT_HEADERS defined at top of file after imports


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


def copy_h5_with_masks(src_path: Path, dst_path: Path, trial_key: str, headers_to_mask: List[str], mask_mode: str = 'full', mask_length: Optional[int] = None):
	import h5py

	with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
		def recurse_copy(snode, dnode, path=''):
			for name, item in snode.items():
				if isinstance(item, h5py.Group):
					ng = dnode.create_group(name)
					# copy attributes
					for k, v in item.attrs.items():
						ng.attrs[k] = v
					recurse_copy(item, ng, path + '/' + name)
				else:
					# dataset
					ds_path = (path + '/' + name).lstrip('/')
					# Check if this dataset should be masked
					should_mask = False
					
					# Check 1: direct dataset name match (for ik_data headers)
					if trial_key != '/':
						if ds_path.startswith(trial_key.lstrip('/')) and name in headers_to_mask:
							should_mask = True
					else:
						if name in headers_to_mask:
							should_mask = True
					
					# Check 2: hierarchical path match (for grf_measured paths like "grf_measured/left/cop/x")
					for header in headers_to_mask:
						if '/' in header:  # hierarchical header
							# Check if current ds_path ends with this header path
							if ds_path.endswith(header) or ds_path.endswith(header.lstrip('/')):
								should_mask = True
								break

					data = item[()]
					if should_mask:
						print(f'Masking dataset: {ds_path}')
						data = cast_to_float_and_mask(data, mode=mask_mode, mask_length=mask_length)
					# write to the correct parent group (dnode, not dst root)
					dnode.create_dataset(name, data=data, dtype=data.dtype)
					# copy attributes
					for k, v in item.attrs.items():
						dnode[name].attrs[k] = v

		# If trial_key is a nested path, we want to preserve structure; copy top-level groups
		for k in src.keys():
			if isinstance(src[k], h5py.Group):
				g = dst.create_group(k)
				for ak, av in src[k].attrs.items():
					g.attrs[ak] = av
				recurse_copy(src[k], g, path='/' + k)
			else:
				# top-level dataset
				data = src[k][()]
				if k in headers_to_mask and trial_key == '/':
					print(f'Masking top-level dataset: {k}')
					data = cast_to_float_and_mask(data, mode=mask_mode, mask_length=mask_length)
				dst.create_dataset(k, data=data, dtype=data.dtype)
				for ak, av in src[k].attrs.items():
					dst[k].attrs[ak] = av


def handle_h5(path: Path, args):
	import h5py

	with h5py.File(path, 'r') as h5:
		print_h5_structure(h5)
		trial_key, trial_group = pick_h5_trial(h5)
		print(f'Automatically selected trial/group: {trial_key}')
		# list available headers under that trial
		hdrs = []
		if hasattr(trial_group, 'keys'):
			hdrs = list(trial_group.keys())
		else:
			# trial_key may be '/', list top-level datasets
			hdrs = list(h5.keys())
		print('Headers/datasets under selected trial (sample):')
		for h in hdrs[:200]:
			print('  -', h)

	# Build destination path
	stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	dst = path.with_name(path.stem + f'_masked_{stamp}.h5')
	# Use provided headers list or default
	headers = args.headers if args.headers else DEFAULT_HEADERS
	copy_h5_with_masks(path, dst, trial_key, headers, mask_mode=args.mask_mode, mask_length=args.mask_length)
	print('Saved masked file to', dst)


def parse_headers_arg(s: Optional[str]) -> Optional[List[str]]:
	if not s:
		return None
	return [x.strip() for x in s.split(',') if x.strip()]


def main():
	ap = argparse.ArgumentParser(description='Generate masked data for inpainting (HDF5 only)')
	ap.add_argument('--path', '-p', required=True, help='File or directory containing data (.h5)')
	ap.add_argument('--trial-idx', type=int, default=None, help='Trial index to pick (overrides automatic selection)')
	ap.add_argument('--trial-name', type=str, default=None, help='Trial name to pick (overrides automatic selection)')
	ap.add_argument('--headers', type=str, default=None, help='Comma-separated header names to mask (default: use built-in list)')
	ap.add_argument('--mask-mode', type=str, default='full', choices=['full', 'temporal'], help='Mask full signals or temporal window')
	ap.add_argument('--mask-length', type=int, default=None, help='If mask-mode=temporal, number of frames to mask')
	args = ap.parse_args()

	p = Path(args.path)
	f = find_first_data_file(p)
	if f is None:
		print('No .h5 file found at', p)
		sys.exit(2)

	args.headers = parse_headers_arg(args.headers)

	print('Selected file:', f)
	if f.suffix.lower() == '.h5':
		handle_h5(f, args)
	else:
		print('Unsupported file type:', f.suffix, '(only .h5 supported)')


if __name__ == '__main__':
	main()

