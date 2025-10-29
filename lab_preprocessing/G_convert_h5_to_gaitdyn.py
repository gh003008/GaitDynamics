#!/usr/bin/env python3
import os
import math
import json
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import yaml

# Converter: combined lab H5 -> standardized sensor dataset for adaptor training
# - Resamples to 100 Hz
# - IMU accel/gyro for back and thigh
# - Insoles: FSR(8) per foot, Fz per foot, COP(x,y,0) per foot
# - Treadmill keys (if present)
# - MoCap labels (if present): GRF and COP from forceplates
# - Metadata per trial
# Output: lab_preprocessed/combined_sensors.h5 with groups /S###/level_XXmps/trial_YY

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT_DIR, 'combined_data.h5')
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, 'lab_preprocessed', 'combined_sensors.h5')
DEFAULT_CONFIG = os.path.join(ROOT_DIR, 'lab_preprocessing', 'config', 'G_lab_h5_mapping.yaml')


def lowpass(data: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz is None or cutoff_hz <= 0:
        return data
    b, a = butter(order, cutoff_hz / (fs_hz / 2.0), btype='low')
    if data.ndim == 1:
        return filtfilt(b, a, data)
    return filtfilt(b, a, data, axis=0)


def resample_to(data: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    if src_hz == dst_hz:
        return data
    n_src = data.shape[0]
    x = np.linspace(0.0, 1.0, n_src)
    n_dst = int(round(n_src * dst_hz / src_hz))
    new_x = np.linspace(0.0, 1.0, n_dst)
    f = interp1d(x, data, axis=0, bounds_error=False, fill_value='extrapolate')
    return f(new_x)


def parse_axis_mapping(spec):
    # spec like ['+x','-y','+z'] -> indices and signs mapping from raw->target
    axis_to_idx = {'x': 0, 'y': 1, 'z': 2}
    idx = []
    sgn = []
    for token in spec:
        token = str(token).strip()
        if token[0] not in ['+','-']:
            raise ValueError(f"Axis token must start with +/-: {token}")
        sign = +1.0 if token[0] == '+' else -1.0
        base = token[1:]
        if base not in axis_to_idx:
            raise ValueError(f"Unknown axis in token {token}")
        idx.append(axis_to_idx[base])
        sgn.append(sign)
    return np.array(idx, dtype=int), np.array(sgn, dtype=float)


def apply_axis_mapping(arr3: np.ndarray, idx: np.ndarray, sgn: np.ndarray) -> np.ndarray:
    if arr3 is None:
        return None
    if arr3.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3, got {arr3.shape}")
    return arr3[..., idx] * sgn


def estimate_sampling_rate_from_time(time_array: np.ndarray) -> float:
    # time may be in seconds or milliseconds; we infer Hz from median diff
    diffs = np.diff(time_array.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 100.0
    dt = float(np.median(diffs))
    # try to detect ms scale
    if dt > 1.5:  # looks like ticks, not seconds
        # If these are integer ticks (e.g., 10ms = 10), try to infer
        # Heuristic: if median ~10, fs=100Hz; if ~8, ~125Hz; etc.
        if 8.0 <= dt <= 12.0:
            return 1000.0 / dt
        # otherwise assume already seconds per sample
    if dt > 0:
        return 1.0 / dt
    return 100.0


def compute_cop_from_fsr(fsr: np.ndarray, pad_xy: np.ndarray, clamp_radius: Optional[float], scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    fsr: (T,8) raw force units
    pad_xy: (8,2) foot-frame coordinates (m)
    returns:
      Fz: (T,) in arbitrary units scaled by 'scale'
      COP: (T,3) [x,y,0] in meters (foot frame), NaN when Fz <= 0
    """
    if fsr is None:
        return None, None
    force = (fsr * scale).astype(np.float64)
    Fz = np.sum(force, axis=1)
    # Avoid divide-by-zero
    eps = 1e-8
    weights = (force + eps)
    x = np.sum(weights * pad_xy[None, :, 0], axis=1) / np.sum(weights, axis=1)
    y = np.sum(weights * pad_xy[None, :, 1], axis=1) / np.sum(weights, axis=1)
    cop = np.stack([x, y, np.zeros_like(x)], axis=1)
    if clamp_radius and clamp_radius > 0:
        r = np.linalg.norm(cop[:, :2], axis=1)
        mask = r > clamp_radius
        if mask.any():
            cop[mask, :2] *= (clamp_radius / (r[mask][:, None] + 1e-8))
    # when Fz is negligible, mark as 0
    near_zero = Fz < (0.5 * np.nanmedian(Fz[Fz > 0]) if np.any(Fz > 0) else 1e-6)
    cop[near_zero] = 0.0
    return Fz.astype(np.float32), cop.astype(np.float32)


def safe_read(h5, path: str):
    try:
        if path in h5:
            return np.array(h5[path])
    except Exception:
        return None
    return None


def collect_trial(h5: h5py.File, cfg: Dict, base: str, level: str, trial: str, target_hz: float):
    out = {}
    prefix = f"/{base}/{level}/{trial}"
    time = safe_read(h5, f"{prefix}/Common/time")
    if time is None:
        return None
    # infer Hz; many labs store ms integer ticks
    fs_src = estimate_sampling_rate_from_time(time)
    # standardized time at target_hz
    T_src = time.shape[0]
    duration_s = (T_src - 1) / fs_src if T_src > 1 else 0
    T_dst = int(round(duration_s * target_hz)) + 1 if duration_s > 0 else T_src

    def resample_if_needed(arr, filt_hz=None):
        if arr is None:
            return None
        arr = arr.astype(np.float32)
        if filt_hz and fs_src > filt_hz:
            arr = lowpass(arr, cutoff_hz=filt_hz, fs_hz=fs_src, order=4)
        if fs_src != target_hz:
            arr = resample_to(arr, fs_src, target_hz)
        # Safety truncate/pad to T_dst
        if arr.shape[0] != T_dst:
            if arr.shape[0] > T_dst:
                arr = arr[:T_dst]
            else:
                pad = np.repeat(arr[-1:], T_dst - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
        return arr

    # IMUs
    back_acc = safe_read(h5, f"{prefix}/Back_imu/Accel")
    back_gyro = safe_read(h5, f"{prefix}/Back_imu/Gyro")
    thigh_acc = safe_read(h5, f"{prefix}/Thigh_imu/Accel")
    thigh_gyro = safe_read(h5, f"{prefix}/Thigh_imu/Gyro")

    filt_hz = cfg.get('sampling', {}).get('filter_hz', 15)
    back_acc = resample_if_needed(back_acc, filt_hz)
    back_gyro = resample_if_needed(back_gyro, filt_hz)
    thigh_acc = resample_if_needed(thigh_acc, filt_hz)
    thigh_gyro = resample_if_needed(thigh_gyro, filt_hz)

    # Axis mapping
    def map_imu(arr, which, kind):
        if arr is None:
            return None
        spec = cfg.get('imu_axes', {}).get(which, {}).get(kind, ['+x','+y','+z'])
        idx, sgn = parse_axis_mapping(spec)
        return apply_axis_mapping(arr, idx, sgn).astype(np.float32)

    back_acc = map_imu(back_acc, 'back', 'accel')
    back_gyro = map_imu(back_gyro, 'back', 'gyro')
    thigh_acc = map_imu(thigh_acc, 'thigh', 'accel')
    thigh_gyro = map_imu(thigh_gyro, 'thigh', 'gyro')

    # Insoles: we expect eight channels per foot
    fsrL = None
    fsrR = None
    # Try common patterns
    for i in range(1, 9):
        vL = safe_read(h5, f"{prefix}/Insole/Left/fsrL{i}")
        vR = safe_read(h5, f"{prefix}/Insole/Right/fsrR{i}")
        if vL is not None:
            fsrL = np.column_stack([safe_read(h5, f"{prefix}/Insole/Left/fsrL{j}") for j in range(1, 9)])
            break
    for i in range(1, 9):
        vR = safe_read(h5, f"{prefix}/Insole/Right/fsrR{i}")
        if vR is not None:
            fsrR = np.column_stack([safe_read(h5, f"{prefix}/Insole/Right/fsrR{j}") for j in range(1, 9)])
            break

    fsrL = resample_if_needed(fsrL, filt_hz)
    fsrR = resample_if_needed(fsrR, filt_hz)

    layout = cfg.get('fsr_layout', {})
    left_xy = np.array(layout.get('left', {}).get('positions_xy', []), dtype=np.float64)
    right_xy = np.array(layout.get('right', {}).get('positions_xy', []), dtype=np.float64)
    if left_xy.shape != (8, 2) or right_xy.shape != (8, 2):
        print('Warning: FSR layout positions not set to (8,2) — COP will be None')
        FzL, copL = (None, None)
        FzR, copR = (None, None)
    else:
        clamp_radius = cfg.get('fsr_cop_clamp_radius', 0.12)
        left_scale = float(cfg.get('fsr_calibration', {}).get('left_scale', 1.0))
        right_scale = float(cfg.get('fsr_calibration', {}).get('right_scale', 1.0))
        FzL, copL = compute_cop_from_fsr(fsrL, left_xy, clamp_radius, left_scale)
        FzR, copR = compute_cop_from_fsr(fsrR, right_xy, clamp_radius, right_scale)

    # Treadmill data (optional)
    treadmill = {}
    keys = cfg.get('treadmill_keys', []) or []
    tread_grp = f"{prefix}/Treadmill_data"
    if tread_grp in h5:
        if keys:
            for k in keys:
                arr = safe_read(h5, f"{tread_grp}/{k}")
                if arr is not None:
                    treadmill[k] = resample_if_needed(arr, filt_hz).astype(np.float32)
        else:
            # include all numeric datasets directly under this group
            grp = h5[tread_grp]
            for k, obj in grp.items():
                if isinstance(obj, h5py.Dataset):
                    try:
                        arr = np.array(obj)
                        if np.issubdtype(arr.dtype, np.number):
                            treadmill[k] = resample_if_needed(arr, filt_hz).astype(np.float32)
                    except Exception:
                        pass

    out = {
        'meta': {
            'base': base,
            'level': level,
            'trial': trial,
            'source_hz': fs_src,
            'target_hz': target_hz,
            'length': T_dst,
        },
        'sensors': {
            'back_accel': back_acc,
            'back_gyro': back_gyro,
            'thigh_accel': thigh_acc,
            'thigh_gyro': thigh_gyro,
        },
        'insole': {
            'fsr_left': fsrL.astype(np.float32) if fsrL is not None else None,
            'fsr_right': fsrR.astype(np.float32) if fsrR is not None else None,
            'Fz_left': FzL,
            'Fz_right': FzR,
            'cop_left': copL,
            'cop_right': copR,
        },
        'treadmill': treadmill,
        'labels': {}
    }

    # MoCap-derived labels (GRF and COP), if present
    def stack_components(base_path: str, comps: list):
        arrs = []
        for c in comps:
            a = safe_read(h5, f"{prefix}/{base_path}/{c}")
            if a is None:
                return None
            arrs.append(a.reshape(-1, 1))
        return np.concatenate(arrs, axis=1)

    grfL = stack_components('MoCap/grf_measured/left/force', ['Fx', 'Fy', 'Fz'])
    grfR = stack_components('MoCap/grf_measured/right/force', ['Fx', 'Fy', 'Fz'])
    copL_m = stack_components('MoCap/grf_measured/left/cop', ['x', 'y', 'z'])
    copR_m = stack_components('MoCap/grf_measured/right/cop', ['x', 'y', 'z'])

    # Resample labels to target_hz if available
    if grfL is not None:
        grfL = resample_if_needed(grfL, filt_hz)
        out['labels']['grf_left'] = grfL.astype(np.float32)
    if grfR is not None:
        grfR = resample_if_needed(grfR, filt_hz)
        out['labels']['grf_right'] = grfR.astype(np.float32)
    if copL_m is not None:
        copL_m = resample_if_needed(copL_m, filt_hz)
        out['labels']['cop_left'] = copL_m.astype(np.float32)
    if copR_m is not None:
        copR_m = resample_if_needed(copR_m, filt_hz)
        out['labels']['cop_right'] = copR_m.astype(np.float32)
    return out


def write_trial(h5w: h5py.File, trial_data: Dict, base: str, level: str, trial: str):
    grp_path = f"/{base}/{level}/{trial}"
    grp = h5w.require_group(grp_path)
    # write meta
    meta = trial_data['meta']
    for k, v in meta.items():
        grp.attrs[k] = v
    # write sensors
    def wds(path, arr):
        if arr is None:
            return
        h5w.create_dataset(path, data=arr, compression='gzip', compression_opts=4, shuffle=True)
    sens = trial_data['sensors']
    wds(f"{grp_path}/sensors/back_accel", sens.get('back_accel'))
    wds(f"{grp_path}/sensors/back_gyro", sens.get('back_gyro'))
    wds(f"{grp_path}/sensors/thigh_accel", sens.get('thigh_accel'))
    wds(f"{grp_path}/sensors/thigh_gyro", sens.get('thigh_gyro'))

    ins = trial_data['insole']
    wds(f"{grp_path}/insole/fsr_left", ins.get('fsr_left'))
    wds(f"{grp_path}/insole/fsr_right", ins.get('fsr_right'))
    wds(f"{grp_path}/insole/Fz_left", ins.get('Fz_left'))
    wds(f"{grp_path}/insole/Fz_right", ins.get('Fz_right'))
    wds(f"{grp_path}/insole/cop_left", ins.get('cop_left'))
    wds(f"{grp_path}/insole/cop_right", ins.get('cop_right'))

    tread = trial_data['treadmill']
    for k, arr in tread.items():
        wds(f"{grp_path}/treadmill/{k}", arr)

    # labels (if present)
    labels = trial_data.get('labels', {}) or {}
    for k, arr in labels.items():
        wds(f"{grp_path}/labels/{k}", arr)


def convert(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT, config_path: str = DEFAULT_CONFIG, subjects: Optional[list] = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_hz = float(cfg.get('sampling', {}).get('target_hz', 100))

    with h5py.File(input_path, 'r') as h5r, h5py.File(output_path, 'w') as h5w:
        n_trials = 0
        for base in h5r:
            if subjects and base not in subjects:
                continue
            base_grp = h5r[base]
            for level in base_grp:
                level_grp = base_grp[level]
                for trial in level_grp:
                    td = collect_trial(h5r, cfg, base, level, trial, target_hz)
                    if td is None:
                        continue
                    write_trial(h5w, td, base, level, trial)
                    n_trials += 1
        h5w.attrs['source_file'] = os.path.abspath(input_path)
        h5w.attrs['config_file'] = os.path.abspath(config_path)
        h5w.attrs['num_trials'] = n_trials
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Convert combined lab H5 to standardized sensor dataset for adaptor training')
    p.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Path to combined_data.h5')
    p.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='Output H5 path')
    p.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='YAML config for sensor mapping')
    p.add_argument('--subjects', type=str, nargs='*', help='Optional subject IDs to include, e.g., S001 S002')
    args = p.parse_args()

    out = convert(args.input, args.output, args.config, args.subjects)
    print(f"Wrote: {out}")
