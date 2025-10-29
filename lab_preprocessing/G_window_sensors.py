#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Tuple
import h5py
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT_DIR, 'lab_preprocessed', 'combined_sensors.h5')
DEFAULT_OUTDIR = os.path.join(ROOT_DIR, 'lab_preprocessed', 'windows')

# Fixed feature order for adaptor inputs
FEATURES = [
    ('sensors', 'back_accel', 3),
    ('sensors', 'back_gyro', 3),
    ('sensors', 'thigh_accel', 3),
    ('sensors', 'thigh_gyro', 3),
    ('insole', 'fsr_left', 8),
    ('insole', 'fsr_right', 8),
    ('insole', 'Fz_left', 1),
    ('insole', 'Fz_right', 1),
    ('insole', 'cop_left', 3),
    ('insole', 'cop_right', 3),
    # treadmill keys are optional; include if exist
    ('treadmill', 'belt_speed_left', 1),
    ('treadmill', 'belt_speed_right', 1),
]

# Optional labels to include if present in standardized H5 (written by converter under /labels)
LABELS = [
    ('labels', 'grf_left', 3),
    ('labels', 'grf_right', 3),
    ('labels', 'cop_left', 3),
    ('labels', 'cop_right', 3),
]


def h5_has_dataset(h: h5py.File, path: str) -> bool:
    try:
        return path in h
    except Exception:
        return False


def load_trial_features(h: h5py.File, grp_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # returns T x D array, mask D (1=present,0=padded), and list of feature names
    arrs = []
    mask = []
    names = []
    T = None
    for (g, k, dim) in FEATURES:
        ds_path = f"{grp_path}/{g}/{k}"
        if h5_has_dataset(h, ds_path):
            data = np.array(h[ds_path])
            if data.ndim == 1:
                data = data[:, None]
            if T is None:
                T = data.shape[0]
            # align time length if slightly off
            if data.shape[0] != T:
                m = min(T, data.shape[0])
                data = data[:m]
                if T != m:
                    T = m
            arrs.append(data)
            mask.extend([1] * data.shape[1])
            if data.shape[1] == dim:
                for i in range(dim):
                    names.append(f"{g}.{k}[{i}]")
            else:
                names.append(f"{g}.{k}")
        else:
            # pad with zeros
            if T is None:
                continue
            data = np.zeros((T, dim), dtype=np.float32)
            arrs.append(data)
            mask.extend([0] * dim)
            for i in range(dim):
                names.append(f"{g}.{k}[{i}]")
    if not arrs:
        return None, None, []
    X = np.concatenate(arrs, axis=1).astype(np.float32)
    return X, np.array(mask, dtype=np.uint8), names


def load_trial_labels(h: h5py.File, grp_path: str, T: int) -> Tuple[np.ndarray, List[str]]:
    arrs = []
    names = []
    for (g, k, dim) in LABELS:
        ds_path = f"{grp_path}/{g}/{k}"
        if h5_has_dataset(h, ds_path):
            data = np.array(h[ds_path])
            if data.ndim == 1:
                data = data[:, None]
            # align to T
            if data.shape[0] != T:
                m = min(T, data.shape[0])
                data = data[:m]
                if T != m:
                    # pad with last value
                    pad = np.repeat(data[-1:], T - m, axis=0)
                    data = np.concatenate([data, pad], axis=0)
            arrs.append(data.astype(np.float32))
            if data.shape[1] == dim:
                for i in range(dim):
                    names.append(f"{g}.{k}[{i}]")
            else:
                names.append(f"{g}.{k}")
    if not arrs:
        return None, []
    Y = np.concatenate(arrs, axis=1).astype(np.float32)
    return Y, names


def make_windows(X: np.ndarray, wlen: int = 150, stride: int = 25) -> np.ndarray:
    T, D = X.shape
    if T < wlen:
        return np.empty((0, wlen, D), dtype=X.dtype)
    idx = [np.arange(i, i + wlen) for i in range(0, T - wlen + 1, stride)]
    idx = np.stack(idx, axis=0)
    return X[idx]


def run(input_path: str = DEFAULT_INPUT, outdir: str = DEFAULT_OUTDIR, wlen: int = 150, stride: int = 25):
    os.makedirs(outdir, exist_ok=True)
    index = []
    with h5py.File(input_path, 'r') as h:
        for base in h:
            for level in h[base]:
                for trial in h[f"/{base}/{level}"]:
                    grp_path = f"/{base}/{level}/{trial}"
                    X, mask, names = load_trial_features(h, grp_path)
                    if X is None:
                        continue
                    # optional labels
                    Y, label_names = load_trial_labels(h, grp_path, T=X.shape[0])
                    windows = make_windows(X, wlen=wlen, stride=stride)
                    if windows.shape[0] == 0:
                        continue
                    # save npz per trial
                    out_path = os.path.join(outdir, f"{base}_{level}_{trial}.npz")
                    if Y is not None:
                        y_win = make_windows(Y, wlen=wlen, stride=stride)
                    else:
                        y_win = None
                    if y_win is not None and y_win.shape[0] == windows.shape[0]:
                        np.savez_compressed(out_path,
                                            windows=windows,
                                            mask=mask,
                                            feature_names=np.array(names),
                                            labels=y_win,
                                            label_names=np.array(label_names))
                    else:
                        np.savez_compressed(out_path, windows=windows, mask=mask, feature_names=np.array(names))
                    index.append({'base': base, 'level': level, 'trial': trial, 'npz': out_path, 'num_windows': int(windows.shape[0])})
    with open(os.path.join(outdir, 'index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    return outdir


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Create fixed-length windows from standardized sensor H5 for adaptor training')
    p.add_argument('--input', type=str, default=DEFAULT_INPUT)
    p.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR)
    p.add_argument('--wlen', type=int, default=150)
    p.add_argument('--stride', type=int, default=25)
    args = p.parse_args()
    out = run(args.input, args.outdir, args.wlen, args.stride)
    print(f"Wrote windows to: {out}")
