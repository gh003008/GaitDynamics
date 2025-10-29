#!/usr/bin/env python3
"""
Quick LD→GD plausibility preview plots for a single trial segment.

Generates small diagnostic plots without converting the entire dataset:
- GRF Fy (left/right) over time
- CoP–calcaneus distance magnitude over time (left/right) with 0.3 m threshold
- Pelvis yaw (raw) and yaw aligned to median yaw (for moving-direction sanity)

Inputs are read from the lab HDF5 (S004–S011). This script only reads a short
segment (start_sec, duration_sec) for fast iteration.

Example usage:
  python lab_preprocessing/G_preview_ld2gd.py \
    --input ./combined_data.h5 \
    --trial_path /S004/accel_sine/trial_01 \
    --start_sec 5.0 --duration_sec 6.0 \
    --outdir previews/S004_accel_sine_trial01

Notes:
- No body mass/height required: plots use raw Newtons and raw distances.
- If time vector is unavailable in some subgroup, we fall back to 100 Hz indexing.
- CoP normalization (GD-style) is not shown here; we instead plot raw CoP distance
  to calcaneus (same metric used for quality check: < 0.3 m during stance).
"""

import argparse
import os
import math
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _read_dset_safe(h5: h5py.File, path: str, default=None):
    try:
        return h5[path][...]
    except Exception:
        return default


def _read_vec3_from_body_pos_global(h5: h5py.File, trial: str, body: str):
    base = f"{trial}/MoCap/body_pos_global/{body}_"
    X = _read_dset_safe(h5, base + "X")
    Y = _read_dset_safe(h5, base + "Y")
    Z = _read_dset_safe(h5, base + "Z")
    if X is None or Y is None or Z is None:
        return None
    return np.stack([X, Y, Z], axis=-1)


def _read_grf_and_cop(h5: h5py.File, trial: str):
    """Returns dict with left/right Fx,Fy,Fz and CoP x,y,z and time."""
    out = {}
    for side in ["left", "right"]:
        Fx = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/force/Fx")
        Fy = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/force/Fy")
        Fz = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/force/Fz")
        cx = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/cop/x")
        cy = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/cop/y")
        cz = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/{side}/cop/z")
        out[side] = {
            "F": None if Fx is None or Fy is None or Fz is None else np.stack([Fx, Fy, Fz], axis=-1),
            "CoP": None if cx is None or cy is None or cz is None else np.stack([cx, cy, cz], axis=-1),
        }
    t = _read_dset_safe(h5, f"{trial}/MoCap/grf_measured/time")
    out["time"] = t
    return out


def _read_pelvis_yaw(h5: h5py.File, trial: str):
    # OpenSim convention used in this repo: pelvis_rotation is yaw (about vertical), in radians.
    yaw = _read_dset_safe(h5, f"{trial}/MoCap/ik_data/pelvis_rotation")
    if yaw is None:
        return None
    return yaw  # radians


def _select_time_range(time_vec: np.ndarray, start_sec: float, duration_sec: float, default_hz: int = 100):
    if time_vec is not None and time_vec.size > 1:
        t0 = time_vec[0]
        s = np.searchsorted(time_vec, t0 + start_sec, side="left")
        e = np.searchsorted(time_vec, t0 + start_sec + duration_sec, side="left")
        return slice(s, max(s + 1, e)), (time_vec[s:e] if e > s else time_vec[s:s + 1])
    # fallback: 100 Hz index
    s_idx = int(start_sec * default_hz)
    e_idx = int((start_sec + duration_sec) * default_hz)
    t = np.arange(s_idx, e_idx) / float(default_hz)
    return slice(s_idx, e_idx), t


def _plot_fy(time, FyL, FyR, out_png, stance_thresh_N: float = 50.0):
    plt.figure(figsize=(9, 3))
    if FyL is not None:
        plt.plot(time, FyL, label="Fy_left (N)")
    if FyR is not None:
        plt.plot(time, FyR, label="Fy_right (N)")
    plt.axhline(stance_thresh_N, color="k", linestyle="--", linewidth=1, label=f"stance ~{stance_thresh_N:.0f} N")
    plt.xlabel("Time (s)")
    plt.ylabel("Fy (N)")
    plt.title("Vertical GRF (Fy)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_cop_distance(time, dL, dR, out_png, limit_m: float = 0.3):
    plt.figure(figsize=(9, 3))
    if dL is not None:
        plt.plot(time, dL, label="|CoP−calcn| left (m)")
    if dR is not None:
        plt.plot(time, dR, label="|CoP−calcn| right (m)")
    plt.axhline(limit_m, color="r", linestyle=":", linewidth=1, label=f"limit {limit_m:.2f} m")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("CoP distance to calcaneus (quality check)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_pelvis_yaw(time, yaw_rad, out_png):
    yaw_deg = np.rad2deg(yaw_rad)
    yaw_med = np.median(yaw_deg)
    yaw_aligned = yaw_deg - yaw_med
    yaw_span = np.max(yaw_deg) - np.min(yaw_deg)

    plt.figure(figsize=(9, 3))
    plt.plot(time, yaw_deg, label="pelvis yaw (deg)")
    plt.plot(time, yaw_aligned, label="yaw (median-aligned)")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("deg")
    plt.title(f"Pelvis yaw (span={yaw_span:.1f}°); exclude if >45°")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def preview_trial_segment(h5_path: str, trial_path: str, start_sec: float, duration_sec: float, outdir: str):
    _ensure_dir(outdir)
    with h5py.File(h5_path, "r") as h5:
        # 1) Load GRF/COP and time
        grf = _read_grf_and_cop(h5, trial_path)
        t_grf = grf.get("time", None)
        sl, t = _select_time_range(t_grf, start_sec, duration_sec)

        FyL = grf["left"]["F"][:, 1][sl] if grf.get("left", {}).get("F", None) is not None else None
        FyR = grf["right"]["F"][:, 1][sl] if grf.get("right", {}).get("F", None) is not None else None

        CoPL = grf["left"]["CoP"][sl] if grf.get("left", {}).get("CoP", None) is not None else None
        CoPR = grf["right"]["CoP"][sl] if grf.get("right", {}).get("CoP", None) is not None else None

        # 2) Load calcaneus world positions
        calcnL = _read_vec3_from_body_pos_global(h5, trial_path, "calcn_l")
        calcnR = _read_vec3_from_body_pos_global(h5, trial_path, "calcn_r")
        calcnL = calcnL[sl] if calcnL is not None else None
        calcnR = calcnR[sl] if calcnR is not None else None

        # 3) Pelvis yaw
        yaw = _read_pelvis_yaw(h5, trial_path)
        yaw = yaw[sl] if yaw is not None else None

    # Compute CoP distances
    dL = None if CoPL is None or calcnL is None else np.linalg.norm(CoPL - calcnL, axis=-1)
    dR = None if CoPR is None or calcnR is None else np.linalg.norm(CoPR - calcnR, axis=-1)

    base = os.path.basename(trial_path.strip("/"))
    out_fy = os.path.join(outdir, f"{base}_fy.png")
    out_cop = os.path.join(outdir, f"{base}_cop_distance.png")
    out_yaw = os.path.join(outdir, f"{base}_pelvis_yaw.png")

    _plot_fy(t, FyL, FyR, out_fy)
    _plot_cop_distance(t, dL, dR, out_cop)
    if yaw is not None:
        _plot_pelvis_yaw(t, yaw, out_yaw)

    return {
        "fy_png": out_fy,
        "cop_distance_png": out_cop,
        "pelvis_yaw_png": out_yaw if yaw is not None else None,
        "time_range": (t[0] if len(t) else None, t[-1] if len(t) else None),
    }


def main():
    ap = argparse.ArgumentParser(description="Preview LD→GD plausibility plots for a trial segment")
    ap.add_argument("--input", required=True, help="Path to combined_data.h5")
    ap.add_argument("--trial_path", required=True, help="Full H5 path to the trial (e.g., /S004/accel_sine/trial_01)")
    ap.add_argument("--start_sec", type=float, default=0.0, help="Start time (s)")
    ap.add_argument("--duration_sec", type=float, default=5.0, help="Duration (s)")
    ap.add_argument("--outdir", default="previews", help="Directory to save PNGs")
    args = ap.parse_args()

    info = preview_trial_segment(args.input, args.trial_path, args.start_sec, args.duration_sec, args.outdir)
    print("Saved:")
    for k, v in info.items():
        if isinstance(v, str) and v is not None:
            print(f"  - {k}: {v}")
    if isinstance(info.get("time_range"), tuple):
        print(f"Time range: {info['time_range'][0]}–{info['time_range'][1]} s")


if __name__ == "__main__":
    main()
