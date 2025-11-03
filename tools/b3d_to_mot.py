#!/usr/bin/env python3
"""
Convert a SubjectOnDisk .b3d trial to an OpenSim .mot (Coordinates) file.
- Reads DOF positions from the first processing pass (same as Visualize/V01_visualize_ab_walk.py)
- Writes time (s) + 23 OSIM coordinates in radians/meters, in the order consts.OSIM_DOF_ALL[:23]

Usage examples:
  python3 tools/b3d_to_mot.py \
    --b3d data/Wang2023_Formatted_No_Arm/Wang2023_Formatted_No_Arm/Subj06/Subj06.b3d \
    --trial-substr walk_09 \
    --out previews/ab_Subj06_walk_09.mot

  # Pick by explicit index
  python3 tools/b3d_to_mot.py --b3d <file.b3d> --trial-idx 19 --out out.mot
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from consts import OSIM_DOF_ALL  # noqa: E402
from tools.export_to_opensim_mot import write_coordinates_mot  # noqa: E402

try:
    import nimblephysics as nimble  # noqa: E402
except Exception as e:
    print("[ERROR] nimblephysics import failed. Install nimblephysics to read .b3d files.\n", e)
    sys.exit(2)


COORDS_23 = list(OSIM_DOF_ALL[:23])


def find_trial_by_substr(subject, substr: str):
    s = substr.lower()
    for i in range(subject.getNumTrials()):
        try:
            name = subject.getTrialName(i)
        except Exception:
            name = f"trial_{i:02d}"
        if s in name.lower():
            return i, name
    return None, None


def read_trial_positions(subject, trial_idx: int) -> tuple[np.ndarray, float, str]:
    """Return (poses[T,D], dt, trial_name). Poses from first processing pass."""
    dt = subject.getTrialTimestep(trial_idx)
    n = subject.getTrialLength(trial_idx)
    name = subject.getTrialName(trial_idx)
    # Read frames with processing passes (fast enough for ~2k frames)
    frames = subject.readFrames(trial_idx, 0, n, False, True)
    if not frames:
        raise RuntimeError("No frames returned from SubjectOnDisk.readFrames")
    if not frames[0].processingPasses:
        raise RuntimeError("Frames do not contain processing passes; cannot read positions")
    pass0 = [fr.processingPasses[0] for fr in frames]
    poses = np.array([fr.pos for fr in pass0], dtype=np.float64)  # [T, D]
    return poses, float(dt), name


def to_mot_df(poses: np.ndarray, dt: float) -> pd.DataFrame:
    """Build a DataFrame with columns: time + 23 coords in OSIM order."""
    T, D = poses.shape
    if D < 23:
        raise ValueError(f"Expected at least 23 DOFs in poses, got {D}")
    poses23 = poses[:, :23]
    t = np.arange(T, dtype=np.float64) * dt
    data = {"time": t}
    for j, col in enumerate(COORDS_23):
        data[col] = poses23[:, j]
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser(description="Convert a .b3d trial to OpenSim .mot (Coordinates)")
    ap.add_argument('--b3d', required=True, help='Path to SubjectOnDisk .b3d file')
    ap.add_argument('--trial-idx', type=int, default=None, help='Exact trial index to export')
    ap.add_argument('--trial-substr', type=str, default='walk', help='Pick first trial containing this substring (ignored if --trial-idx set)')
    ap.add_argument('--out', type=str, default=None, help='Output .mot path (default: previews/mot/<trial_name>.mot)')
    args = ap.parse_args()

    b3d_path = os.path.abspath(args.b3d)
    if not os.path.exists(b3d_path):
        raise FileNotFoundError(b3d_path)

    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Decide trial
    if args.trial_idx is not None:
        trial_idx = int(args.trial_idx)
        trial_name = subject.getTrialName(trial_idx)
    else:
        trial_idx, trial_name = find_trial_by_substr(subject, args.trial_substr)
        if trial_idx is None:
            raise SystemExit(f"No trial containing '{args.trial_substr}' found in {b3d_path}")

    poses, dt, trial_name = read_trial_positions(subject, trial_idx)
    df = to_mot_df(poses, dt)

    # Output path
    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        base = os.path.splitext(os.path.basename(b3d_path))[0]
        safe_trial = trial_name.replace('/', '_')
        out_dir = os.path.join(REPO_ROOT, 'previews', 'mot')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_{safe_trial}.mot")

    write_coordinates_mot(df, out_path)


if __name__ == '__main__':
    main()
