#!/usr/bin/env python3
"""
Convert generated OSIM 23-DoF CSV (from S_generate_unconditional_speed.py) into an OpenSim .mot file
that can be opened in OpenSim (Coordinates) and in this repo's downstream tools.

Input CSV format expected:
- Columns: 'time' (seconds) + OSIM_DOF_ALL[:23] in any order
- Units: angles in radians (pelvis rotations, hips, knees, ankles, etc.), translations in meters

Output .mot:
- Coordinates format, inDegrees=no, with time column

Usage:
  python3 tools/export_to_opensim_mot.py --csv previews/samples/20251103_131600/gen_speed_1.20ms.csv \
    --out previews/samples/20251103_131600/gen_speed_1.20ms.mot

Optional batch:
  python3 tools/export_to_opensim_mot.py --glob 'previews/samples/*/*.csv' --out-dir previews/mot
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from consts import OSIM_DOF_ALL


COORD_COLUMNS = list(OSIM_DOF_ALL[:23])


def write_coordinates_mot(df: pd.DataFrame, out_path: str):
    # Assert required columns
    if 'time' not in df.columns:
        raise ValueError("Input CSV must contain a 'time' column in seconds")
    missing = [c for c in COORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OSIM columns: {missing}")

    # Reorder columns to OpenSim's expected order
    coord_df = df[COORD_COLUMNS].copy()
    time_series = df['time'].values

    num_frames = len(coord_df)
    out = []
    out.append('Coordinates')
    out.append('version=1')
    out.append(f'nRows={num_frames}')
    out.append(f'nColumns={len(COORD_COLUMNS)+1}')
    out.append('inDegrees=no')
    out.append('')
    out.append("If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).")
    out.append('')
    out.append('endheader')

    header = ['time'] + COORD_COLUMNS
    out.append('\t'.join(header))

    for i in range(num_frames):
        row = [f"{time_series[i]:.5f}"] + [f"{coord_df.iloc[i, j]:.8f}" for j in range(coord_df.shape[1])]
        out.append('\t'.join(row))

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(out))
    print(f"[WRITE] {out_path}")


def convert_one(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    write_coordinates_mot(df, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, help='Single CSV path to convert')
    ap.add_argument('--out', type=str, help='Output .mot path for single conversion')
    ap.add_argument('--glob', type=str, help='Glob pattern for batch CSV conversion')
    ap.add_argument('--out-dir', type=str, help='Output directory for batch conversion')
    args = ap.parse_args()

    if args.csv:
        if not args.out:
            raise SystemExit('--out is required when using --csv')
        convert_one(args.csv, args.out)
        return

    if args.glob:
        if not args.out_dir:
            raise SystemExit('--out-dir is required when using --glob')
        paths = sorted(glob.glob(args.glob))
        if not paths:
            raise SystemExit(f'No CSV matched pattern: {args.glob}')
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0] + '.mot'
            out_p = os.path.join(args.out_dir, base)
            convert_one(p, out_p)
        return

    raise SystemExit('Provide either --csv and --out, or --glob and --out-dir')


if __name__ == '__main__':
    main()
