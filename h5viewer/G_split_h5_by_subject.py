#!/usr/bin/env python3
"""
Split a large combined HDF5 (e.g., combined_data.h5) by top-level subject groups (e.g., S001..S011).

This creates an output folder alongside the input (default: <input>_split/), and
writes one H5 per subject, preserving the subject group name at the root of each output file.

Reusable defaults are declared at the top (DEFAULT_INPUTS), but can be overridden by CLI.

Examples:
  - Use defaults declared below:
      python h5viewer/G_split_h5_by_subject.py
  - Custom input and output directory:
      python h5viewer/G_split_h5_by_subject.py --input ./combined_data.h5 --outdir ./combined_data_split
  - Only split selected subjects:
      python h5viewer/G_split_h5_by_subject.py --input ./combined_data.h5 --subjects S004 S005 S006
  - Use regex to filter subjects:
      python h5viewer/G_split_h5_by_subject.py --input ./combined_data.h5 --subject_regex "^S00(4|5|6|7|8|9|10|11)$"
"""

import argparse
import os
from pathlib import Path
import re
import sys
import h5py


# ----- Configure default targets here for quick reuse -----
# You can edit this list and run the script with no CLI args to process them.
DEFAULT_INPUTS = [
    "./combined_data.h5",
]


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def default_outdir_for(input_path: str) -> str:
    p = Path(input_path)
    base = p.name
    name_wo_ext = base[:-3] if base.endswith(".h5") else base
    return str(p.parent / f"{name_wo_ext}_split")


def list_top_groups(h5: h5py.File):
    """Return a list of top-level names that are groups (e.g., ['S001', 'S002', ...])."""
    names = []
    for k in h5.keys():
        obj = h5.get(k, getclass=True)
        if obj is h5py.Group:
            names.append(k)
    return names


def subject_filter(names, subjects=None, subject_regex=None):
    if subjects:
        wanted = set(subjects)
        return [n for n in names if n in wanted]
    if subject_regex:
        pat = re.compile(subject_regex)
        return [n for n in names if pat.search(n)]
    return names


def copy_group_to_new_file(src_h5: h5py.File, group_name: str, out_path: str, overwrite=False):
    out_p = Path(out_path)
    if out_p.exists():
        if not overwrite:
            print(f"[skip] {out_path} exists. Use --overwrite to replace.")
            return
        out_p.unlink()

    with h5py.File(out_path, "w") as dst:
        # Copy the entire subject group into the root of the new file
        # The copy API handles nested datasets/attributes recursively.
        src_h5.copy(f"/{group_name}", dst, name=group_name)
        # Optionally, copy file-level attributes if any (rare in our case)
        for k, v in src_h5.attrs.items():
            dst.attrs[k] = v

    print(f"[ok] Wrote {out_path}")


def split_file_by_subject(input_path: str, outdir: str = None, subjects=None, subject_regex=None, overwrite=False):
    if outdir is None:
        outdir = default_outdir_for(input_path)
    ensure_dir(outdir)

    if not Path(input_path).exists():
        print(f"[err] Input not found: {input_path}")
        return 1

    with h5py.File(input_path, "r") as h5:
        top = list_top_groups(h5)
        # Heuristic: if dataset includes non-subject groups, you can filter by regex
        # e.g., subject_regex='^S\\d{3}$'
        selected = subject_filter(top, subjects=subjects, subject_regex=subject_regex)
        if not selected:
            print(f"[warn] No subjects selected from {input_path}. Top groups: {top}")
            return 0

        print(f"[info] Splitting {input_path} into {len(selected)} files → {outdir}")
        for s in selected:
            out_path = str(Path(outdir) / f"{s}.h5")
            copy_group_to_new_file(h5, s, out_path, overwrite=overwrite)

    return 0


def build_argparser():
    ap = argparse.ArgumentParser(description="Split combined HDF5 by top-level subject groups (e.g., S001…)")
    ap.add_argument("--input", nargs="?", help="Path to input H5 (default uses DEFAULT_INPUTS list)")
    ap.add_argument("--outdir", help="Directory to write outputs (default: <input>_split)")
    ap.add_argument("--subjects", nargs="*", help="Explicit subjects to include (e.g., S004 S005)")
    ap.add_argument("--subject_regex", help="Regex to filter subject names (e.g., '^S00[4-9]|S01[01]$')")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    inputs = [args.input] if args.input else DEFAULT_INPUTS
    rc = 0
    for inp in inputs:
        rc |= split_file_by_subject(
            inp,
            outdir=args.outdir,
            subjects=args.subjects,
            subject_regex=args.subject_regex,
            overwrite=args.overwrite,
        )
    sys.exit(rc)


if __name__ == "__main__":
    main()
