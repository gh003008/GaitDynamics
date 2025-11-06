#!/usr/bin/env python3
import argparse
import os
import json
import re
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from consts import OSIM_DOF_ALL, KINETICS_ALL, JOINTS_3D_ALL
from model.utils import (
    data_filter,
    linear_resample_data,
    align_moving_direction,
    convert_addb_state_to_model_input,
)


def build_opt(target_fs: int, window_len: int):
    class Opt:
        pass
    opt = Opt()
    opt.target_sampling_rate = target_fs
    opt.window_len = window_len

    # q (23 DOFs, no arm) + kinetics (12)
    opt.osim_dof_columns = OSIM_DOF_ALL[:23] + KINETICS_ALL
    # we only use pelvis/hip/lumbar for 6D + angular v in convert_addb_state_to_model_input
    opt.joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ["pelvis", "hip_r", "hip_l", "lumbar"]}

    # Model state columns are derived inside convert_addb_state_to_model_input, but we still keep a reference name list
    # Minimal placeholder; we'll overwrite after conversion
    opt.model_states_column_names = None

    # indices inside the osim_dof_columns layout
    opt.grf_osim_col_loc = [i for i, c in enumerate(opt.osim_dof_columns) if ("force" in c and "_cop_" not in c)]
    opt.cop_osim_col_loc = [i for i, c in enumerate(opt.osim_dof_columns) if "_cop_" in c]
    opt.kinematic_osim_col_loc = [i for i, c in enumerate(opt.osim_dof_columns) if "force" not in c]
    return opt


def list_trials(h5: h5py.File, subject: str = None) -> List[str]:
    trials = []
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and name.endswith("/MoCap/ik_data"):
            if subject is None or ("/"+name).split("/")[1] == subject:
                trials.append("/" + name)
    h5.visititems(visit)
    # strip ik_data group to trial root
    roots = sorted(list(set([p.rsplit("/MoCap/ik_data", 1)[0] for p in trials])))
    return roots


def read_ik(h5: h5py.File, trial_root: str) -> Tuple[np.ndarray, List[str], float]:
    """Read IK columns as per OSIM_DOF_ALL[:23]. Returns (poses[T,23], names, sampling_rate_hz)."""
    g = h5[trial_root + "/MoCap/ik_data"]
    names = OSIM_DOF_ALL[:23]
    series = []
    lengths = []
    for name in names:
        if name not in g:
            # missing: fill zeros
            series.append(np.zeros_like(h5[trial_root + "/MoCap/ik_data/" + list(g.keys())[0]][:]))
        else:
            series.append(g[name][:])
        lengths.append(series[-1].shape[0])
    T = min(lengths) if lengths else 0
    series = [s[:T] for s in series]
    poses = np.stack(series, axis=1).astype(np.float64)
    # Convert angular DOFs from degrees to radians. Translations (tx,ty,tz) stay in meters.
    deg_cols = [i for i, n in enumerate(names) if n not in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    poses[:, deg_cols] = np.deg2rad(poses[:, deg_cols])
    # sampling rate via time vector if present
    fs = None
    if "time" in g:
        t = g["time"][:]
        if t.shape[0] >= 2:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                fs = 1.0 / dt
    return poses, names, fs


def read_grf_cop(h5: h5py.File, trial_root: str, T_ref: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return force_r[T,3], cop_r[T,3], force_l[T,3], cop_l[T,3] clipped to T_ref.
    LD stores force as a group with Fx/Fy/Fz datasets and cop as group with x/y/z.
    """
    base = trial_root + "/MoCap/grf_measured/"
    def read_force(side: str):
        g = h5[base + side + "/force"]
        Fx = g["Fx"][:]; Fy = g["Fy"][:]; Fz = g["Fz"][:]
        T = min(len(Fx), len(Fy), len(Fz))
        return np.stack([Fx[:T], Fy[:T], Fz[:T]], axis=1)
    def read_cop(side: str):
        g = h5[base + side + "/cop"]
        x = g["x"][:]; y = g["y"][:]; z = g["z"][:]
        T = min(len(x), len(y), len(z))
        return np.stack([x[:T], y[:T], z[:T]], axis=1)
    r_force = read_force("right"); l_force = read_force("left")
    r_cop = read_cop("right"); l_cop = read_cop("left")
    T = min(T_ref, r_force.shape[0], l_force.shape[0], r_cop.shape[0], l_cop.shape[0])
    return (r_force[:T], r_cop[:T], l_force[:T], l_cop[:T])


def remap_ld_axes_for_grf_cop(r_force: np.ndarray, r_cop: np.ndarray,
                               l_force: np.ndarray, l_cop: np.ndarray,
                               ld_vertical_is_fz: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """If LD stores vertical as Fz (Z-up), remap to AB convention (Y-up):
    Forces: (Fx, Fy, Fz) -> (Fx, Fz, Fy)
    CoP:    (x,  y,  z)  -> (x,  z,  y)
    """
    if not ld_vertical_is_fz:
        return r_force, r_cop, l_force, l_cop
    def swap_yz(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        out[:, [1, 2]] = out[:, [2, 1]]
        return out
    return swap_yz(r_force), swap_yz(r_cop), swap_yz(l_force), swap_yz(l_cop)


def remap_ld_axes_for_pos(p: np.ndarray, ld_vertical_is_fz: bool) -> np.ndarray:
    """Remap body world positions from Z-up to Y-up if needed (swap Y<->Z)."""
    if not ld_vertical_is_fz:
        return p
    out = p.copy()
    out[:, [1, 2]] = out[:, [2, 1]]
    return out


def apply_cop_sign(states: np.ndarray, opt, sign_r: np.ndarray, sign_l: np.ndarray) -> np.ndarray:
    """Apply sign multipliers to normalized CoP vectors in-place (right and left)."""
    s = states.copy()
    # Right CoP block
    r0 = opt.cop_osim_col_loc[0]
    s[:, r0 + 0] *= sign_r[0]
    s[:, r0 + 1] *= sign_r[1]
    s[:, r0 + 2] *= sign_r[2]
    # Left CoP block
    l0 = opt.cop_osim_col_loc[3]
    s[:, l0 + 0] *= sign_l[0]
    s[:, l0 + 1] *= sign_l[1]
    s[:, l0 + 2] *= sign_l[2]
    return s


def read_calcn_world(h5: h5py.File, trial_root: str, T_ref: int) -> Tuple[np.ndarray, np.ndarray]:
    base = trial_root + "/MoCap/body_pos_global/"
    rx = h5[base + "calcn_r_X"][:]
    ry = h5[base + "calcn_r_Y"][:]
    rz = h5[base + "calcn_r_Z"][:]
    lx = h5[base + "calcn_l_X"][:]
    ly = h5[base + "calcn_l_Y"][:]
    lz = h5[base + "calcn_l_Z"][:]
    T = min(T_ref, rx.shape[0], ry.shape[0], rz.shape[0], lx.shape[0], ly.shape[0], lz.shape[0])
    r = np.stack([rx[:T], ry[:T], rz[:T]], axis=1)
    l = np.stack([lx[:T], ly[:T], lz[:T]], axis=1)
    return r, l


def norm_cops_from_ld(states: np.ndarray, opt, height_m: float, weight_kg: float, calcn_r: np.ndarray, calcn_l: np.ndarray,
                       check_distance: bool = True, wrong_ratio: float = 0.002) -> np.ndarray:
    """Mimic model.utils.norm_cops using LD calcaneus world pos instead of a nimble skeleton."""
    states = torch.from_numpy(states).float()
    
    # Replace NaN values with 0 (COP is NaN when foot is in air, GRF=0)
    states = torch.nan_to_num(states, nan=0.0)
    
    poses = states[:, opt.kinematic_osim_col_loc]
    forces = states[:, opt.grf_osim_col_loc]
    cops = states[:, opt.cop_osim_col_loc]

    # Build heel centers from LD
    heel_centers = torch.from_numpy(np.hstack([calcn_r, calcn_l])).to(states.dtype)

    for i_plate in range(2):
        force_vector = forces[:, 3 * i_plate:3 * (i_plate + 1)]
        vector = (cops - heel_centers)[:, 3 * i_plate:3 * (i_plate + 1)]
        stance_phase = force_vector[:, 1] > (50.0 / weight_kg)
        if check_distance:
            far = (vector[stance_phase, :].abs() > 0.3).sum() / 3
            if far > wrong_ratio * len(stance_phase):
                return None  # signal bad CoP
        normed_vector = vector * force_vector[:, 1:2] / height_m
        # zero when low force
        normed_vector[force_vector[:, 1] * weight_kg < 20.0] = 0
        states[:, opt.cop_osim_col_loc[3 * i_plate:3 * (i_plate + 1)]] = normed_vector.to(states.dtype)
    return states.numpy()


def build_states(poses: np.ndarray, r_force: np.ndarray, r_cop: np.ndarray, l_force: np.ndarray, l_cop: np.ndarray,
                 weight_kg: float, grf_scale: float = 1.0) -> np.ndarray:
    # GRF to %BW
    rF = (r_force * grf_scale) / weight_kg
    lF = (l_force * grf_scale) / weight_kg
    # concatenate in the same order the pipeline expects: q(23) + rF(3) + rCOP(3) + lF(3) + lCOP(3)
    return np.concatenate([poses, rF, r_cop, lF, l_cop], axis=1)


def compute_probably_missing(r_force: np.ndarray, l_force: np.ndarray, thresh: float = 1e-8) -> np.ndarray:
    # Missing when both feet forces are zero/negligible
    r_mag = np.linalg.norm(r_force, axis=1)
    l_mag = np.linalg.norm(l_force, axis=1)
    return (r_mag < thresh) & (l_mag < thresh)


def process_trial(h5: h5py.File, opt, trial_root: str, height_m: float, weight_kg: float,
                  resample_to: int, align_mdir: bool, check_cop_dist: bool, grf_scale: float,
                  ld_vertical_is_fz: bool, flip_dofs: List[str],
                  cop_sign_r: np.ndarray, cop_sign_l: np.ndarray) -> Dict:
    poses, pose_names, fs = read_ik(h5, trial_root)
    rF, rC, lF, lC = read_grf_cop(h5, trial_root, poses.shape[0])

    # Remap LD axes to AB convention if needed (make Fy vertical)
    rF, rC, lF, lC = remap_ld_axes_for_grf_cop(rF, rC, lF, lC, ld_vertical_is_fz)

    # Sync lengths
    T = min(poses.shape[0], rF.shape[0], rC.shape[0], lF.shape[0], lC.shape[0])
    poses, rF, rC, lF, lC = poses[:T], rF[:T], rC[:T], lF[:T], lC[:T]

    # Resample to target fs if needed
    if fs is None:
        # try from GRF: assume 1000/Fs in LD isn't available; proceed without resample
        fs = resample_to
    if int(round(fs)) != resample_to:
        poses = linear_resample_data(poses, int(round(fs)), resample_to)
        rF = linear_resample_data(rF, int(round(fs)), resample_to)
        lF = linear_resample_data(lF, int(round(fs)), resample_to)
        rC = linear_resample_data(rC, int(round(fs)), resample_to)
        lC = linear_resample_data(lC, int(round(fs)), resample_to)

    # Filter a little for stability
    poses = data_filter(poses, 15, resample_to, 4).astype(np.float32)

    # Apply DOF sign fixes if requested
    if flip_dofs:
        name_to_idx = {n: i for i, n in enumerate(pose_names)}
        for n in flip_dofs:
            if n in name_to_idx:
                poses[:, name_to_idx[n]] *= -1.0

    # Build states in osim_dof_columns order
    states = build_states(poses, rF, rC, lF, lC, weight_kg, grf_scale=grf_scale)

    # Replace COPs with normalized vectors using LD calcaneus world positions
    calcn_r, calcn_l = read_calcn_world(h5, trial_root, states.shape[0])
    # Ensure calcn world positions are in the same Y-up frame as GRF/CoP
    calcn_r = remap_ld_axes_for_pos(calcn_r, ld_vertical_is_fz)
    calcn_l = remap_ld_axes_for_pos(calcn_l, ld_vertical_is_fz)
    states = norm_cops_from_ld(states, opt, height_m, weight_kg, calcn_r, calcn_l, check_distance=check_cop_dist)
    if states is None:
        return {"skip": True, "reason": "cop_far_from_foot"}

    # Optional CoP sign correction per side (after normalization, before yaw alignment)
    if cop_sign_r is not None and cop_sign_l is not None:
        states = apply_cop_sign(states, opt, cop_sign_r, cop_sign_l)

    # Align moving direction (yaw)
    if align_mdir:
        aligned, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
        if aligned is False:
            return {"skip": True, "reason": "large_mdir_change"}
        states = aligned.numpy() if isinstance(aligned, torch.Tensor) else aligned

    # Convert to model input (6D rotations + angular v + kinematics v)
    states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
    states_df, pos_vec = convert_addb_state_to_model_input(states_df, opt.joints_3d, resample_to)
    model_cols = list(states_df.columns)
    opt.model_states_column_names = model_cols  # record

    # Compute probably_missing from GRF
    prob_missing = compute_probably_missing(rF, lF)

    return {
        "model_states": states_df.values.astype(np.float32),
        "model_states_columns": model_cols,
        "height_m": float(height_m),
        "weight_kg": float(weight_kg),
        "pos_vec": [float(x) for x in pos_vec],
        "probably_missing": prob_missing.astype(bool),
        "sampling_rate": int(resample_to),
    }


def estimate_weight_from_grf(h5: h5py.File, trial_root: str, ld_vertical_is_fz: bool = True) -> float:
    """Estimate body mass (kg) from vertical GRF (N): median((Fvert_r+Fvert_l)/9.81) over stance frames.
    If LD uses Z-up for measured GRF (common for plates), set ld_vertical_is_fz=True.
    """
    # LD stores force as a group with Fx/Fy/Fz datasets
    base = trial_root + "/MoCap/grf_measured/"
    try:
        vert_comp = "Fz" if ld_vertical_is_fz else "Fy"
        rFv = h5[base + f"right/force/{vert_comp}"][:]
        lFv = h5[base + f"left/force/{vert_comp}"][:]
    except Exception:
        # Fallback: if stored as [T,3] dataset (older exports)
        rF = h5[base + "right/force"][:]
        lF = h5[base + "left/force"][:]
        if ld_vertical_is_fz:
            rFv = rF[:, 2]
            lFv = lF[:, 2]
        else:
            rFv = rF[:, 1]
            lFv = lF[:, 1]
    T = min(len(rFv), len(lFv))
    Fv_sum = rFv[:T] + lFv[:T]
    stance = Fv_sum > 100.0  # N
    vals = Fv_sum[stance] / 9.81 if stance.any() else Fv_sum / 9.81
    if vals.size == 0:
        return float(np.median(Fv_sum) / 9.81) if Fv_sum.size else 70.0
    return float(np.median(vals))


def estimate_height_from_com_and_feet(h5: h5py.File, trial_root: str) -> float:
    """Estimate height using CoM vertical vs floor (feet vertical min). Assumes vertical axis is Y.
    Uses relation CoM height ~ 0.56 * body height.
    """
    base = trial_root + "/MoCap/body_pos_global/"
    comy_path = base + "center_of_mass_Y"
    if comy_path not in h5:
        return 1.70
    com_y = h5[comy_path][:]
    # feet vertical (Y) for both feet
    try:
        ry = h5[base + "calcn_r_Y"][:]
        ly = h5[base + "calcn_l_Y"][:]
        T = min(len(com_y), len(ry), len(ly))
        floor_y = float(np.percentile(np.minimum(ry[:T], ly[:T]), 5))
        com_rel = float(np.median(com_y[:T] - floor_y))
    except Exception:
        com_rel = float(np.median(com_y))
    height_est = com_rel / 0.56 if com_rel > 0 else 1.70
    # clamp to reasonable human bounds
    return float(np.clip(height_est, 1.45, 2.10))


def build_or_get_subject_meta(h5: h5py.File, subject: str, trials_for_subj: List[str], cache: Dict[str, Dict[str, float]],
                              ld_vertical_is_fz: bool):
    if subject in cache:
        return cache[subject]
    # sample up to 5 trials for robustness
    sample = trials_for_subj[:5]
    weights = []
    heights = []
    for tr in sample:
        try:
            w = estimate_weight_from_grf(h5, tr, ld_vertical_is_fz=ld_vertical_is_fz)
            h = estimate_height_from_com_and_feet(h5, tr)
            if np.isfinite(w) and w > 20 and w < 140:
                weights.append(w)
            if np.isfinite(h):
                heights.append(h)
        except Exception:
            continue
    if not weights:
        weights = [70.0]
    if not heights:
        heights = [1.70]
    meta = {"height_m": float(np.median(heights)), "weight_kg": float(np.median(weights))}
    cache[subject] = meta
    print(f"[auto-meta] {subject}: height={meta['height_m']:.3f} m, weight={meta['weight_kg']:.1f} kg (estimated)")
    return meta


def load_meta(meta_path: str) -> Dict[str, Dict[str, float]]:
    if not meta_path:
        return {}
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # expected: {"S006": {"height_m": 1.70, "weight_kg": 68.0}, ...}
    return meta


def main():
    ap = argparse.ArgumentParser(description="Convert LD .h5 trials to GaitDynamics preprocessed (GDP) format")
    ap.add_argument("--input", required=True, help="Path to combined_data.h5 or per-subject H5")
    ap.add_argument("--subject", default=None, help="Filter subject id, e.g., S006")
    ap.add_argument("--out", required=True, help="Output directory for GDP per-trial npz")
    ap.add_argument("--meta", default=None, help="JSON with subject height/weight: {subject: {height_m, weight_kg}}")
    ap.add_argument("--auto_meta", action="store_true", help="Auto-estimate missing subject meta (height/weight) from GRF and CoM")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N trials (for quick sanity runs)")
    ap.add_argument("--target_fs", type=int, default=100)
    ap.add_argument("--window_len", type=int, default=150)
    ap.add_argument("--align_mdir", action="store_true")
    ap.add_argument("--skip_cop_distance_check", action="store_true")
    ap.add_argument("--grf_scale", type=float, default=1.0, help="Multiply GRF by this factor before dividing by weight_kg (e.g., 9.81 if source is kgf)")
    ap.add_argument("--ld_vertical", choices=["fy", "fz"], default="fz",
                    help="Which axis is vertical in LD measured GRF/CoP. Will be remapped to AB convention (Fy up). Default: fz")
    ap.add_argument("--flip_dofs", type=str, default="",
                    help="Comma-separated DOF names to flip sign for (e.g., 'pelvis_tx,knee_angle_r,knee_angle_l')")
    ap.add_argument("--cop_sign", type=str, default=None,
                    help="Global CoP sign as '+,-,+' or '1,-1,1' for (x,y,z). Applied to both sides unless side-specific provided.")
    ap.add_argument("--cop_sign_r", type=str, default=None, help="Right CoP sign override as '+,-,+'")
    ap.add_argument("--cop_sign_l", type=str, default=None, help="Left CoP sign override as '+,-,+'")
    args = ap.parse_args()
    def parse_sign(s: str):
        if s is None:
            return None
        parts = [p.strip() for p in s.split(',')]
        if len(parts) != 3:
            raise ValueError("cop_sign must have 3 comma-separated values")
        out = []
        for p in parts:
            if p in ['+','+1','1']:
                out.append(1.0)
            elif p in ['-','-1']:
                out.append(-1.0)
            else:
                out.append(float(p))
        return np.array(out, dtype=np.float32)

    global_sign = parse_sign(args.cop_sign)
    sign_r = parse_sign(args.cop_sign_r) if args.cop_sign_r else None
    sign_l = parse_sign(args.cop_sign_l) if args.cop_sign_l else None
    if global_sign is not None:
        if sign_r is None:
            sign_r = global_sign.copy()
        if sign_l is None:
            sign_l = global_sign.copy()
    # default to no-op if none specified
    if sign_r is None:
        sign_r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if sign_l is None:
        sign_l = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    os.makedirs(args.out, exist_ok=True)
    opt = build_opt(args.target_fs, args.window_len)
    meta = load_meta(args.meta)

    with h5py.File(args.input, "r") as h5:
        trials = list_trials(h5, subject=args.subject)
        if not trials:
            print("[warn] No trials found under", args.subject or "(all)")
            return
        # group trials by subject for auto-meta
        trials_by_subj: Dict[str, List[str]] = {}
        for tr in trials:
            subj = tr.split("/")[1]
            trials_by_subj.setdefault(subj, []).append(tr)
        auto_meta_cache: Dict[str, Dict[str, float]] = {}
        count = 0
        for trial_root in trials:
            subject = trial_root.split("/")[1]
            cond = trial_root.split("/")[2]
            trial = trial_root.split("/")[3]

            if subject in meta:
                height_m = meta[subject]["height_m"]
                weight_kg = meta[subject]["weight_kg"]
            else:
                if args.auto_meta:
                    sub_meta = build_or_get_subject_meta(h5, subject, trials_by_subj.get(subject, []), auto_meta_cache,
                                                          ld_vertical_is_fz=(args.ld_vertical == "fz"))
                    height_m = sub_meta["height_m"]
                    weight_kg = sub_meta["weight_kg"]
                else:
                    print(f"[skip] {trial_root}: missing height/weight in meta; provide --meta or enable --auto_meta")
                    continue

            out_dir = os.path.join(args.out, subject, cond)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{trial}.npz")

            result = process_trial(
                h5,
                opt,
                trial_root,
                height_m,
                weight_kg,
                resample_to=args.target_fs,
                align_mdir=args.align_mdir,
                check_cop_dist=(not args.skip_cop_distance_check),
                grf_scale=args.grf_scale,
                ld_vertical_is_fz=(args.ld_vertical == "fz"),
                flip_dofs=[s.strip() for s in args.flip_dofs.split(',') if s.strip()],
                cop_sign_r=sign_r,
                cop_sign_l=sign_l,
            )
            if result.get("skip"):
                print(f"[skip] {trial_root}: {result['reason']}")
                continue

            np.savez_compressed(
                out_path,
                model_states=result["model_states"],
                model_states_columns=np.array(result["model_states_columns"], dtype=object),
                height_m=result["height_m"],
                weight_kg=result["weight_kg"],
                pos_vec=np.array(result["pos_vec"], dtype=np.float32),
                probably_missing=result["probably_missing"],
                sampling_rate=result["sampling_rate"],
            )
            print(f"[ok] wrote {out_path}")
            count += 1
            if args.limit is not None and count >= args.limit:
                break

    print("[done] GDP export complete →", args.out)


if __name__ == "__main__":
    main()
