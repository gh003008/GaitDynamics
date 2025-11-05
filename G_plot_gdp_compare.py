#!/usr/bin/env python3
import argparse
import os
import json
import glob
from datetime import datetime
import tempfile
import re
import numpy as np
import torch
from consts import OSIM_DOF_ALL, JOINTS_3D_ALL, KINETICS_ALL
from model.utils import inverse_convert_addb_state_to_model_input


def resolve_from_base(base, rel):
    # allow direct .npz path or relative pattern under base
    if rel.endswith('.npz'):
        candidate = rel if os.path.isabs(rel) else os.path.join(base, rel)
        return candidate if os.path.exists(candidate) else None
    # try exact join with .npz
    candidate = os.path.join(base, rel + '.npz')
    if os.path.exists(candidate):
        return candidate
    # fallback: glob search by substring pattern
    pat = os.path.join(base, '**', f"*{rel}*.npz")
    matches = glob.glob(pat, recursive=True)
    return matches[0] if matches else None


def load_gdp(path):
    npz = np.load(path, allow_pickle=True)
    ms = npz["model_states"]
    cols = [str(x) for x in npz["model_states_columns"]]
    fs = int(npz["sampling_rate"]) if "sampling_rate" in npz else 100
    height_m = float(npz["height_m"]) if "height_m" in npz else 1.7
    pos_vec = npz["pos_vec"] if "pos_vec" in npz else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return ms, cols, fs, height_m, pos_vec


def reconstruct_q23(model_states, model_cols, joints_3d, osim_dofs, pos_vec, height_m, fs):
    ms_t = torch.from_numpy(model_states).float().unsqueeze(0)
    q_full = inverse_convert_addb_state_to_model_input(ms_t, model_cols, joints_3d, osim_dofs, pos_vec, torch.tensor(height_m), sampling_fre=fs)
    q_full = q_full[0].cpu().numpy()
    # first 23 are DOF q; the rest are kinetics
    return q_full[:, :23], q_full


def extract_grf_cop(model_states, model_cols, side="right"):
    # Use KINETICS_ALL names; pick side or sum
    col_idx = {c: i for i, c in enumerate(model_cols)}
    rF = np.stack([model_states[:, col_idx['calcn_r_force_vx']],
                   model_states[:, col_idx['calcn_r_force_vy']],
                   model_states[:, col_idx['calcn_r_force_vz']]], axis=1)
    lF = np.stack([model_states[:, col_idx['calcn_l_force_vx']],
                   model_states[:, col_idx['calcn_l_force_vy']],
                   model_states[:, col_idx['calcn_l_force_vz']]], axis=1)
    rC = np.stack([model_states[:, col_idx['calcn_r_force_normed_cop_x']],
                   model_states[:, col_idx['calcn_r_force_normed_cop_y']],
                   model_states[:, col_idx['calcn_r_force_normed_cop_z']]], axis=1)
    lC = np.stack([model_states[:, col_idx['calcn_l_force_normed_cop_x']],
                   model_states[:, col_idx['calcn_l_force_normed_cop_y']],
                   model_states[:, col_idx['calcn_l_force_normed_cop_z']]], axis=1)
    if side == 'right':
        return rF, rC
    if side == 'left':
        return lF, lC
    # sum/combined: sum forces, average cop (force-weighted by Fy)
    F = rF + lF
    Fy = np.clip(F[:, 1:2], 1e-6, None)
    C = (rC * np.abs(rF[:, 1:2]) + lC * np.abs(lF[:, 1:2])) / np.maximum(np.abs(rF[:, 1:2]) + np.abs(lF[:, 1:2]), 1e-6)
    return F, C


def main():
    ap = argparse.ArgumentParser(description="Plot comparison of two GDP files (AB vs LD) with 29 traces: q(23)+GRF(3)+COP(3)")
    ap.add_argument('--gdp_a', default=None, help='Path to GDP npz A (e.g., from addbiomechanics or LD pipeline)')
    ap.add_argument('--gdp_b', default=None, help='Optional path to GDP npz B for overlay comparison')
    ap.add_argument('--config', default=None, help='JSON config with base paths and defaults')
    ap.add_argument('--ab_rel', default=None, help='Relative selector for A under config.ab_base (e.g., "Subj06/Subj06_land_segment_0")')
    ap.add_argument('--ld_rel', default=None, help='Relative selector for B under config.ld_base (e.g., "S004/cadence_120p/trial_01")')
    ap.add_argument('--t0', type=float, default=0.0, help='Start time in seconds')
    ap.add_argument('--t1', type=float, default=None, help='End time in seconds')
    ap.add_argument('--side', choices=['right','left','sum','both'], default='right', help='Which GRF/COP to plot; use "both" to overlay R and L in the same axes')
    ap.add_argument('--title_a', default='A', help='Legend name for A')
    ap.add_argument('--title_b', default='B', help='Legend name for B')
    ap.add_argument('--save', default=None, help='If provided, save figure to this path instead of showing')
    # New: direct AB/LD sources and batch options
    ap.add_argument('--ab_b3d', default=None, help='Path to AB SubjectOnDisk .b3d (e.g., /.../p9/p9.b3d)')
    ap.add_argument('--ld_h5', default=None, help='Path to LD subject .h5 (e.g., /.../S004.h5)')
    ap.add_argument('--trial_keyword', default='walk', help='Filter trials containing this keyword (case-insensitive), e.g., "walk"')
    ap.add_argument('--out_dir', default='./plots/G_plot_gdp_compare_plot', help='Directory to save generated plots')
    ap.add_argument('--limit_pairs', type=int, default=None, help='Limit number of AB/LD trial pairs to plot')
    ap.add_argument('--target_fs', type=int, default=100, help='Target sampling rate for GDP conversion (when using --ab_b3d/--ld_h5)')
    ap.add_argument('--align_mdir', action='store_true', help='Align moving direction during GDP conversion')
    ap.add_argument('--ld_vertical', choices=['fy','fz'], default='fz', help='Vertical axis for LD measured GRF (will be remapped to Fy-up)')
    # Backend selection needs to be applied before importing pyplot
    ap.add_argument('--backend', default=None, help='Matplotlib backend to use (e.g., TkAgg, Qt5Agg, WebAgg)')
    ap.add_argument('--web', action='store_true', help='Shortcut to use WebAgg backend (HTTP-based interactive)')
    ap.add_argument('--web_port', type=int, default=8988, help='Port for WebAgg (when using --backend WebAgg or --web)')
    ap.add_argument('--web_address', default='127.0.0.1', help='Bind address for WebAgg (e.g., 0.0.0.0 for LAN)')
    ap.add_argument('--open_browser', action='store_true', help='Ask WebAgg to open a browser automatically (if possible)')
    ap.add_argument('--html', default=None, help='Save interactive HTML (uses mpld3 if installed)')

    def _safe_id(s: str) -> str:
        s = s.strip().replace(' ', '_')
        s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
        # collapse repeats
        s = re.sub(r"_+", "_", s)
        return s.strip('_')[:120]

    def _ts_no_year() -> str:
        # mmddHHMMSS (no year)
        return datetime.now().strftime('%m%d%H%M%S')

    def make_plot(gdp_a_path, gdp_b_path, args, save_path):
        import matplotlib as mpl
        if getattr(args, 'web', False) and not getattr(args, 'backend', None):
            args.backend = 'WebAgg'
        if getattr(args, 'backend', None):
            try:
                mpl.use(args.backend)
            except Exception as e:
                print(f"[warn] Failed to set backend '{args.backend}': {e}")
        if (getattr(args, 'backend', '') and args.backend.lower() == 'webagg') or getattr(args, 'web', False):
            try:
                from matplotlib import rcParams
                rcParams['webagg.open_in_browser'] = getattr(args, 'open_browser', False)
                rcParams['webagg.address'] = getattr(args, 'web_address', '127.0.0.1')
                rcParams['webagg.port'] = getattr(args, 'web_port', 8988)
            except Exception as e:
                print(f"[warn] Could not set WebAgg rcParams: {e}")
        import matplotlib.pyplot as plt

        msA, colsA, fsA, hA, posA = load_gdp(gdp_a_path)
        msB, colsB, fsB, hB, posB = (None, None, None, None, None)
        if gdp_b_path:
            msB, colsB, fsB, hB, posB = load_gdp(gdp_b_path)

        if msB is not None and fsA != fsB:
            print(f"[warn] sampling rates differ: A={fsA}, B={fsB}; plotting by index with A's fs")
        fs = fsA

        osim_dofs = OSIM_DOF_ALL[:23] + KINETICS_ALL
        joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis','hip_r','hip_l','lumbar']}

        q23A, qfullA = reconstruct_q23(msA, colsA, joints_3d, osim_dofs, posA, hA, fsA)
        q23B = qfullB = None
        if msB is not None:
            q23B, qfullB = reconstruct_q23(msB, colsB, joints_3d, osim_dofs, posB, hB, fsB)

        side = getattr(args, 'side', 'right')
        if side == 'both':
            rFA, rCA = extract_grf_cop(msA, colsA, side='right')
            lFA, lCA = extract_grf_cop(msA, colsA, side='left')
            if msB is not None:
                rFB, rCB = extract_grf_cop(msB, colsB, side='right')
                lFB, lCB = extract_grf_cop(msB, colsB, side='left')
        else:
            FA, CA = extract_grf_cop(msA, colsA, side=side)
            FB = CB = None
            if msB is not None:
                FB, CB = extract_grf_cop(msB, colsB, side=side)

        if side == 'both':
            T = q23A.shape[0]
            T = min(T, rFA.shape[0], lFA.shape[0])
            if msB is not None:
                T = min(T, rFB.shape[0], lFB.shape[0], q23B.shape[0])
            q23A = q23A[:T]
            if msB is not None:
                q23B = q23B[:T]
            rFA, lFA, rCA, lCA = rFA[:T], lFA[:T], rCA[:T], lCA[:T]
            if msB is not None:
                rFB, lFB, rCB, lCB = rFB[:T], lFB[:T], rCB[:T], lCB[:T]
        else:
            T = q23A.shape[0]
            T = min(T, FA.shape[0], CA.shape[0])
            if msB is not None:
                T = min(T, q23B.shape[0], FB.shape[0], CB.shape[0])
            q23A = q23A[:T]
            if msB is not None:
                q23B = q23B[:T]
            FA, CA = FA[:T], CA[:T]
            if msB is not None:
                FB, CB = FB[:T], CB[:T]

        # time window
        t0 = getattr(args, 't0', 0.0) or 0.0
        t1 = getattr(args, 't1', None)
        s = int(t0 * fs)
        e = int(t1 * fs) if t1 else T
        s = max(0, min(s, T-1)); e = max(s+1, min(e, T))
        t = np.arange(s, e) / fs

        n_plots = 23 + 3 + 3
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 1.2), sharex=True)
        # q(23)
        title_a = getattr(args, 'title_a', 'A')
        title_b = getattr(args, 'title_b', 'B')
        for i in range(23):
            ax = axes[i]
            ax.plot(t, q23A[s:e, i], label=title_a, color='tab:blue', lw=1)
            if q23B is not None:
                ax.plot(t, q23B[s:e, i], label=title_b, color='tab:orange', lw=1, alpha=0.8)
            ax.set_ylabel(OSIM_DOF_ALL[i], fontsize=8)
        # GRF(3)
        names_grf = ['Fx','Fy','Fz']
        for i in range(3):
            ax = axes[23+i]
            if side == 'both':
                ax.plot(t, rFA[s:e, i], label=f'{title_a} R', color='tab:blue', lw=1)
                ax.plot(t, lFA[s:e, i], label=f'{title_a} L', color='tab:blue', lw=1, linestyle='--')
                if msB is not None:
                    ax.plot(t, rFB[s:e, i], label=f'{title_b} R', color='tab:orange', lw=1)
                    ax.plot(t, lFB[s:e, i], label=f'{title_b} L', color='tab:orange', lw=1, linestyle='--')
                ax.set_ylabel(f'GRF R/L {names_grf[i]}', fontsize=8)
            else:
                ax.plot(t, FA[s:e, i], label=title_a, color='tab:blue', lw=1)
                if msB is not None:
                    ax.plot(t, FB[s:e, i], label=title_b, color='tab:orange', lw=1, alpha=0.8)
                ax.set_ylabel(f'GRF {side} {names_grf[i]}', fontsize=8)
        # COP(3) normalized
        names_cop = ['nx','ny','nz']
        for i in range(3):
            ax = axes[26+i]
            if side == 'both':
                ax.plot(t, rCA[s:e, i], label=f'{title_a} R', color='tab:blue', lw=1)
                ax.plot(t, lCA[s:e, i], label=f'{title_a} L', color='tab:blue', lw=1, linestyle='--')
                if msB is not None:
                    ax.plot(t, rCB[s:e, i], label=f'{title_b} R', color='tab:orange', lw=1)
                    ax.plot(t, lCB[s:e, i], label=f'{title_b} L', color='tab:orange', lw=1, linestyle='--')
                ax.set_ylabel(f'CoP_norm R/L {names_cop[i]}', fontsize=8)
            else:
                ax.plot(t, CA[s:e, i], label=title_a, color='tab:blue', lw=1)
                if msB is not None:
                    ax.plot(t, CB[s:e, i], label=title_b, color='tab:orange', lw=1, alpha=0.8)
                ax.set_ylabel(f'CoP_norm {side} {names_cop[i]}', fontsize=8)

        axes[-1].set_xlabel('Time (s)')
        if q23B is not None:
            axes[0].legend(loc='upper right', ncol=2, fontsize=8)
        if side == 'both':
            axes[23].legend(loc='upper right', ncol=2 if msB is None else 4, fontsize=8)
        import os as _os
        _os.makedirs(_os.path.dirname(save_path) or '.', exist_ok=True)
        import matplotlib.pyplot as _plt
        _plt.tight_layout()
        _plt.savefig(save_path, dpi=200)
        _plt.close(fig)
        print(f"[ok] saved figure to {save_path}")


    def _build_ab_walk_gdps(b3d_path: str, keyword: str, target_fs: int, align_mdir: bool, skip_cop_check: bool, tmp_dir: str):
        """Build GDP npz files for AB .b3d trials matching keyword. Returns list of (npz_path, label)."""
        try:
            from G_b3d_to_gdp import subject_to_trials, to_gdp
        except Exception as e:
            raise RuntimeError(f"Failed to import AB converter (G_b3d_to_gdp.py). Install nimblephysics? Error: {e}")
        trials = subject_to_trials(b3d_path)
        if not trials:
            return []
        kw = (keyword or '').lower()
        subj = os.path.splitext(os.path.basename(b3d_path))[0]

        def build_from_deterministic(trials_iter):
            cands = list(trials_iter)
            if not cands:
                return []
            print("[list][AB] candidates:")
            for idx, tr in enumerate(cands, start=1):
                print(f"  [{idx}] {tr.get('name','')}")
            # pick 3rd (index 2), or last if fewer than 3
            start_idx = 2 if len(cands) >= 3 else len(cands) - 1
            subdir = os.path.join(tmp_dir, 'ab')
            os.makedirs(subdir, exist_ok=True)
            out_local = []
            for tr in cands[start_idx:]:  # try 3rd, then 4th, ... until one succeeds
                name = tr.get('name','')
                data, reason = to_gdp(tr, target_fs, align_mdir, skip_cop_check)
                if data is None:
                    print(f"[skip][AB] {name}: {reason}")
                    continue
                label = f"{subj}_{name}"
                npz_path = os.path.join(subdir, _safe_id(label) + '.npz')
                np.savez_compressed(
                    npz_path,
                    model_states=data['model_states'],
                    model_states_columns=np.array(data['model_states_columns'], dtype=object),
                    height_m=data['height_m'],
                    weight_kg=data['weight_kg'],
                    pos_vec=np.array(data['pos_vec'], dtype=np.float32),
                    sampling_rate=data['sampling_rate'],
                    probably_missing=data['probably_missing'],
                )
                out_local.append((npz_path, label))
                break  # use only one deterministically
            return out_local

        # 1) try keyword filter
        selected = [tr for tr in trials if (not kw or kw in tr.get('name','').lower())]
        out = build_from_deterministic(selected)
        if out:
            return out
        # 2) fallback: exclude obvious non-walking like standing/calibration
        fallback = [tr for tr in trials if not re.search(r"stand|calib", tr.get('name','').lower())]
        print(f"[info] AB keyword '{kw}' matched 0; fallback to {len(fallback)} trial(s) excluding standing/calib")
        return build_from_deterministic(fallback)


    def _build_ld_walk_gdps(h5_path: str, keyword: str, target_fs: int, align_mdir: bool, ld_vertical: str, tmp_dir: str):
        """Build GDP npz files for LD .h5 trials matching keyword. Returns list of (npz_path, label)."""
        from G_LD_to_pipeline import (
            build_opt, list_trials, build_or_get_subject_meta,
            process_trial,
        )
        import h5py
        if ld_vertical not in ('fy','fz'):
            ld_vertical = 'fz'
        opt = build_opt(target_fs, window_len=150)
        out = []
        kw = (keyword or '').lower()
        with h5py.File(h5_path, 'r') as h5:
            trials_all = list_trials(h5)
            sel = [tr for tr in trials_all if (not kw or kw in tr.lower())]
            if not sel and kw:
                # First: prefer speed-based level trials (walk-like)
                speed_pat = re.compile(r"/level_(?:08|12|16)mps/")
                level_sel = [tr for tr in trials_all if speed_pat.search(tr)]
                if level_sel:
                    print(f"[info] LD keyword '{kw}' matched 0; using level speed trials ({len(level_sel)} found)")
                    sel = level_sel
                else:
                    # Fallback: exclude standing/calib
                    sel = [tr for tr in trials_all if not re.search(r"stand|calib", tr.lower())]
                    print(f"[info] LD keyword '{kw}' matched 0; fallback to {len(sel)} trial(s) excluding standing/calib")
            # If no kw provided and still empty for some reason, try speed-based
            if not sel:
                speed_pat = re.compile(r"/level_(?:08|12|16)mps/")
                sel = [tr for tr in trials_all if speed_pat.search(tr)]
                if sel:
                    print(f"[info] LD fallback to level speed trials: {len(sel)} found")
            trials = sel
            if not trials:
                return []
            # list candidates and pick 3rd deterministically
            print("[list][LD] candidates:")
            for idx, tr in enumerate(trials, start=1):
                print(f"  [{idx}] {tr}")
            pick_idx = 2 if len(trials) >= 3 else len(trials)-1
            # Try from the 3rd forward; if none succeed, we'll wrap and try earlier ones
            trials = trials[pick_idx:] + trials[:pick_idx]
            # group by subject for meta estimation
            trials_by_subj = {}
            for tr in trials:
                subj = tr.split('/')[1]
                trials_by_subj.setdefault(subj, []).append(tr)
            meta_cache = {}
            for tr in trials:
                subject = tr.split('/')[1]
                cond = tr.split('/')[2] if len(tr.split('/'))>2 else 'cond'
                trial_name = tr.split('/')[3] if len(tr.split('/'))>3 else os.path.basename(tr)
                meta = build_or_get_subject_meta(h5, subject, trials_by_subj.get(subject, []), meta_cache, ld_vertical_is_fz=(ld_vertical=='fz'))
                res = process_trial(
                    h5, opt, tr,
                    height_m=meta['height_m'], weight_kg=meta['weight_kg'],
                    resample_to=target_fs, align_mdir=align_mdir,
                    check_cop_dist=True, grf_scale=1.0,
                    ld_vertical_is_fz=(ld_vertical=='fz'), flip_dofs=[],
                    cop_sign_r=np.array([1.0,1.0,1.0], dtype=np.float32),
                    cop_sign_l=np.array([1.0,1.0,1.0], dtype=np.float32),
                )
                if res.get('skip'):
                    print(f"[skip][LD] {tr}: {res['reason']}")
                    continue
                subdir = os.path.join(tmp_dir, 'ld')
                os.makedirs(subdir, exist_ok=True)
                label = f"{subject}_{cond}_{trial_name}"
                npz_path = os.path.join(subdir, _safe_id(label) + '.npz')
                np.savez_compressed(
                    npz_path,
                    model_states=res['model_states'],
                    model_states_columns=np.array(res['model_states_columns'], dtype=object),
                    height_m=res['height_m'],
                    weight_kg=res['weight_kg'],
                    pos_vec=np.array(res['pos_vec'], dtype=np.float32),
                    sampling_rate=res['sampling_rate'],
                    probably_missing=res['probably_missing'],
                )
                out.append((npz_path, label))
                break  # only one deterministically
        return out
    args = ap.parse_args()

    # Choose matplotlib backend before importing pyplot
    import matplotlib as mpl
    if args.web and not args.backend:
        args.backend = 'WebAgg'
    if args.backend:
        try:
            mpl.use(args.backend)
        except Exception as e:
            print(f"[warn] Failed to set backend '{args.backend}': {e}")
    # Configure WebAgg if selected
    if (args.backend and args.backend.lower() == 'webagg') or args.web:
        try:
            from matplotlib import rcParams
            rcParams['webagg.open_in_browser'] = args.open_browser
            rcParams['webagg.address'] = args.web_address
            rcParams['webagg.port'] = args.web_port
            print(f"[info] WebAgg configured at http://{args.web_address}:{args.web_port}/ (open in your browser)")
            if args.web_address == '0.0.0.0':
                print("[info] Listening on all interfaces; use your server IP instead of 0.0.0.0 in the URL.")
        except Exception as e:
            print(f"[warn] Could not set WebAgg rcParams: {e}")
    import matplotlib.pyplot as plt

    # If user provided AB/LD raw sources, build GDPs and batch plot
    if args.ab_b3d and args.ld_h5:
        tmp_root = tempfile.mkdtemp(prefix='gdp_tmp_')
        print(f"[info] Building temporary GDPs under {tmp_root} ...")
        try:
            ab_list = _build_ab_walk_gdps(args.ab_b3d, args.trial_keyword, args.target_fs, args.align_mdir, False, tmp_root)
        except Exception as e:
            print(f"[error] AB conversion failed: {e}")
            ab_list = []
        try:
            ld_list = _build_ld_walk_gdps(args.ld_h5, args.trial_keyword, args.target_fs, args.align_mdir, args.ld_vertical, tmp_root)
        except Exception as e:
            print(f"[error] LD conversion failed: {e}")
            ld_list = []
        if not ab_list:
            raise SystemExit("No AB trials matched the keyword (or conversion failed).")
        if not ld_list:
            raise SystemExit("No LD trials matched the keyword (or conversion failed).")
        n = min(len(ab_list), len(ld_list))
        if args.limit_pairs is not None:
            n = min(n, args.limit_pairs)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[info] Pairing {n} trial(s) → saving plots to {args.out_dir}")
        for i in range(n):
            (ab_npz, ab_label) = ab_list[i]
            (ld_npz, ld_label) = ld_list[i]
            ts = _ts_no_year()
            fname = f"LD_{_safe_id(ld_label)}_AB_{_safe_id(ab_label)}_{ts}.png"
            save_path = os.path.join(args.out_dir, fname)
            args.title_a = 'LD'
            args.title_b = 'AB'
            # Note: we want LD as A or B? Keep LD as A for naming consistency here.
            make_plot(ld_npz, ab_npz, args, save_path)
        print(f"[done] Saved {n} plot(s) in {args.out_dir}")
        return

    # config defaults
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    ab_base = cfg.get('ab_base')
    ld_base = cfg.get('ld_base')
    if args.ab_rel and ab_base:
        resolved = resolve_from_base(ab_base, args.ab_rel)
        if not resolved:
            raise FileNotFoundError(f"Could not resolve ab_rel '{args.ab_rel}' under base '{ab_base}'")
        args.gdp_a = resolved
    if args.ld_rel and ld_base:
        resolved = resolve_from_base(ld_base, args.ld_rel)
        if not resolved:
            raise FileNotFoundError(f"Could not resolve ld_rel '{args.ld_rel}' under base '{ld_base}'")
        args.gdp_b = resolved

    if args.t0 is None and 'default_t0' in cfg:
        args.t0 = cfg['default_t0']
    if args.t1 is None and 'default_t1' in cfg:
        args.t1 = cfg['default_t1']
    if args.side == 'right' and 'default_side' in cfg:
        args.side = cfg['default_side']

    if not args.gdp_a:
        raise SystemExit("--gdp_a is required unless --config with --ab_rel is provided and resolves to a file")
    msA, colsA, fsA, hA, posA = load_gdp(args.gdp_a)
    msB = colsB = fsB = hB = posB = None
    if args.gdp_b:
        msB, colsB, fsB, hB, posB = load_gdp(args.gdp_b)

    if msB is not None and fsA != fsB:
        print(f"[warn] sampling rates differ: A={fsA}, B={fsB}; plotting by index with A's fs")
    fs = fsA

    osim_dofs = OSIM_DOF_ALL[:23] + KINETICS_ALL
    joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis','hip_r','hip_l','lumbar']}

    q23A, qfullA = reconstruct_q23(msA, colsA, joints_3d, osim_dofs, posA, hA, fsA)
    q23B = qfullB = None
    if msB is not None:
        q23B, qfullB = reconstruct_q23(msB, colsB, joints_3d, osim_dofs, posB, hB, fsB)

    if args.side == 'both':
        # extract both sides separately
        rFA, rCA = extract_grf_cop(msA, colsA, side='right')
        lFA, lCA = extract_grf_cop(msA, colsA, side='left')
        if msB is not None:
            rFB, rCB = extract_grf_cop(msB, colsB, side='right')
            lFB, lCB = extract_grf_cop(msB, colsB, side='left')
    else:
        FA, CA = extract_grf_cop(msA, colsA, side=args.side)
        FB = CB = None
        if msB is not None:
            FB, CB = extract_grf_cop(msB, colsB, side=args.side)

    if args.side == 'both':
        T = q23A.shape[0]
        T = min(T, rFA.shape[0], lFA.shape[0])
        if msB is not None:
            T = min(T, rFB.shape[0], lFB.shape[0], q23B.shape[0])
        q23A = q23A[:T]
        if msB is not None:
            q23B = q23B[:T]
        rFA, lFA, rCA, lCA = rFA[:T], lFA[:T], rCA[:T], lCA[:T]
        if msB is not None:
            rFB, lFB, rCB, lCB = rFB[:T], lFB[:T], rCB[:T], lCB[:T]
    else:
        T = q23A.shape[0]
        T = min(T, FA.shape[0], CA.shape[0])
        if msB is not None:
            T = min(T, q23B.shape[0], FB.shape[0], CB.shape[0])
        q23A = q23A[:T]
        if msB is not None:
            q23B = q23B[:T]
        FA, CA = FA[:T], CA[:T]
        if msB is not None:
            FB, CB = FB[:T], CB[:T]

    # time window
    s = int(args.t0 * fs)
    e = int(args.t1 * fs) if args.t1 else T
    s = max(0, min(s, T-1)); e = max(s+1, min(e, T))

    t = np.arange(s, e) / fs

    n_plots = 23 + 3 + 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 1.2), sharex=True)
    # q(23)
    for i in range(23):
        ax = axes[i]
        ax.plot(t, q23A[s:e, i], label=args.title_a, color='tab:blue', lw=1)
        if q23B is not None:
            ax.plot(t, q23B[s:e, i], label=args.title_b, color='tab:orange', lw=1, alpha=0.8)
        ax.set_ylabel(OSIM_DOF_ALL[i], fontsize=8)
    # GRF(3)
    names_grf = ['Fx','Fy','Fz']
    for i in range(3):
        ax = axes[23+i]
        if args.side == 'both':
            ax.plot(t, rFA[s:e, i], label=f'{args.title_a} R', color='tab:blue', lw=1)
            ax.plot(t, lFA[s:e, i], label=f'{args.title_a} L', color='tab:blue', lw=1, linestyle='--')
            if msB is not None:
                ax.plot(t, rFB[s:e, i], label=f'{args.title_b} R', color='tab:orange', lw=1)
                ax.plot(t, lFB[s:e, i], label=f'{args.title_b} L', color='tab:orange', lw=1, linestyle='--')
            ax.set_ylabel(f'GRF R/L {names_grf[i]}', fontsize=8)
        else:
            ax.plot(t, FA[s:e, i], label=args.title_a, color='tab:blue', lw=1)
            if msB is not None:
                ax.plot(t, FB[s:e, i], label=args.title_b, color='tab:orange', lw=1, alpha=0.8)
            ax.set_ylabel(f'GRF {args.side} {names_grf[i]}', fontsize=8)
    # COP(3) normalized
    names_cop = ['nx','ny','nz']
    for i in range(3):
        ax = axes[26+i]
        if args.side == 'both':
            ax.plot(t, rCA[s:e, i], label=f'{args.title_a} R', color='tab:blue', lw=1)
            ax.plot(t, lCA[s:e, i], label=f'{args.title_a} L', color='tab:blue', lw=1, linestyle='--')
            if msB is not None:
                ax.plot(t, rCB[s:e, i], label=f'{args.title_b} R', color='tab:orange', lw=1)
                ax.plot(t, lCB[s:e, i], label=f'{args.title_b} L', color='tab:orange', lw=1, linestyle='--')
            ax.set_ylabel(f'CoP_norm R/L {names_cop[i]}', fontsize=8)
        else:
            ax.plot(t, CA[s:e, i], label=args.title_a, color='tab:blue', lw=1)
            if msB is not None:
                ax.plot(t, CB[s:e, i], label=args.title_b, color='tab:orange', lw=1, alpha=0.8)
            ax.set_ylabel(f'CoP_norm {args.side} {names_cop[i]}', fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    # Legend: for q panes, keep a compact legend if B exists; for both-sides, add legend on first GRF axis
    if q23B is not None:
        axes[0].legend(loc='upper right', ncol=2, fontsize=8)
    if args.side == 'both':
        axes[23].legend(loc='upper right', ncol=2 if msB is None else 4, fontsize=8)
    plt.tight_layout()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"[ok] saved figure to {args.save}")
    # Save interactive HTML if requested (best-effort)
    if args.html:
        try:
            import mpld3
            os.makedirs(os.path.dirname(args.html) or '.', exist_ok=True)
            mpld3.save_html(fig, args.html)
            print(f"[ok] saved interactive HTML to {args.html}")
        except Exception as e:
            print(f"[warn] Could not save interactive HTML: {e}. Try: pip install mpld3")
    else:
        # Try to show interactive plot with pan/zoom if backend allows
        try:
            plt.show()
        except Exception as e:
            tmp = './plots/tmp_plot.png'
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            plt.savefig(tmp, dpi=200)
            print(f"[warn] Could not open interactive window ({e}). Saved to {tmp} instead.")


if __name__ == '__main__':
    main()
