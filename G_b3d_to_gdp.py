#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict

import nimblephysics as nimble

from consts import OSIM_DOF_ALL, KINETICS_ALL, JOINTS_3D_ALL
from model.utils import (
    linear_resample_data,
    data_filter,
    norm_cops,
    align_moving_direction,
    convert_addb_state_to_model_input,
)


def subject_to_trials(b3d_path: str):
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
    dset_name = os.path.basename(os.path.dirname(os.path.dirname(b3d_path))) or 'AB'
    subject_name = os.path.splitext(os.path.basename(b3d_path))[0]
    weight_kg = subject.getMassKg()
    height_m = subject.getHeightM()
    skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/data/Geometry/")
    trials = []
    for trial_id in range(subject.getNumTrials()):
        name = subject.getTrialName(trial_id)
        fs = int(1.0 / subject.getTrialTimestep(trial_id))
        frames = subject.readFrames(trial_id, 0, subject.getTrialLength(trial_id), includeSensorData=False, includeProcessingPasses=True)
        if not frames or not frames[0].processingPasses:
            continue
        first = [fr.processingPasses[0] for fr in frames]
        poses = np.array([fr.pos for fr in first])  # q
        forces = np.array([fr.groundContactForce for fr in first])

        # CoP from subject API or force plates
        cops = np.array([fr.groundContactCenterOfPressure for fr in first])

        # Keep only calcn_r / calcn_l (two feet)
        contact_bodies = subject.getGroundForceBodies()
        if 'calcn_r' not in contact_bodies or 'calcn_l' not in contact_bodies:
            continue
        r_idx = contact_bodies.index('calcn_r')
        l_idx = contact_bodies.index('calcn_l')
        foot_idx = [r_idx * 3, r_idx * 3 + 1, r_idx * 3 + 2, l_idx * 3, l_idx * 3 + 1, l_idx * 3 + 2]
        forces = forces[:, foot_idx]
        cops = cops[:, foot_idx]

        # Build states: q(23) + rF(3) + rCOP(3) + lF(3) + lCOP(3)
        states = np.concatenate([poses, forces[:, :3] / weight_kg, cops[:, :3], forces[:, 3:] / weight_kg, cops[:, 3:]], axis=1)

        trials.append({
            'name': name,
            'fs': fs,
            'states': states,
            'skel': skel,
            'height_m': height_m,
            'weight_kg': weight_kg,
            'dset': dset_name,
            'subject': subject_name,
        })
    return trials


def to_gdp(tr, target_fs: int, align: bool, skip_cop_check: bool):
    states = tr['states']
    fs = tr['fs']
    weight_kg = tr['weight_kg']
    height_m = tr['height_m']
    skel = tr['skel']

    # Resample if needed
    if fs != target_fs:
        states = linear_resample_data(states, fs, target_fs)
        fs = target_fs

    # Filter
    states[:, :23] = data_filter(states[:, :23], 15, fs, 4)

    class Opt: pass
    opt = Opt()
    opt.osim_dof_columns = OSIM_DOF_ALL[:23] + KINETICS_ALL
    opt.joints_3d = {k: v for k, v in JOINTS_3D_ALL.items() if k in ['pelvis','hip_r','hip_l','lumbar']}
    opt.grf_osim_col_loc = [i for i,c in enumerate(opt.osim_dof_columns) if ('force' in c and '_cop_' not in c)]
    opt.cop_osim_col_loc = [i for i,c in enumerate(opt.osim_dof_columns) if '_cop_' in c]
    opt.kinematic_osim_col_loc = [i for i,c in enumerate(opt.osim_dof_columns) if 'force' not in c]

    # Normalize CoPs with model utils (uses nimble skel)
    states_nc = norm_cops(skel, states, opt, weight_kg, height_m, check_cop_to_calcn_distance=not skip_cop_check)
    if states_nc is False:
        return None, 'cop_far_from_foot'
    states = states_nc.numpy() if isinstance(states_nc, torch.Tensor) else states_nc

    # Align moving direction
    if align:
        aligned, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
        if aligned is False:
            return None, 'large_mdir_change'
        states = aligned.numpy()

    # Convert to model input
    df = pd.DataFrame(states, columns=opt.osim_dof_columns)
    df2, pos_vec = convert_addb_state_to_model_input(df, opt.joints_3d, fs)

    # Missing flag via GRF
    rFy = states[:, opt.grf_osim_col_loc[1]] if len(opt.grf_osim_col_loc)>=2 else states[:, -12+1]
    lFy = states[:, opt.grf_osim_col_loc[4]] if len(opt.grf_osim_col_loc)>=5 else states[:, -6+1]
    missing = (np.abs(rFy)+np.abs(lFy) < 1e-6)

    return {
        'model_states': df2.values.astype(np.float32),
        'model_states_columns': list(df2.columns),
        'height_m': float(height_m),
        'weight_kg': float(weight_kg),
        'pos_vec': [float(x) for x in pos_vec],
        'sampling_rate': fs,
        'probably_missing': missing.astype(bool),
    }, None


def main():
    ap = argparse.ArgumentParser(description='Convert .b3d (AddBiomechanics) to GDP npz files')
    ap.add_argument('--b3d', required=True, help='Path to a SubjectOnDisk .b3d')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--target_fs', type=int, default=100)
    ap.add_argument('--align_mdir', action='store_true')
    ap.add_argument('--skip_cop_distance_check', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trials = subject_to_trials(args.b3d)
    n = 0
    for tr in trials:
        data, reason = to_gdp(tr, args.target_fs, args.align_mdir, args.skip_cop_distance_check)
        if data is None:
            print(f"[skip] {tr['name']}: {reason}")
            continue
        dset = tr['dset']
        subj = tr['subject']
        out_dir = os.path.join(args.out, dset, subj)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{tr['name']}.npz")
        np.savez_compressed(
            out_path,
            model_states=data['model_states'],
            model_states_columns=np.array(data['model_states_columns'], dtype=object),
            height_m=data['height_m'],
            weight_kg=data['weight_kg'],
            pos_vec=np.array(data['pos_vec'], dtype=np.float32),
            sampling_rate=data['sampling_rate'],
            probably_missing=data['probably_missing'],
        )
        print('[ok] wrote', out_path)
        n += 1
        if args.limit is not None and n >= args.limit:
            break

    print('[done] b3d → GDP complete')


if __name__ == '__main__':
    main()
