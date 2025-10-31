#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure repo root is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import nimblephysics as nimble
except Exception as e:
    print('\n[오류] nimblephysics 모듈을 찾을 수 없습니다.\n' \
          ' - 이 스크립트는 AddBiomechanics의 .b3d 파일을 읽고 NimbleGUI로 보여주기 위해 nimblephysics가 필요합니다.\n' \
          ' - Python 3.10/3.11 기반의 별도 conda 환경을 권장합니다.\n' \
          f' - 내부 예외: {e}\n')
    sys.exit(2)

from figures.fig_utils import set_up_gui, show_skeletons  # noqa: E402
from consts import OSIM_DOF_ALL, KINETICS_ALL  # noqa: E402


def find_walk_trial(subject, substring='walk'):
    sub = substring.lower()
    for t in range(subject.getNumTrials()):
        name = subject.getTrialName(t)
        if sub in name.lower():
            return t, name
    return None, None


def build_opt_indices(osim_dof_columns):
    class Opt:
        pass
    opt = Opt()
    opt.osim_dof_columns = osim_dof_columns
    opt.grf_osim_col_loc = [i for i,c in enumerate(osim_dof_columns) if ('force' in c and '_cop_' not in c)]
    opt.cop_osim_col_loc = [i for i,c in enumerate(osim_dof_columns) if '_cop_' in c]
    opt.kinematic_osim_col_loc = [i for i,c in enumerate(osim_dof_columns) if 'force' not in c]
    return opt


def read_states(subject, trial_id):
    # Frames from first processing pass
    n_frames = subject.getTrialLength(trial_id)
    frames = subject.readFrames(trial_id, 0, n_frames, includeSensorData=False, includeProcessingPasses=True)
    if not frames or not frames[0].processingPasses:
        raise RuntimeError('No processing passes found in frames. Cannot read poses/forces.')
    first = [fr.processingPasses[0] for fr in frames]

    # Pose DOFs (q)
    poses = [fr.pos for fr in first]  # list of arrays length J (>= 23)
    import numpy as np
    poses = np.asarray(poses, dtype=float)

    # Ground forces and COP from subject API
    forces = [fr.groundContactForce for fr in first]
    cops = [fr.groundContactCenterOfPressure for fr in first]
    forces = np.asarray(forces, dtype=float)
    cops = np.asarray(cops, dtype=float)

    # Restrict to calcn_r and calcn_l
    contact_bodies = subject.getGroundForceBodies()
    if 'calcn_r' not in contact_bodies or 'calcn_l' not in contact_bodies:
        raise RuntimeError('Foot contact bodies (calcn_r/l) not found in subject ground force bodies.')
    r_idx = contact_bodies.index('calcn_r')
    l_idx = contact_bodies.index('calcn_l')
    foot_idx = [r_idx * 3, r_idx * 3 + 1, r_idx * 3 + 2, l_idx * 3, l_idx * 3 + 1, l_idx * 3 + 2]

    forces = forces[:, foot_idx]
    cops = cops[:, foot_idx]

    # Compose state vector matching OSIM_DOF_ALL[:23] + KINETICS_ALL
    osim_cols = OSIM_DOF_ALL[:23] + KINETICS_ALL
    states = np.concatenate([poses[:, :23], forces[:, :3], cops[:, :3], forces[:, 3:], cops[:, 3:]], axis=1)
    return states, osim_cols


def main():
    ap = argparse.ArgumentParser(description='NimbleGUI로 AB walk 트라이얼을 실시간 시각화')
    ap.add_argument('--b3d', type=str, default=os.path.join(REPO_ROOT, 'data', 'Wang2023_Formatted_No_Arm', 'Wang2023_Formatted_No_Arm', 'Subj06', 'Subj06.b3d'))
    ap.add_argument('--trial_substr', type=str, default='walk')
    ap.add_argument('--frames', type=int, default=None, help='앞에서부터 표시할 프레임 수 (None이면 전체)')
    args = ap.parse_args()

    if not os.path.exists(args.b3d):
        print('[오류] .b3d 파일을 찾을 수 없습니다:', args.b3d)
        sys.exit(1)

    # Load subject and pick trial
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    trial_id, trial_name = find_walk_trial(subject, args.trial_substr)
    if trial_id is None:
        print(f"[오류] '{args.trial_substr}' 가 포함된 트라이얼을 찾지 못했습니다.")
        sys.exit(1)

    # Build states and opt indices for fig_utils.show_skeletons()
    states, osim_cols = read_states(subject, trial_id)
    if args.frames is not None:
        states = states[:max(1, args.frames)]

    opt = build_opt_indices(osim_cols)

    # Prepare GUI and skeleton
    gui = set_up_gui()
    # geometryFolder를 지정하지 않아도 동작(메시 불필요)
    skel = subject.readSkel(0)

    name_states = {'walk': states}
    print('[정보] NimbleGUI가 브라우저에서 열리면, 애니메이션이 재생됩니다.')
    print('       창이 바로 열리지 않으면, 콘솔에 표시된 포트(기본 8090+)로 접속하세요.')
    show_skeletons(opt, name_states, gui, skel)


if __name__ == '__main__':
    main()
