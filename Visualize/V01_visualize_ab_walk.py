#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Repo root on sys.path so we can import data.osim_fk
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import nimblephysics as nimble  # noqa: E402
except Exception as e:
    print('\n[오류] nimblephysics 모듈을 찾을 수 없습니다.\n' \
          ' - 이 스크립트는 AddBiomechanics의 .b3d 파일을 읽기 위해 nimblephysics가 필요합니다.\n' \
          ' - conda 또는 별도의 환경에서 Python 3.10/3.11로 nimblephysics를 설치해 주세요.\n' \
          ' - 설치 가이드: https://github.com/keenon/nimblephysics (또는 배포된 wheel/conda 채널 이용)\n' \
          f' - 내부 예외: {e}\n')
    import sys
    sys.exit(2)

from data.osim_fk import get_model_offsets, forward_kinematics  # noqa: E402


def find_walk_trial(subject, substring='walk'):
    """Return (trial_id, name) of the first trial containing substring (case-insensitive)."""
    sub = substring.lower()
    for t in range(subject.getNumTrials()):
        name = subject.getTrialName(t)
        if sub in name.lower():
            return t, name
    return None, None


def read_trial_states(subject, trial_id):
    """Read poses (q) for a trial from the first processing pass.
    Returns: poses [T, D], fps (int)
    """
    timestep = subject.getTrialTimestep(trial_id)
    fps = int(round(1.0 / timestep)) if timestep > 0 else 100
    n_frames = subject.getTrialLength(trial_id)
    frames = subject.readFrames(trial_id, 0, n_frames, includeSensorData=False, includeProcessingPasses=True)
    if not frames or not frames[0].processingPasses:
        raise RuntimeError('No processing passes found in frames. Cannot read poses.')
    pass0 = [fr.processingPasses[0] for fr in frames]
    poses = np.array([fr.pos for fr in pass0], dtype=np.float32)
    return poses, fps


def build_lines(ax, joint_names):
    """Define simple lower-body connectivity for plotting lines."""
    # Expected joint_names (no arms):
    # ['pelvis','hip_r','knee_r','ankle_r','calcn_r','mtp_r','hip_l','knee_l','ankle_l','calcn_l','mtp_l']
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    edges = []
    def e(a, b):
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))
    # Right leg chain
    e('pelvis','hip_r'); e('hip_r','knee_r'); e('knee_r','ankle_r'); e('ankle_r','calcn_r'); e('calcn_r','mtp_r')
    # Left leg chain
    e('pelvis','hip_l'); e('hip_l','knee_l'); e('knee_l','ankle_l'); e('ankle_l','calcn_l'); e('calcn_l','mtp_l')
    # Create Line3D objects
    lines = []
    for _ in edges:
        line, = ax.plot([0, 0], [0, 0], [0, 0], lw=2)
        lines.append(line)
    return edges, lines


def animate_skeleton(joints_xyz, joint_names, out_path, fps=60):
    """Animate joints over time and save as MP4 (fallback GIF). joints_xyz: [T, J, 3]."""
    T, J, _ = joints_xyz.shape

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('AB Walk (lower body)')

    # Set nice equal aspect and padding
    all_xyz = joints_xyz.reshape(-1, 3)
    xmin, ymin, zmin = np.min(all_xyz, axis=0)
    xmax, ymax, zmax = np.max(all_xyz, axis=0)
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    span = max(span, 0.5)
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_zlim(cz - span / 2, cz + span / 2)
    ax.view_init(elev=15, azim=-80)

    edges, lines = build_lines(ax, joint_names)

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        P = joints_xyz[frame]  # [J, 3]
        for line, (i, j) in zip(lines, edges):
            xs = [P[i, 0], P[j, 0]]
            ys = [P[i, 1], P[j, 1]]
            zs = [P[i, 2], P[j, 2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
        return lines

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=1000.0 / fps, blit=True)

    # Try MP4 first, fallback to GIF
    base, _ = os.path.splitext(out_path)
    mp4_path = base + '.mp4'
    gif_path = base + '.gif'
    png_path = base + '.png'

    saved_path = None
    try:
        Writer = animation.writers['ffmpeg']  # type: ignore[attr-defined]
        writer = Writer(fps=fps, metadata=dict(artist='GaitDynamics'), bitrate=1800)
        anim.save(mp4_path, writer=writer)
        saved_path = mp4_path
    except Exception:
        try:
            anim.save(gif_path, writer='pillow', fps=fps)
            saved_path = gif_path
        except Exception:
            # Save a static frame as last resort
            fig.savefig(png_path, dpi=200)
            saved_path = png_path

    plt.close(fig)
    return saved_path


def main():
    ap = argparse.ArgumentParser(description='Visualize AB walk trial as a simple lower-body skeleton animation (no arms).')
    ap.add_argument('--b3d', type=str, default=os.path.join(REPO_ROOT, 'data', 'Wang2023_Formatted_No_Arm', 'Wang2023_Formatted_No_Arm', 'Subj06', 'Subj06.b3d'), help='Path to a SubjectOnDisk .b3d file')
    ap.add_argument('--trial_substr', type=str, default='walk', help='Substring to match a walk trial name')
    ap.add_argument('--out_dir', type=str, default=os.path.join(REPO_ROOT, 'previews'), help='Output directory for video/image')
    ap.add_argument('--start', type=int, default=0, help='Start frame index (inclusive)')
    ap.add_argument('--frames', type=int, default=250, help='Number of frames to render (<= trial length)')
    ap.add_argument('--fps', type=int, default=60, help='Output video FPS')
    args = ap.parse_args()

    if not os.path.exists(args.b3d):
        raise FileNotFoundError(f".b3d not found: {args.b3d}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load subject and pick a walk trial
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    trial_id, trial_name = find_walk_trial(subject, args.trial_substr)
    if trial_id is None:
        raise RuntimeError(f"No trial containing '{args.trial_substr}' found in {args.b3d}")

    # Read poses and skeleton
    poses, fs = read_trial_states(subject, trial_id)  # poses [T, D]
    T_total, D = poses.shape

    # Restrict to a window
    start = max(0, min(args.start, T_total - 1))
    end = min(T_total, start + args.frames)
    poses = poses[start:end]

    # Take the first 23 OSIM DOFs (pelvis + lower body + lumbar)
    if D < 23:
        raise RuntimeError(f"Expected at least 23 DOFs, got {D}")
    poses23 = poses[:, :23]

    # Skeleton and model offsets (no arms)
    # geometryFolder is optional for FK; we skip to avoid requiring meshes
    skel = subject.readSkel(0)
    offsets = get_model_offsets(skel, with_arm=False)

    # Forward kinematics to get joint points
    foot_xyz, joint_xyz, joint_names, _ = forward_kinematics(poses23, offsets, with_arm=False)
    # joint_xyz shape is [J, B, T, 3] (B=1). Convert to [T, J, 3] on CPU numpy.
    if hasattr(joint_xyz, 'detach'):
        j = joint_xyz.detach().cpu()
    else:
        j = np.asarray(joint_xyz)
    # Torch path
    try:
        if hasattr(j, 'numpy'):
            j_np = j.numpy()
        else:
            j_np = np.asarray(j)
    except Exception:
        j_np = np.asarray(j)
    # Squeeze batch dim if present: [J, 1, T, 3] -> [J, T, 3]
    while j_np.ndim == 4 and j_np.shape[1] == 1:
        j_np = j_np[:, 0]
    # Now expect [J, T, 3]; transpose to [T, J, 3]
    if j_np.ndim == 3 and j_np.shape[-1] == 3 and j_np.shape[0] < j_np.shape[1]:
        joint_xyz_np = j_np.transpose(1, 0, 2)
    elif j_np.ndim == 3 and j_np.shape[-1] == 3 and j_np.shape[0] > j_np.shape[1]:
        # Already [T, J, 3]
        joint_xyz_np = j_np
    else:
        raise RuntimeError(f"예상치 못한 joint_xyz shape: {j_np.shape}")

    # Save animation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subj = os.path.splitext(os.path.basename(args.b3d))[0]
    base = f"ab_walk_{subj}_{trial_name}_{timestamp}"
    base = base.replace('/', '_')
    out_base = os.path.join(args.out_dir, base)
    saved_path = animate_skeleton(joint_xyz_np, joint_names, out_base, fps=args.fps)

    print('[ok] Visualized:', trial_name)
    print('  frames:', joint_xyz_np.shape[0], '(subset of', T_total, ')')
    print('  output:', saved_path)


if __name__ == '__main__':
    main()
