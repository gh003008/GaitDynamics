#!/usr/bin/env python3
import argparse
import os
import sys

try:
    import nimblephysics as nimble
except Exception as e:
    print("[ERROR] Failed to import nimblephysics: ", e)
    sys.exit(1)


def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}PB"


def main():
    parser = argparse.ArgumentParser(description="Inspect a .b3d biomechanics file (SubjectOnDisk)")
    parser.add_argument('--b3d', required=True, help='Path to .b3d file')
    parser.add_argument('--trial', type=int, default=None, help='Optional trial index to inspect frames for')
    parser.add_argument('--frames', type=int, default=3, help='How many frames to sample for detailed info')
    args = parser.parse_args()

    path = os.path.abspath(args.b3d)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    print(f"[LOAD] {path}")
    try:
        subject = nimble.biomechanics.SubjectOnDisk(path)
    except Exception as e:
        print(f"[ERROR] Could not open .b3d: {e}")
        sys.exit(1)

    # Basic metadata
    try:
        num_trials = subject.getNumTrials()
    except Exception:
        # older API name
        num_trials = subject.getNumTrials() if hasattr(subject, 'getNumTrials') else None

    print("\n== Subject Metadata ==")
    print(f"Trials: {num_trials}")
    try:
        timestep = subject.getTrialTimestep(0)
        print(f"Timestep (trial 0): {timestep:.6f} s  (~{1.0/timestep:.2f} Hz)")
    except Exception as e:
        print(f"Timestep: <unknown> ({e})")

    try:
        if hasattr(subject, 'getNumProcessingPasses'):
            print(f"Processing passes: {subject.getNumProcessingPasses()}")
    except Exception:
        pass

    # Trial list
    print("\n== Trials ==")
    trial_names = []
    for i in range(num_trials):
        name = None
        try:
            name = subject.getTrialName(i)
        except Exception:
            name = f"trial_{i:02d}"
        trial_names.append(name)
        try:
            length = subject.getTrialLength(i)
        except Exception as e:
            length = None
        try:
            ts = subject.getTrialTimestep(i)
        except Exception:
            ts = None
        f = f"{name} (idx {i})"
        if length is not None:
            f += f", frames={length}"
        if ts is not None:
            f += f", dt={ts:.4f}s (~{1.0/ts:.1f}Hz)"
        print(" - ", f)

    # Skeleton overview
    print("\n== Skeleton (trial 0) ==")
    try:
        skel = subject.readSkel(0)
        dofs = []
        try:
            # Preferred: get all DOFs list directly if available
            dofs = [d.getName() for d in skel.getDofs()]
        except Exception:
            # Fallback: iterate by index if API supports it
            try:
                dofs = [skel.getDof(i).getName() for i in range(skel.getNumDofs())]
            except Exception as e:
                print(f"  Warning: could not enumerate DOFs ({e})")
                dofs = []
        print(f"Num DOFs: {len(dofs)}")
        if dofs:
            print("First 30 DOFs:")
            for n in dofs[:30]:
                print("   ", n)
            if len(dofs) > 30:
                print("   ...")
    except Exception as e:
        print(f"Could not read skeleton: {e}")

    # Sample frames
    idx = args.trial if args.trial is not None else 0
    print(f"\n== Sample Frames (trial {idx}) ==")
    try:
        length = subject.getTrialLength(idx)
        step = max(1, length // max(1, args.frames))
        num_to_read = max(1, min(length // step, args.frames))
        # Use stride-based sampling to accommodate API that takes start/num/stride
        print(f"Sampling {num_to_read} frames with stride={step} across {length} frames")
        frames = subject.readFrames(idx, 0, num_to_read, True, True, step)
        # Each frame has .pos, .vel, .acc, .tau, .grf, maybe .markers
        # Limit to at most args.frames frames for printing
        to_show = min(len(frames), args.frames)
        import numpy as np

        def get_field(fr, candidates):
            for name in candidates:
                try:
                    val = getattr(fr, name, None)
                except Exception:
                    val = None
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        val = None
                if val is not None:
                    return val
            return None

        def to_shape(x):
            if x is None:
                return None
            try:
                arr = np.asarray(x)
                return tuple(arr.shape)
            except Exception:
                try:
                    return (len(x),)
                except Exception:
                    return None

        for k in range(to_show):
            j = k * step
            fr = frames[k]
            pos = get_field(fr, ['pos', 'positions', 'q', 'getPos', 'getPositions'])
            vel = get_field(fr, ['vel', 'velocity', 'dq', 'getVel', 'getVelocity', 'getDq'])
            grf = getattr(fr, 'grf', None)
            markers = getattr(fr, 'markers', None)
            print(f"  Frame {j}:")
            print(f"    pos shape: {to_shape(pos)}")
            print(f"    vel shape: {to_shape(vel)}")
            if to_shape(pos) is None:
                try:
                    attrs = [a for a in dir(fr) if not a.startswith('_')]
                    print("    frame attrs:", ', '.join(attrs[:20]), ("..." if len(attrs)>20 else ""))
                except Exception:
                    pass
            if grf is not None:
                print(f"    grf type: {type(grf)}")
            if markers is not None:
                try:
                    print(f"    markers: {len(markers)}")
                except Exception:
                    print(f"    markers: <unknown>")
    except Exception as e:
        print(f"Failed to read frames: {e}")

    # File size
    try:
        sz = os.path.getsize(path)
        print(f"\n== File ==\nPath: {path}\nSize: {human_bytes(sz)}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
