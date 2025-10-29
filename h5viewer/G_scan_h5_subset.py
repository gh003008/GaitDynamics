#!/usr/bin/env python3
import os
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
H5_PATH = os.path.join(ROOT, 'combined_data.h5')
OUT = os.path.join(ROOT, 'h5viewer', 'G_combined_data_S004plus_summary.txt')

KEYWORDS = ['mocap', 'marker', 'force', 'forceplate', 'grf', 'cop', 'opensim']


def sub_id_num(s):
    try:
        return int(''.join([c for c in s if c.isdigit()]))
    except Exception:
        return -1


def main():
    with h5py.File(H5_PATH, 'r') as f, open(OUT, 'w') as w:
        subs = [k for k in f.keys() if k.startswith('S') and sub_id_num(k) >= 4]
        subs.sort(key=sub_id_num)
        w.write(f"Scanning {H5_PATH} for subjects >= S004\n")
        w.write(f"Found subjects: {subs}\n\n")
        for s in subs:
            w.write(f"== Subject {s} ==\n")
            subj_grp = f[s]
            for level in subj_grp.keys():
                level_path = f"/{s}/{level}"
                if not isinstance(f[level_path], h5py.Group):
                    continue
                w.write(f"  - {level_path}\n")
                level_grp = f[level_path]
                for trial in level_grp.keys():
                    trial_path = f"{level_path}/{trial}"
                    if not isinstance(f[trial_path], h5py.Group):
                        continue
                    w.write(f"    * {trial_path}\n")
                    g = f[trial_path]
                    # list immediate subgroups
                    w.write("      subgroups: ")
                    subgroups = [k for k in g.keys() if isinstance(g[k], h5py.Group)]
                    w.write(str(subgroups) + "\n")
                    # find keyword datasets/subgroups
                    hits = []
                    def visit(name, obj):
                        low = name.lower()
                        if any(k in low for k in KEYWORDS):
                            hits.append(name)
                    g.visititems(visit)
                    if hits:
                        w.write("      keyword hits:\n")
                        for h in hits:
                            full = h if h.startswith('/') else (trial_path + '/' + h)
                            obj = f[full]
                            if isinstance(obj, h5py.Dataset):
                                w.write(f"        - {h} (dataset) shape={obj.shape} dtype={obj.dtype}\n")
                            else:
                                w.write(f"        - {h} (group)\n")
            w.write("\n")
    print(f"Wrote summary to {OUT}")


if __name__ == '__main__':
    main()
