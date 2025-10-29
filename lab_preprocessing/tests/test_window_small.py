import os
import sys
import tempfile
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lab_preprocessing.G_convert_h5_to_gaitdyn import convert, DEFAULT_CONFIG
from lab_preprocessing.G_window_sensors import run as window_run

def make_dummy_h5(path):
    T = 500
    with h5py.File(path, 'w') as f:
        g = f.require_group('S001/level_12mps/trial_01')
        g.create_dataset('Common/time', data=np.arange(T, dtype=np.int32))
        bi = g.require_group('Back_imu')
        bi.create_dataset('Accel', data=np.random.randn(T,3).astype(np.float32))
        bi.create_dataset('Gyro', data=np.random.randn(T,3).astype(np.float32))
        ti = g.require_group('Thigh_imu')
        ti.create_dataset('Accel', data=np.random.randn(T,3).astype(np.float32))
        ti.create_dataset('Gyro', data=np.random.randn(T,3).astype(np.float32))
        il = g.require_group('Insole/Left')
        ir = g.require_group('Insole/Right')
        for i in range(1,9):
            il.create_dataset(f'fsrL{i}', data=np.abs(np.random.randn(T).astype(np.float32)))
            ir.create_dataset(f'fsrR{i}', data=np.abs(np.random.randn(T).astype(np.float32)))
        td = g.require_group('Treadmill_data')
        td.create_dataset('belt_speed_left', data=np.ones(T, dtype=np.float32))
        td.create_dataset('belt_speed_right', data=np.ones(T, dtype=np.float32))


def test_end_to_end_window():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'dummy.h5')
        out = os.path.join(d, 'out.h5')
        win_dir = os.path.join(d, 'wins')
        make_dummy_h5(src)
        convert(src, out, DEFAULT_CONFIG, subjects=['S001'])
        window_run(out, win_dir, wlen=150, stride=50)
        assert os.path.exists(os.path.join(win_dir, 'index.json'))
        files = [f for f in os.listdir(win_dir) if f.endswith('.npz')]
        assert len(files) >= 1


if __name__ == '__main__':
    test_end_to_end_window()
    print('OK')
