#!/usr/bin/env python3
"""Quick script to inspect S004.h5 structure and find MoCap/ik_data datasets."""
"""
HDF5 파일(S004.h5) 내부 구조를 탐색하여 실제 데이터 경로와 필드명을 찾기 위한 일회성 디버깅 도구
===========
MoCap, ik_data, grf 관련 경로만 출력, 불필요한 IMU/센서 데이터는 필터링, 계층 구조 출력

GROUP: 하위 child 목록 (최대 30개)
DATASET: shape, dtype 정보
"""

import h5py
from pathlib import Path

h5_path = Path(r'C:\workspace\GaitDynamics-1\data\LD\S004.h5')

with h5py.File(h5_path, 'r') as f:
    print('=== S004.h5 Structure Inspection ===\n')
    
    def visit_func(name, obj):
        if 'MoCap' in name or 'ik_data' in name or 'grf' in name.lower():
            if isinstance(obj, h5py.Group):
                print(f'GROUP: {name}')
                print(f'  Children: {list(obj.keys())[:30]}')
            elif isinstance(obj, h5py.Dataset):
                print(f'DATASET: {name}')
                print(f'  Shape: {obj.shape}, dtype: {obj.dtype}')
    
    f.visititems(visit_func)
    
    # Try to directly access expected paths
    print('\n=== Attempting direct access ===')
    try_paths = [
        'S004/level_08mps/trial_01/MoCap/ik_data',
        'S004/trial_01/level_08mps/MoCap/ik_data',
        'S004/level_08mps/MoCap/ik_data',
    ]
    
    for p in try_paths:
        try:
            obj = f[p]
            print(f'\nFound: {p}')
            if isinstance(obj, h5py.Group):
                print(f'  Datasets: {list(obj.keys())}')
            else:
                print(f'  Shape: {obj.shape}, dtype: {obj.dtype}')
        except KeyError:
            pass
