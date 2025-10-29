#!/usr/bin/env python3
"""
GaitDynamics Pre-trained Model Test Script
서버 환경에서 pre-trained 모델 테스트용
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
import copy

# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# GaitDynamics 모듈들 import  
from args import parse_opt
from consts import *
from model.model import MotionModel, BaselineModel

# TransformerEncoderArchitecture는 gait_dynamics.py에 정의됨
sys.path.append(os.path.join(project_root, 'example_usage'))
from gait_dynamics import TransformerEncoderArchitecture

def test_model_loading():
    """Pre-trained 모델들이 정상적으로 로드되는지 테스트"""
    print("=" * 50)
    print("GaitDynamics Pre-trained Model Test")
    print("=" * 50)
    
    # 설정 초기화
    opt = parse_opt()
    
    # Pre-trained 모델 경로 설정
    opt.checkpoint = os.path.join(project_root, 'example_usage', 'GaitDynamicsDiffusion.pt')
    opt.checkpoint_bl = os.path.join(project_root, 'example_usage', 'GaitDynamicsRefinement.pt')
    
    print(f"✅ Diffusion 모델 경로: {opt.checkpoint}")
    print(f"✅ Refinement 모델 경로: {opt.checkpoint_bl}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(opt.checkpoint):
        print(f"❌ Diffusion 모델 파일이 없습니다: {opt.checkpoint}")
        return False
        
    if not os.path.exists(opt.checkpoint_bl):
        print(f"❌ Refinement 모델 파일이 없습니다: {opt.checkpoint_bl}")
        return False
    
    try:
        print("\n🔄 Diffusion 모델 로딩 중...")
        diffusion_model = MotionModel(opt)
        print("✅ Diffusion 모델 로딩 성공!")
        
        print("\n🔄 Refinement 모델 로딩 중...")
        refinement_model = BaselineModel(opt, TransformerEncoderArchitecture)
        print("✅ Refinement 모델 로딩 성공!")
        
        print(f"\n📊 모델 정보:")
        print(f"   - 입력 차원: {diffusion_model.repr_dim}")
        print(f"   - 시퀀스 길이: {diffusion_model.horizon}")
        print(f"   - 디바이스: {diffusion_model.diffusion.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def create_sample_data():
    """테스트용 샘플 데이터 생성"""
    print("\n🔄 샘플 데이터 생성 중...")
    
    # 간단한 보행 패턴 생성 (150 프레임, 1.5초)
    n_frames = 150
    time = np.linspace(0, 1.49, n_frames)
    
    # 주기적 보행 패턴
    cycle_freq = 2 * np.pi  # 1초 주기
    
    data = {
        'time': time,
        'pelvis_tilt': 2 * np.sin(time * cycle_freq),
        'pelvis_list': 0.5 * np.sin(time * cycle_freq * 2),
        'pelvis_rotation': 1 * np.sin(time * cycle_freq * 0.5),
        'pelvis_tx': time * 1.2,  # 전진
        'pelvis_ty': 0.8 + 0.02 * np.sin(time * cycle_freq * 2),  # 수직 움직임
        'pelvis_tz': 0.01 * time,
        'hip_flexion_r': 20 + 15 * np.sin(time * cycle_freq),
        'hip_adduction_r': -5 + 3 * np.sin(time * cycle_freq * 2),
        'hip_rotation_r': 2 * np.sin(time * cycle_freq),
        'knee_angle_r': 5 + 15 * np.maximum(0, np.sin(time * cycle_freq)),
        'ankle_angle_r': 5 * np.sin(time * cycle_freq - np.pi/4),
        'subtalar_angle_r': np.zeros(n_frames),
        'hip_flexion_l': 20 + 15 * np.sin(time * cycle_freq + np.pi),  # 반대 위상
        'hip_adduction_l': 5 + 3 * np.sin(time * cycle_freq * 2),
        'hip_rotation_l': -2 * np.sin(time * cycle_freq),
        'knee_angle_l': 5 + 15 * np.maximum(0, np.sin(time * cycle_freq + np.pi)),
        'ankle_angle_l': 5 * np.sin(time * cycle_freq - np.pi/4 + np.pi),
        'subtalar_angle_l': np.zeros(n_frames),
        'lumbar_extension': 2 * np.sin(time * cycle_freq * 0.5),
        'lumbar_bending': np.zeros(n_frames),
        'lumbar_rotation': np.zeros(n_frames),
    }
    
    return pd.DataFrame(data)

def save_sample_mot_file(df, filename):
    """샘플 데이터를 .mot 파일로 저장"""
    n_frames, n_cols = df.shape
    
    with open(filename, 'w') as f:
        # 헤더 작성
        f.write('Coordinates\n')
        f.write('version=1\n')
        f.write(f'nRows={n_frames}\n')
        f.write(f'nColumns={n_cols}\n')
        f.write('inDegrees=yes\n\n')
        f.write('If the header above contains a line with \'inDegrees\', this indicates whether rotational values are in degrees (yes) or radians (no).\n\n')
        f.write('endheader\n')
        
        # 컬럼명 작성
        f.write('\t'.join(df.columns) + '\n')
        
        # 데이터 작성
        for _, row in df.iterrows():
            f.write('\t'.join([f'{val:.5f}' for val in row]) + '\n')
    
    print(f"✅ 샘플 .mot 파일 저장: {filename}")

def main():
    print("🚀 GaitDynamics Pre-trained Model 테스트 시작\n")
    
    # CUDA 환경 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
    
    # 모델 로딩 테스트
    success = test_model_loading()
    
    if success:
        print("\n🎉 모든 테스트 통과!")
        print("\n📝 다음 단계:")
        print("1. .mot 파일과 .osim 파일 준비")
        print("2. 실제 데이터로 추론 실행")
        
        # 샘플 데이터 생성
        df = create_sample_data()
        save_sample_mot_file(df, 'sample_gait_data.mot')
        
    else:
        print("\n❌ 테스트 실패. 환경 설정을 확인하세요.")

if __name__ == "__main__":
    main()