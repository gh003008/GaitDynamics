#!/usr/bin/env python3
"""
GaitDynamics 결과 분석 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_grf_results():
    """Ground Reaction Forces 분석"""
    print("=" * 60)
    print("🦶 Ground Reaction Forces (GRF) 분석")
    print("=" * 60)
    
    # GRF 데이터 로드 (헤더 처리)
    with open('gait_sample_2sec_grf_pred___.mot', 'r') as f:
        lines = f.readlines()
    
    # 헤더 건너뛰고 데이터 시작점 찾기
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('time'):
            start_idx = i
            break
    
    # 데이터 파싱
    data_lines = lines[start_idx:]
    header = data_lines[0].strip().split('\t')
    
    grf_data = []
    for line in data_lines[1:]:
        values = [float(x) for x in line.strip().split('\t')]
        grf_data.append(values)
    
    grf_df = pd.DataFrame(grf_data, columns=header)
    
    print(f"📊 데이터 개요:")
    print(f"   - 시간 범위: {grf_df['time'].min():.2f}s ~ {grf_df['time'].max():.2f}s")
    print(f"   - 데이터 포인트: {len(grf_df)}개")
    print(f"   - 샘플링 레이트: {1/(grf_df['time'][1]-grf_df['time'][0]):.0f} Hz")
    
    print(f"\n🔍 Force 데이터 (Newton):")
    print(f"   - 오른발 수직력 (force1_vy): {grf_df['force1_vy'].min():.1f} ~ {grf_df['force1_vy'].max():.1f} N")
    print(f"   - 왼발 수직력 (force2_vy): {grf_df['force2_vy'].min():.1f} ~ {grf_df['force2_vy'].max():.1f} N")
    print(f"   - 오른발 전후력 (force1_vx): {grf_df['force1_vx'].min():.1f} ~ {grf_df['force1_vx'].max():.1f} N")
    print(f"   - 왼발 전후력 (force2_vx): {grf_df['force2_vx'].min():.1f} ~ {grf_df['force2_vx'].max():.1f} N")
    
    # 체중 추정 (수직력 최대값으로부터)
    max_vertical_force = max(grf_df['force1_vy'].max(), grf_df['force2_vy'].max())
    estimated_weight = max_vertical_force / 9.81  # N to kg
    print(f"\n⚖️  추정 체중: {estimated_weight:.1f} kg (최대 수직력 기준)")
    
    return grf_df

def analyze_kinematics_results():
    """Missing Kinematics 분석"""
    print("\n" + "=" * 60)
    print("🚶 Missing Kinematics 분석")
    print("=" * 60)
    
    # 원본 데이터 로드
    original_df = pd.read_csv('../gait_sample_2sec.mot', sep='\t', skiprows=10)
    
    # 예측 데이터 로드
    predicted_df = pd.read_csv('gait_sample_2sec_missing_kinematics_pred___.mot', sep='\t', skiprows=10)
    
    print(f"📊 컬럼 비교:")
    print(f"   - 원본: {len(original_df.columns)}개 컬럼")
    print(f"   - 예측: {len(predicted_df.columns)}개 컬럼")
    
    # 새로 추가된 컬럼 찾기
    new_columns = set(predicted_df.columns) - set(original_df.columns)
    print(f"\n✨ 새로 생성된 컬럼들:")
    for col in sorted(new_columns):
        if col != 'time':
            print(f"   - {col}")
    
    # MTP 각도 분석 (새로 생성된 중요한 데이터)
    if 'mtp_angle_r' in predicted_df.columns:
        print(f"\n👣 MTP (발가락) 관절 각도:")
        print(f"   - 오른발 MTP: {predicted_df['mtp_angle_r'].min():.3f} ~ {predicted_df['mtp_angle_r'].max():.3f} rad")
        print(f"   - 왼발 MTP: {predicted_df['mtp_angle_l'].min():.3f} ~ {predicted_df['mtp_angle_l'].max():.3f} rad")
    
    return original_df, predicted_df

def main():
    print("🔬 GaitDynamics 결과 분석 리포트")
    print("생성 시간:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # GRF 분석
    grf_df = analyze_grf_results()
    
    # Kinematics 분석  
    original_df, predicted_df = analyze_kinematics_results()
    
    print("\n" + "=" * 60)
    print("🎯 활용 방안:")
    print("=" * 60)
    print("1. 📈 Biomechanical Analysis:")
    print("   - 보행 패턴 분석 및 비대칭성 검출")
    print("   - 관절 각도 변화 추적")
    print("   - 지면반발력을 통한 보행 안정성 평가")
    
    print("\n2. 🏥 Clinical Applications:")
    print("   - 재활 치료 효과 모니터링")
    print("   - 보행 장애 진단 보조")
    print("   - 의족/보조기 설계 데이터")
    
    print("\n3. 🤖 Robotics & Simulation:")
    print("   - 휴머노이드 로봇 보행 제어")
    print("   - OpenSim 시뮬레이션 입력 데이터")
    print("   - 가상 인간 모델링")
    
    print("\n4. 🔬 Research:")
    print("   - 보행 데이터셋 확장")
    print("   - 머신러닝 모델 훈련 데이터")
    print("   - 바이오메카닉스 연구")

if __name__ == "__main__":
    main()