#!/usr/bin/env python3
"""
GaitDynamics 결과를 CSV로 변환하고 요약 통계 출력
"""

import pandas as pd
import numpy as np

# GRF 데이터를 pandas로 읽기
def convert_grf_to_csv():
    # GRF 파일의 헤더 건너뛰고 읽기
    with open('gait_sample_2sec_grf_pred___.mot', 'r') as f:
        lines = f.readlines()
    
    # 'time' 라인 찾기
    for i, line in enumerate(lines):
        if line.startswith('time'):
            header_line = i
            break
    
    # 데이터 파싱
    header = lines[header_line].strip().split('\t')
    data = []
    for line in lines[header_line+1:]:
        if line.strip():
            values = [float(x) for x in line.strip().split('\t')]
            data.append(values)
    
    grf_df = pd.DataFrame(data, columns=header)
    grf_df.to_csv('grf_results.csv', index=False)
    
    print("📊 Ground Reaction Forces 요약:")
    print(f"시간 범위: {grf_df['time'].min():.2f} - {grf_df['time'].max():.2f}초")
    print(f"데이터 포인트: {len(grf_df)}개")
    print("\n주요 힘 성분 (Newton):")
    print(f"- 오른발 수직력 최대: {grf_df['force1_vy'].max():.1f}N")
    print(f"- 왼발 수직력 최대: {grf_df['force2_vy'].max():.1f}N")
    print(f"- 총 수직력 최대: {(grf_df['force1_vy'] + grf_df['force2_vy']).max():.1f}N")
    
    return grf_df

# Kinematics 데이터 처리
def convert_kinematics_to_csv():
    # 예측된 kinematics 읽기 
    kin_df = pd.read_csv('gait_sample_2sec_missing_kinematics_pred___.mot', 
                         sep='\t', skiprows=10)
    kin_df.to_csv('kinematics_results.csv', index=False)
    
    print("\n🚶 Kinematics 요약:")
    print(f"컬럼 수: {len(kin_df.columns)}")
    print(f"시간 범위: {kin_df['time'].min():.2f} - {kin_df['time'].max():.2f}초")
    
    # 각 관절의 움직임 범위
    key_joints = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r',
                  'hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
    
    print("\n관절 움직임 범위 (라디안):")
    for joint in key_joints:
        if joint in kin_df.columns:
            range_val = kin_df[joint].max() - kin_df[joint].min()
            print(f"- {joint}: {range_val:.3f} rad ({np.degrees(range_val):.1f}°)")
    
    return kin_df

if __name__ == "__main__":
    print("📁 GaitDynamics 결과 → CSV 변환")
    print("=" * 50)
    
    grf_df = convert_grf_to_csv()
    kin_df = convert_kinematics_to_csv()
    
    print(f"\n✅ 생성된 파일:")
    print(f"- grf_results.csv ({len(grf_df)} 행)")
    print(f"- kinematics_results.csv ({len(kin_df)} 행)")
    
    print(f"\n💡 다음 단계 제안:")
    print(f"- MATLAB/Python으로 데이터 시각화")
    print(f"- OpenSim에서 시뮬레이션 실행")
    print(f"- 다른 보행 데이터와 비교 분석")