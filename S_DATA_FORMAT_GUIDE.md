# 📊 GaitDynamics 데이터 형식 및 저장 위치 완전 가이드

## 🗂️ **1. 생성된 데이터 저장 위치**

### 📍 **저장 위치**: 실행한 디렉토리 (현재 작업 디렉토리)
```bash
/home/ghlee/GaitDynamics/example_usage/
```

### 📁 **생성되는 파일들**
```bash
# 1. Ground Reaction Forces (지면반발력)
{원본파일명}_grf_pred___.mot

# 2. Missing Kinematics (누락된 관절각도)  
{원본파일명}_missing_kinematics_pred___.mot

# 예시:
gait_sample_2sec_grf_pred___.mot
gait_sample_2sec_missing_kinematics_pred___.mot
```

---

## 📋 **2. 출력 파일 형식 상세 분석**

### 🦶 **A. GRF 파일 (.mot 형식)**
```
파일명: {original}_grf_pred___.mot
크기: ~27KB (200 프레임 기준)
형식: OpenSim Motion File

헤더 구조:
nColumns=9
nRows=200
DataType=double  
version=3
OpenSimVersion=4.1
endheader

데이터 컬럼:
time | force1_vx | force1_vy | force1_vz | force1_px | force1_py | force1_pz | 
     | torque1_x | torque1_y | torque1_z | force2_vx | force2_vy | force2_vz |
     | force2_px | force2_py | force2_pz | torque2_x | torque2_y | torque2_z

설명:
- force1_*: 오른발 지면반발력 (N)
- force2_*: 왼발 지면반발력 (N)  
- vx,vy,vz: X,Y,Z 방향 힘 성분
- px,py,pz: 압력중심점(COP) 위치
- torque*: 모멘트 성분 (현재 0으로 설정)
```

### 🚶 **B. Kinematics 파일 (.mot 형식)**
```
파일명: {original}_missing_kinematics_pred___.mot
크기: ~46KB (200 프레임 기준)
형식: OpenSim Coordinates File

헤더 구조:
Coordinates
version=1
nRows=200
nColumns=24
inDegrees=no

데이터 컬럼 (24개):
time | pelvis_tilt | pelvis_list | pelvis_rotation | pelvis_tx | pelvis_ty | pelvis_tz |
     | hip_flexion_r | hip_adduction_r | hip_rotation_r | knee_angle_r | ankle_angle_r |
     | subtalar_angle_r | mtp_angle_r | hip_flexion_l | hip_adduction_l | hip_rotation_l |
     | knee_angle_l | ankle_angle_l | subtalar_angle_l | mtp_angle_l | lumbar_extension |
     | lumbar_bending | lumbar_rotation

새로 생성된 컬럼:
- mtp_angle_r: 오른발 발가락 관절각 (라디안)
- mtp_angle_l: 왼발 발가락 관절각 (라디안)
```

---

## 📚 **3. AddBiomechanics 데이터셋 구조**

### 🗄️ **입력 데이터셋 형식**
```
원본 데이터셋: AddBiomechanics Dataset
다운로드: https://addbiomechanics.org/download_data.html

데이터 구조:
/dataset_backups/
├── train_cleaned/
│   ├── Camargo2021_Formatted_No_Arm/
│   ├── Carter2023_Formatted_No_Arm/
│   ├── Fregly2012_Formatted_No_Arm/
│   ├── Falisse2017_Formatted_No_Arm/
│   ├── Hamner2013_Formatted_No_Arm/
│   ├── Han2023_Formatted_No_Arm/
│   ├── Li2021_Formatted_No_Arm/
│   ├── Moore2015_Formatted_No_Arm/
│   ├── Santos2017_Formatted_No_Arm/
│   ├── Tan2021_Formatted_No_Arm/
│   ├── Tan2022_Formatted_No_Arm/
│   ├── Tiziana2019_Formatted_No_Arm/
│   ├── Uhlrich2023_Formatted_No_Arm/
│   ├── vanderZee2022_Formatted_No_Arm/
│   └── Wang2023_Formatted_No_Arm/
└── test_cleaned/
    └── (동일 구조)
```

### 📊 **데이터셋별 특성**
```python
# 15개 연구 데이터셋
DSET_SHORT_NAMES = [
    'Camargo2021',    # 보행 재활
    'Carter2023',     # 달리기 (고속)
    'Fregly2012',     # 무릎 관절
    'Falisse2017',    # 외골격
    'Hamner2013',     # 달리기 바이오메카닉스
    'Han2023',        # 한국인 보행
    'Li2021',         # 아시아인 보행
    'Moore2015',      # 다양한 연령대
    'Santos2017',     # 브라질 보행
    'Tan2021',        # 러닝 속도
    'Tan2022',        # 러닝 패턴
    'Tiziana2019',    # 이탈리아 보행
    'Uhlrich2023',    # 스탠포드 보행
    'vanderZee2022',  # 네덜란드 보행
    'Wang2023'        # 중국 보행
]

# 러닝 전문 데이터셋
RUNNING_DSET_SHORT_NAMES = ['Carter2023', 'Hamner2013', 'Tan2021', 'Wang2023']

# Overground 보행 (트레드밀 아님)
OVERGROUND_DSETS = ['Fregly', 'Falisse', 'Han', 'Li', 'Santos', 'Uhlrich', 'Tiziana']
```

---

## 🔧 **4. 내부 데이터 처리 과정**

### 📥 **입력 → 처리 → 출력 파이프라인**
```python
# 1. 입력 데이터 (.mot)
원본_mot_파일 = {
    "형식": "OpenSim Motion File",
    "컬럼": 22개 (pelvis~lumbar, 양발 관절각도),
    "단위": "degrees (inDegrees=yes)",
    "누락": ['mtp_angle_r', 'mtp_angle_l'] # 발가락 관절
}

# 2. 내부 처리
전처리_단계 = {
    "1단계": "degrees → radians 변환",
    "2단계": "Nimble Physics로 Forward Kinematics",  
    "3단계": "데이터 정규화 (Normalizer)",
    "4단계": "1.5초 윈도우로 분할 (150 프레임)",
    "5단계": "Missing data masking"
}

diffusion_처리 = {
    "모델": "DanceDecoder + GaussianDiffusion",
    "입력": "Masked kinematics window", 
    "출력": "Complete kinematics window",
    "조건": "체형(키,몸무게), 속도 정보"
}

refinement_처리 = {
    "모델": "BaselineModel + TransformerEncoder",
    "입력": "Complete kinematics",
    "출력": "Ground Reaction Forces", 
    "후처리": "역정규화, 단위 변환 (N)"
}

# 3. 출력 데이터 (.mot)
출력_grf = {
    "형식": "OpenSim External Forces File",
    "컬럼": 19개 (시간 + 양발 6DOF force/moment)",
    "단위": "Newton, Meter", 
    "특징": "100Hz, 지면반발력 + COP"
}

출력_kinematics = {
    "형식": "OpenSim Coordinates File", 
    "컬럼": 24개 (원본 22개 + mtp_angle_r/l)",
    "단위": "radians (inDegrees=no)",
    "특징": "누락된 관절각도 복원"
}
```

---

## 📁 **5. 파일 형식별 상세 스펙**

### 🎯 **A. .mot (Motion) 파일**
```
용도: OpenSim 운동학/역학 데이터
확장자: .mot
인코딩: UTF-8 텍스트
구분자: Tab separated values
헤더: OpenSim 메타데이터

예시 구조:
Coordinates (또는 nColumns=N)
version=1
nRows=200  
nColumns=24
inDegrees=no
endheader
time<TAB>pelvis_tilt<TAB>pelvis_list<TAB>...
0.0<TAB>-0.0<TAB>0.0<TAB>...
```

### 🎯 **B. .osim (Model) 파일** 
```
용도: OpenSim 인체 모델 정의
확장자: .osim
형식: XML
내용: 관절구조, 근육, 물리속성

구조:
- Bodies (신체 세그먼트)
- Joints (관절 연결)  
- Muscles (근육 정의)
- Forces (외력 정의)
- Geometry (3D 형상)
```

---

## 💾 **6. 데이터 활용 방법**

### 🔬 **OpenSim에서 활용**
```bash
# 1. OpenSim GUI에서 로드
File > Open Model > example_opensim_model.osim
Tools > Analyze Tool > 생성된 .mot 파일 로드

# 2. Python OpenSim API
import opensim as osim
model = osim.Model('example_opensim_model.osim')
motion = osim.Storage('gait_sample_2sec_grf_pred___.mot')
```

### 📊 **MATLAB에서 분석**
```matlab
% MOT 파일 읽기
data = importdata('gait_sample_2sec_grf_pred___.mot', '\t', 6);
time = data.data(:,1);
force_ry = data.data(:,3); % 오른발 수직력
force_ly = data.data(:,12); % 왼발 수직력

% 보행 주기 분석
plot(time, force_ry, time, force_ly);
xlabel('Time (s)'); ylabel('Force (N)');
```

### 🐍 **Python에서 처리**
```python
import pandas as pd
import numpy as np

# GRF 데이터 로드
grf_data = pd.read_csv('gait_sample_2sec_grf_pred___.mot', 
                       sep='\t', skiprows=6)

# 보행 사이클 추출
right_heel_strikes = find_peaks(grf_data['force1_vy'])[0]
left_heel_strikes = find_peaks(grf_data['force2_vy'])[0]

# 최대힘 분석
max_right_force = grf_data['force1_vy'].max()
max_left_force = grf_data['force2_vy'].max()
```

---

## 🎯 **7. 실제 사용 시나리오**

### 🏥 **임상 응용**
1. **환자 보행 분석**: 불완전한 모션캡처 → 완전한 관절각도
2. **재활 모니터링**: 치료 전후 GRF 패턴 비교
3. **보조기 설계**: 개인별 맞춤형 GRF 데이터

### 🤖 **로봇 응용**  
1. **휴머노이드 제어**: 자연스러운 보행 패턴 생성
2. **시뮬레이션**: 다양한 체형/속도에서의 보행 예측
3. **학습 데이터**: 강화학습용 리워드 함수

### 📚 **연구 응용**
1. **데이터 증강**: 기존 불완전한 데이터셋 보완  
2. **가상 실험**: 실제 실험 없이 보행 패턴 예측
3. **바이오메카닉스**: 관절 부하, 근육 활성화 분석

이제 GaitDynamics의 모든 데이터 형식과 저장 위치를 완벽히 파악했습니다! 🎉