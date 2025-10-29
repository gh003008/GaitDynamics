# 🎯 GaitDynamics 조건부 생성 기능 가이드

## 📋 GaitDynamics의 주요 기능들

### 🔧 **1. 기본 설정 파라미터 (`args.py`)**

#### 🏗️ **모델 구조 설정**
```python
--window_len 150           # 시계열 윈도우 길이 (기본: 150 프레임 = 1.5초)
--target_sampling_rate 100 # 샘플링 레이트 (기본: 100Hz)
--with_arm False           # 팔 관절 포함 여부 (기본: False)
--with_kinematics_vel True # 관절 속도 포함 여부 (기본: True)
```

#### 🎨 **Diffusion 모델 설정**
```python
--batch_size_inference 32  # 추론시 배치 크기
--guide_x_start_the_beginning_step -10  # 가이던스 시작 스텝 (-10: 비활성화)
```

#### 📁 **데이터 경로 설정**
```python
--processed_data_dir "dataset_backups/"  # 데이터셋 경로
--checkpoint ""                          # Diffusion 모델 체크포인트
--checkpoint_bl ""                       # Baseline 모델 체크포인트
```

---

## 🎮 **2. 조건부 생성 방법들**

### 🔹 **A. 인체 특성 조건부 생성**
현재 서버 버전에서 설정 가능한 조건들:

```python
# 📏 신체 특성
opt.height_m = 1.75      # 키 (미터)
opt.weight_kg = 70.0     # 몸무게 (킬로그램)

# 🏃 보행 특성  
opt.treadmill_speed = 1.2  # 트레드밀 속도 (m/s, 0=overground)
```

### 🔹 **B. Missing Data Inpainting (현재 구현됨)**
```python
# 자동으로 감지되는 누락 데이터:
missing_columns = ['mtp_angle_r', 'mtp_angle_r_vel', 
                   'mtp_angle_l', 'mtp_angle_l_vel']

# Diffusion 모델이 자동으로 보완:
constraint = {
    'mask': masks,           # 어떤 부분이 누락되었는지
    'value': known_data,     # 알려진 데이터 값들
    'cond': conditioning     # 조건부 벡터
}
```

### 🔹 **C. 관절별 선택적 생성**
```python
# 특정 관절만 생성하고 싶을 때:
opt.knee_diffusion_col_loc     # 무릎 관절만
opt.ankle_diffusion_col_loc    # 발목 관절만  
opt.hip_diffusion_col_loc      # 고관절만
opt.kinematic_diffusion_col_loc # 모든 운동학적 데이터
opt.kinetic_diffusion_col_loc   # 모든 역학적 데이터 (힘)
```

---

## 🚀 **3. 고급 조건부 생성 활용법**

### 💡 **A. 체형별 맞춤 생성**
```python
# 다양한 체형으로 실험:
체형_조건들 = [
    {"height_m": 1.60, "weight_kg": 55.0},  # 소형 체형
    {"height_m": 1.75, "weight_kg": 70.0},  # 평균 체형  
    {"height_m": 1.90, "weight_kg": 95.0},  # 대형 체형
]
```

### 💡 **B. 보행 속도별 생성**
```python
# 다양한 보행 패턴:
속도_조건들 = [
    {"treadmill_speed": 0.8},   # 느린 걷기
    {"treadmill_speed": 1.2},   # 일반 걷기
    {"treadmill_speed": 1.8},   # 빠른 걷기
    {"treadmill_speed": 0.0},   # Overground (자연 보행)
]
```

### 💡 **C. 질환별 조건부 생성 (확장 가능)**
```python
# 향후 확장 가능한 조건들:
pathology_conditions = {
    "normal": [1, 0, 0, 0, 0],
    "hemiplegia": [0, 1, 0, 0, 0], 
    "parkinson": [0, 0, 1, 0, 0],
    "prosthetic": [0, 0, 0, 1, 0],
    "elderly": [0, 0, 0, 0, 1]
}
```

---

## 🛠️ **4. 실제 사용 예시**

### 🎯 **시나리오 1: 다양한 체형으로 실험**
```python
# S_gait_dynamics_server.py 수정:
def usr_inputs():
    opt = parse_opt()
    
    # 실험할 체형들
    체형_리스트 = [
        (1.60, 50.0, "petite"),
        (1.75, 70.0, "average"), 
        (1.90, 90.0, "tall")
    ]
    
    for height, weight, name in 체형_리스트:
        opt.height_m = height
        opt.weight_kg = weight
        print(f"🧍 체형: {name} ({height}m, {weight}kg)")
        # 각 체형별로 결과 생성
```

### 🎯 **시나리오 2: 보행 속도 변화 실험**
```python
속도_리스트 = [0.8, 1.0, 1.2, 1.5, 1.8]  # m/s

for speed in 속도_리스트:
    opt.treadmill_speed = speed
    print(f"🏃 속도: {speed} m/s")
    # 각 속도별로 결과 생성
```

### 🎯 **시나리오 3: 특정 관절만 예측**
```python
# 무릎 관절만 예측하고 나머지는 원본 유지:
col_loc_to_unmask = opt.knee_diffusion_col_loc
windows, s_list, e_list = dataset.get_overlapping_wins(
    col_loc_to_unmask, 20, i_trial, i_trial+1
)
```

---

## 📊 **5. 결과 활용 및 분석**

### 🔬 **조건별 결과 비교**
```python
# 체형별 GRF 패턴 비교
for condition in conditions:
    grf_max = analyze_grf_peak(condition)
    walking_pattern = analyze_kinematics(condition)
    print(f"{condition}: GRF={grf_max}N, Pattern={walking_pattern}")
```

### 📈 **시각화 및 검증**
```python
# 조건별 결과를 matplotlib으로 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, condition in enumerate(conditions):
    plot_gait_cycle(axes[i//2, i%2], condition)
plt.suptitle("조건별 보행 패턴 비교")
plt.show()
```

---

## ⚙️ **6. 파라미터 튜닝 가이드**

### 🎛️ **성능 최적화**
```python
# GPU 메모리에 따른 배치 크기 조정:
opt.batch_size_inference = 32   # RTX 3080 기준
opt.batch_size_inference = 64   # RTX 4090 기준  
opt.batch_size_inference = 16   # GTX 1080 기준
```

### 🎛️ **품질 vs 속도 트레이드오프**
```python
# 고품질 생성 (느림):
diffusion_steps = 1000
guidance_weight = 2.0

# 빠른 생성 (품질 약간 하락):
diffusion_steps = 250  
guidance_weight = 1.5
```

---

## 💡 **실제 연구/임상 활용 예시**

1. **🏥 재활 치료**: 환자별 맞춤형 보행 패턴 생성
2. **🤖 로봇공학**: 다양한 체형의 휴머노이드 로봇 제어
3. **🎮 게임/VR**: 실시간 캐릭터 애니메이션 생성
4. **📊 스포츠 과학**: 운동선수 보행 분석 및 최적화

이제 GaitDynamics의 모든 조건부 생성 기능을 활용할 수 있습니다! 🎉