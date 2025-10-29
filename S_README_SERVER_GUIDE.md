# 🚀 GaitDynamics 서버 환경 사용 가이드

## 📁 파일 구조 설명

### 🔹 원본 파일들 (Stanford 제공)
- `args.py`, `consts.py` - 설정 파일들
- `train_models.py` - 모델 훈련 스크립트  
- `gait_sample_2sec.mot` - 샘플 보행 데이터
- `data/`, `model/`, `model_baseline/`, `figures/` - 패키지 구조
- `example_usage/gait_dynamics.py` - 원본 Colab 버전

### 🔸 우리가 만든 파일들 (S_ 접두사)
- `S_gait_dynamics_server.py` - **❌ 사용하지 마세요 (복잡한 버전)**
- `S_gait_dynamics_full.py` - **❌ 사용하지 마세요 (복잡한 버전)**  
- `S_gait_dynamics_complete.py` - **❌ 사용하지 마세요 (복잡한 버전)**
- `S_simple_test.py` - 환경 테스트용
- `S_test_pretrained.py` - 기본 모델 테스트용

### ✅ **추천 사용 파일**: `example_usage/S_gait_dynamics_server.py`
**이 파일이 최종 완성된 서버 버전입니다!**

---

## 🎯 사용 방법

### 1. 환경 활성화
```bash
conda activate gaitdyn
cd /home/ghlee/GaitDynamics/example_usage
```

### 2. GaitDynamics 실행
```bash
python S_gait_dynamics_server.py
```

### 3. 결과 분석
```bash
python S_analyze_results.py
```

---

## 📊 결과 파일들

### 🦶 Ground Reaction Forces
- `gait_sample_2sec_grf_pred___.mot` - 예측된 지면반발력
- 200 데이터 포인트 (0-1.99초, 100Hz)
- 오른발/왼발 3차원 힘벡터

### 🚶 Missing Kinematics  
- `gait_sample_2sec_missing_kinematics_pred___.mot` - 보완된 관절각도
- 원본에 없던 `mtp_angle_r`, `mtp_angle_l` (발가락 관절) 추가

---

## 🛠️ 주요 수정사항

1. **Colab 업로드 UI 제거** → 자동 파일 검색
2. **대화형 입력 제거** → 하드코딩된 기본값 사용
3. **경로 문제 수정** → 정확한 모델 파일 경로
4. **missing column 안전 처리** → try-catch로 오류 방지

---

## 📝 개발 히스토리

1. `S_simple_test.py` - 환경 설정 검증
2. `S_test_pretrained.py` - 모델 로딩 테스트  
3. `S_gait_dynamics_full.py` - 복잡한 첫 번째 시도 (❌)
4. `S_gait_dynamics_server.py` (in example_usage/) - **✅ 성공한 최종 버전**

---

## 💡 활용 방안

- 🏥 **임상**: 보행 분석, 재활 치료 모니터링
- 🤖 **로봇공학**: 휴머노이드 로봇 보행 제어
- 🔬 **연구**: 바이오메카닉스, OpenSim 시뮬레이션
- 📊 **데이터 과학**: 보행 데이터셋 확장

---

**⚠️ 중요**: 실제 사용 시에는 `example_usage/S_gait_dynamics_server.py`만 사용하세요!