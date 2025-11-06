# WandB Logging Metrics Guide

## 📊 추가된 학습 건강도 모니터링 메트릭

### 1. Hyperparameters (Sweep 추적용)
```
wandb.config:
  - learning_rate
  - weight_decay
  - batch_size
  - pseudo_dataset_len
  - epochs
  - window_len
  - num_params
  - num_train_trials
```

### 2. 초기 데이터 통계 (Baseline)
```
data_stats/normalized_mean
data_stats/normalized_std
data_stats/normalized_min
data_stats/normalized_max
data_stats/{channel_name}_mean
data_stats/{channel_name}_std
```
**목적**: 정규화된 데이터의 분포 확인

---

## 🔍 매 Epoch 로깅 메트릭

### 3. Gradient Health (학습 안정성)
```
train/grad_norm_mean      # 평균 gradient norm
train/grad_norm_std       # Gradient 분산
train/grad_norm_max       # 최대 gradient
train/grad_norm_min       # 최소 gradient
```

**건강한 학습 신호**:
- `grad_norm_mean`: 0.01 ~ 10 범위 (너무 크면 exploding, 너무 작으면 vanishing)
- `grad_norm_std`: 안정적으로 유지 (급격한 변화 = 불안정)

**경고 신호**:
- ⚠️ grad_norm > 100: Exploding gradients
- ⚠️ grad_norm < 0.001: Vanishing gradients
- ⚠️ grad_norm_std 급증: 불안정한 학습

---

### 4. Batch Loss Statistics (Mode Collapse 감지)
```
train/batch_loss_mean     # 배치별 loss 평균
train/batch_loss_std      # 배치별 loss 분산
train/batch_loss_max
train/batch_loss_min
train/batch_loss_cv       # Coefficient of Variation
```

**건강한 학습 신호**:
- `batch_loss_std` > 0: 배치마다 다양한 loss (정상)
- `batch_loss_cv` > 0.05: 적당한 분산

**경고 신호 (Mode Collapse)**:
- ⚠️ batch_loss_std < 0.01: 모든 배치가 비슷한 loss → 모델이 같은 것만 생성
- ⚠️ batch_loss_cv < 0.01: 분산 너무 작음

---

### 5. Model Weight Statistics (Dead Neurons / Exploding Weights)
```
model/weight_mean         # 평균 weight 크기
model/weight_std          # Weight 분산
model/weight_max          # 최대 weight
```

**건강한 학습 신호**:
- 안정적으로 유지되거나 천천히 증가
- weight_mean: 0.01 ~ 1 정도

**경고 신호**:
- ⚠️ weight_max > 10: Exploding weights
- ⚠️ weight_mean < 0.001: Dead neurons 가능성
- ⚠️ 급격한 변화: 불안정

---

### 6. EMA Model Tracking
```
model/ema_diff            # EMA 모델과 현재 모델의 차이
```

**건강한 학습 신호**:
- 점진적으로 감소하거나 안정적 유지
- ema_diff: 0.001 ~ 0.1

**경고 신호**:
- ⚠️ ema_diff 급증: 모델이 너무 빠르게 변화
- ⚠️ ema_diff > 1: 학습 불안정

---

### 7. Loss Improvement Tracking
```
train/loss_decrease       # 이전 epoch 대비 loss 감소량
train/loss_decrease_pct   # 감소 퍼센트
train/is_improving        # 1=개선중, 0=정체/악화
```

**건강한 학습 신호**:
- loss_decrease > 0 (계속 감소)
- loss_decrease_pct: 초반 5-10%, 후반 0.1-1%

**경고 신호**:
- ⚠️ is_improving = 0이 10+ epoch 연속: 학습 정체
- ⚠️ loss_decrease < 0: Loss 증가 (overfitting 또는 불안정)

---

### 8. Loss Component Ratios
```
loss_ratio/simple
loss_ratio/vel
loss_ratio/fk
loss_ratio/drift
loss_ratio/slide
```

**목적**: 어떤 loss term이 지배적인지 확인

**건강한 학습**:
- 비율이 크게 변하지 않고 안정적

**문제 신호**:
- ⚠️ 한 term이 0.9+ 차지: 다른 항들이 무시됨

---

## ⭐ 가장 중요: Validation Sample Quality (50 epoch마다)

### 9. Generated Sample Statistics
```
validation/gen_std        # 생성된 샘플 전체 std
validation/gen_mean
validation/gen_range
```

**건강한 생성**:
- gen_std > 0.5 (정규화된 값 기준)
- gen_range > 2.0

**Mode Collapse 경고**:
- ⚠️ gen_std < 0.1: 거의 상수 생성
- ⚠️ gen_range < 0.5: 다양성 없음

---

### 10. Knee Angle Quality Check
```
validation/knee_r_std     # Knee angle 표준편차 (radians)
validation/knee_r_mean
validation/knee_healthy   # 1=정상, 0=비정상
```

**건강한 생성**:
- knee_r_std > 0.1 rad (~5.7°)
- knee_healthy = 1

**Mode Collapse 확정**:
- ⚠️ knee_r_std < 0.05 rad (~3°): 거의 상수
- ⚠️ knee_healthy = 0: 비정상 생성

**정상 보행 기준**: Knee ROM 60-70°, std ~20-30° (0.35-0.52 rad)

---

### 11. GRF (Ground Reaction Force) Quality Check
```
validation/grf_vz_r_std   # GRF 표준편차 (Newtons)
validation/grf_vz_r_mean
validation/grf_healthy    # 1=정상, 0=비정상
```

**건강한 생성**:
- grf_vz_r_std > 10 N
- grf_healthy = 1

**Mode Collapse 확정**:
- ⚠️ grf_vz_r_std < 1 N: 거의 상수
- ⚠️ grf_healthy = 0: 비정상 생성

**정상 보행 기준**: GRF std ~100-300 N

---

## 🚨 Mode Collapse 조기 감지 체크리스트

다음 조건 중 **3개 이상** 해당하면 Mode Collapse:

1. ⚠️ `batch_loss_cv < 0.01` (배치 loss 분산 너무 작음)
2. ⚠️ `validation/knee_healthy = 0` (무릎 각도 비정상)
3. ⚠️ `validation/grf_healthy = 0` (GRF 비정상)
4. ⚠️ `validation/gen_std < 0.1` (생성 샘플 분산 너무 작음)
5. ⚠️ `train/is_improving = 0` 연속 20+ epochs (학습 정체)
6. ⚠️ `grad_norm_std / grad_norm_mean < 0.1` (gradient 다양성 없음)

---

## 📈 학습 단계별 기대 패턴

### Phase 1: 초기 학습 (Epoch 1-50)
- `train/total_loss`: 급격히 감소 (예: 1.0 → 0.3)
- `grad_norm_mean`: 크지만 안정적 (1-10)
- `validation/gen_std`: 점진적 증가 (0.3 → 0.8)
- `train/is_improving = 1` 지속

### Phase 2: 수렴 (Epoch 50-200)
- `train/total_loss`: 천천히 감소 (0.3 → 0.15)
- `grad_norm_mean`: 감소 및 안정화 (1-3)
- `validation/knee_healthy = 1` 달성
- `validation/grf_healthy = 1` 달성

### Phase 3: Fine-tuning (Epoch 200+)
- `train/total_loss`: 미세 감소 (0.15 → 0.12)
- `loss_decrease_pct < 1%` 지속
- 모든 validation 메트릭 안정적 유지

---

## 🎯 WandB Sweep에서 확인할 핵심 메트릭

### Sweep 비교 시 우선순위:

1. **최종 성능**:
   - `validation/knee_healthy` (마지막 값 = 1)
   - `validation/grf_healthy` (마지막 값 = 1)
   - `train/total_loss` (최소값)

2. **학습 안정성**:
   - `train/grad_norm_mean` (안정적 유지)
   - `model/ema_diff` (작고 안정적)

3. **수렴 속도**:
   - `train/total_loss` 곡선 기울기
   - 몇 epoch에 `validation/knee_healthy = 1` 달성?

---

## 🔧 문제별 진단 및 해결

### 문제 1: Loss는 감소하는데 validation 품질이 나쁨
**증상**:
- ✓ train/total_loss 감소
- ✗ validation/knee_healthy = 0
- ✗ validation/grf_healthy = 0

**원인**: Overfitting 또는 loss function 문제
**해결**: 
- Pseudo dataset 크기 증가 (10k → 20k)
- Learning rate 감소

---

### 문제 2: Loss가 안 떨어짐
**증상**:
- ✗ train/is_improving = 0 연속
- ✗ loss_decrease_pct < 0.1%

**원인**: Learning rate 너무 낮음
**해결**:
- Learning rate 증가 (1e-4 → 4e-4)

---

### 문제 3: Gradient 폭발
**증상**:
- ✗ grad_norm_max > 100
- ✗ 학습 중 NaN 발생

**원인**: Learning rate 너무 높음
**해결**:
- Learning rate 감소 (4e-4 → 1e-4)
- Gradient clipping 추가

---

### 문제 4: Mode Collapse 확정
**증상**:
- ✗ validation/gen_std < 0.1
- ✗ batch_loss_cv < 0.01
- ✗ knee_healthy = 0, grf_healthy = 0

**원인**: 
- Learning rate 너무 높음
- 학습 데이터 부족
- Batch size 너무 작음

**해결**:
- Learning rate 낮추기 (1e-4)
- Batch size 키우기 (64)
- 더 오래 학습 (1000 epochs)

---

## 📊 WandB Dashboard 추천 Layout

### Panel 1: Training Health
- `train/total_loss` (line)
- `train/grad_norm_mean` (line)
- `train/is_improving` (bar)

### Panel 2: Validation Quality ⭐
- `validation/knee_r_std` (line) + threshold line at 0.1
- `validation/grf_vz_r_std` (line) + threshold line at 10
- `validation/knee_healthy` & `validation/grf_healthy` (bar)

### Panel 3: Mode Collapse Detection
- `batch_loss_cv` (line)
- `validation/gen_std` (line)
- `model/ema_diff` (line)

### Panel 4: Loss Components
- `loss_ratio/*` (stacked area chart)
