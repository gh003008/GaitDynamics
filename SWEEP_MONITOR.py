"""
Sweep Monitoring Guide
======================

Sweep 실행 중: 8 configs, 예상 ~5일

중간 체크 포인트 (50 epoch마다):
----------------------------------

1. Mode Collapse 조기 감지 (즉시 중단 필요):
   ✗ validation/knee_healthy = 0
   ✗ validation/grf_healthy = 0  
   ✗ validation/gen_std < 0.1
   ✗ batch_loss_cv < 0.01

2. 학습 불안정 (경고):
   ⚠ grad_norm_mean > 100 (exploding)
   ⚠ grad_norm_mean < 0.001 (vanishing)
   ⚠ train/is_improving = 0 연속 20+ epochs

3. 정상 학습 (계속 진행):
   ✓ validation/knee_healthy = 1
   ✓ validation/grf_healthy = 1
   ✓ train/loss_decrease_pct > 0
   ✓ grad_norm_mean: 0.1~10 범위

WandB 대시보드에서 확인:
------------------------
https://wandb.ai/your-team/MotionModel/sweeps/

추천 Panel 구성:
- validation/knee_healthy (bar chart)
- validation/grf_healthy (bar chart)  
- train/total_loss (line, 모든 runs 오버레이)
- validation/knee_r_std (line, threshold=0.1)

중단 기준:
---------
만약 처음 2-3개 config가 모두 mode collapse면:
→ Learning rate 범위를 더 낮춰야 함 (5e-5, 1e-4)
→ Sweep 중단하고 재설정

좋은 결과 발견시:
----------------
해당 config의 checkpoint 사용:
runs/train/<exp_name>/weights/train-<epoch>_diffusion.pt

테스트:
python LD03_test_generation.py --exp_name <best_exp_name> --num_samples 10
"""
print(__doc__)
