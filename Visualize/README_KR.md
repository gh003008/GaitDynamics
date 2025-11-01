# Visualize 폴더 안내 (한국어)

이 폴더에는 AddBiomechanics(.b3d) 보행 데이터를 빠르게 확인하기 위한 시각화 스크립트가 포함되어 있습니다.

현재 제공 스크립트:
- `V01_visualize_ab_walk.py`: 한 개의 .b3d 파일을 열어, 이름에 "walk"가 포함된 트라이얼을 찾아 하체 스켈레톤(골반~발끝)을 간단한 스틱 피겨로 애니메이션 저장(MP4/GIF)합니다.
- `V02_gui_ab_walk.py`: 논문/데모 스타일처럼 NimbleGUI를 띄워, 실시간으로 스켈레톤과 GRF 라인을 시각화합니다(figures/fig_utils.py의 방식을 재사용).

## 의존성
- Python 패키지: `numpy`, `matplotlib`
- 핵심: `nimblephysics` (AddBiomechanics의 .b3d 파일 로딩, 스켈레톤 사용)
  - 주의: nimblephysics는 Python 3.10/3.11 환경에서 배포되는 경우가 많습니다. 현재 시스템 Python이 3.13이면 설치가 어려울 수 있으므로, 별도의 conda 환경을 권장합니다.

예시 (conda 권장 흐름):
1) Python 3.10 환경 만들기
2) nimblephysics 설치
3) numpy, matplotlib 설치 후 스크립트 실행

각 환경 별 설치법은 nimblephysics의 공식 문서를 참고하세요: https://github.com/keenon/nimblephysics

## 사용법
기본적으로 레포 내부 샘플 경로를 기본값으로 잡고 있으므로, 아래처럼 바로 실행할 수 있습니다.

```bash
python3 Visualize/V01_visualize_ab_walk.py --frames 200 --fps 60
```

주요 옵션:
- `--b3d`: 시각화할 .b3d 파일 경로 (기본값: `data/Wang2023_Formatted_No_Arm/.../Subj06.b3d`)
- `--trial_substr`: 트라이얼 이름에 포함될 키워드(기본: `walk`)
- `--out_dir`: 결과 동영상/이미지 저장 경로(기본: `previews/`)
- `--start`: 시작 프레임(기본: 0)
- `--frames`: 렌더링 프레임 수(기본: 250)
- `--fps`: 저장 FPS(기본: 60)

출력:
- 우선 MP4를 시도합니다(ffmpeg 필요). ffmpeg가 없을 경우 GIF로 폴백합니다. GIF도 불가하면 마지막 프레임 PNG를 저장합니다.

### NimbleGUI 방식(실시간 미리보기)

```bash
python3 Visualize/V02_gui_ab_walk.py --frames 300
```

- 브라우저로 NimbleGUI가 열리고, 하체 스켈레톤과 발 접촉점 기준의 GRF 라인이 프레임마다 업데이트됩니다.
- 서버 환경에서는 포트(기본 8090부터 순차 증가)를 열어 웹 브라우저로 접속해 보시면 됩니다.

## 트러블슈팅
- `ModuleNotFoundError: nimblephysics`: nimblephysics 미설치. Python 버전 호환 및 conda 환경 확인 후 nimblephysics 설치가 필요합니다.
- `ffmpeg` 관련 에러: 시스템에 ffmpeg가 없다면 GIF로 자동 전환됩니다.
- 스켈레톤이 펼쳐지지 않거나 화면 밖에 있는 경우: `--frames`, `--start`를 조정하거나, 카메라 각도는 스크립트 코드 내 `ax.view_init`에서 간단히 바꿀 수 있습니다.
