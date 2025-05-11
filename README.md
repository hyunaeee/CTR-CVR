## Real-time CTR/CVR Prediction and Bid Optimization | 실시간 CTR/CVR 예측 및 입찰 최적화


### Project Overview | 프로젝트 개요
This repository implements a complete system for real-time Click-Through Rate (CTR) and Conversion Rate (CVR) prediction with bid optimization for online advertising platforms. The system is designed to handle high-throughput scenarios with low-latency requirements typically found in Real-Time Bidding (RTB) environments.

이 리포지토리는 온라인 광고 플랫폼을 위한 실시간 클릭률(CTR) 및 전환율(CVR) 예측과 입찰 최적화 시스템을 구현합니다. 이 시스템은 실시간 입찰(RTB) 환경에서 흔히 발생하는 고처리량, 저지연 요구사항을 처리할 수 있도록 설계되었습니다.

### Key Features | 주요 기능
- **Multi-stage prediction pipeline**: GBDT + Deep Learning hybrid architecture for accurate CTR/CVR prediction
- **Real-time feature processing**: Efficient feature engineering pipeline with numeric and categorical feature handling
- **Bid optimization strategy**: Thompson sampling based bid optimizer with budget constraints
- **Model explainability**: Feature importance analysis and prediction explanation tools
- **Evaluation framework**: Comprehensive offline and online evaluation metrics
- **Deployment ready**: Docker containerization and serving API

- **다단계 예측 파이프라인**: GBDT + 딥러닝 하이브리드 아키텍처를 통한 정확한 CTR/CVR 예측
- **실시간 특성 처리**: 수치형 및 범주형 특성을 효율적으로 처리하는 특성 엔지니어링 파이프라인
- **입찰 최적화 전략**: 예산 제약 조건을 고려한 톰슨 샘플링 기반 입찰 최적화
- **모델 설명 가능성**: 특성 중요도 분석 및 예측 설명 도구
- **평가 프레임워크**: 포괄적인 오프라인 및 온라인 평가 지표
- **배포 준비 완료**: Docker 컨테이너화 및 서빙 API

### Problem Statement | 문제 정의
In online advertising, accurately predicting CTR and CVR is crucial for maximizing campaign performance. This project addresses:
1. How to build models that can predict user engagement probabilities in milliseconds
2. How to translate these predictions into optimal bid prices
3. How to continuously update models with fresh data

온라인 광고에서 CTR과 CVR을 정확하게 예측하는 것은 캠페인 성과를 최대화하는 데 중요합니다. 이 프로젝트는 다음 문제를 해결합니다:
1. 밀리초 단위로 사용자 참여 확률을 예측할 수 있는 모델을 구축하는 방법
2. 이러한 예측을 최적의 입찰 가격으로 변환하는 방법
3. 새로운 데이터로 모델을 지속적으로 업데이트하는 방법

### Data | 데이터
This implementation uses the [Criteo Dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) for training and validation. The dataset contains 45 million click records over 7 days with the following features:
- Label: Click (1) or No-click (0)
- 13 dense numerical features
- 26 categorical features (hashed)

이 구현은 [Criteo 데이터셋](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)을 훈련 및 검증에 사용합니다. 이 데이터셋은 7일 동안의 4,500만 건의 클릭 기록을 포함하며 다음과 같은 특성이 있습니다:
- 라벨: 클릭(1) 또는 비클릭(0)
- 13개의 조밀한 수치형 특성
- 26개의 범주형 특성(해시 처리됨)

### Model Architecture | 모델 아키텍처
The solution implements a two-stage model:
1. **GBDT (LightGBM)**: For feature interaction extraction and initial prediction
2. **Deep Neural Network**: For fine-grained prediction using raw features and GBDT outputs

이 솔루션은 2단계 모델을 구현합니다:
1. **GBDT (LightGBM)**: 특성 상호작용 추출 및 초기 예측용
2. **딥 뉴럴 네트워크**: 원시 특성과 GBDT 출력을 사용한 세밀한 예측용

![Model Architecture](https://raw.githubusercontent.com/username/rtb-prediction/main/docs/images/model_architecture.png)

### Performance Results | 성능 결과

| Model | AUC | LogLoss | Time/inference |
|-------|-----|---------|----------------|
| Logistic Regression | 0.749 | 0.458 | 0.2ms |
| LightGBM | 0.783 | 0.437 | 0.8ms |
| DeepFM | 0.789 | 0.431 | 1.2ms |
| **Our Hybrid Model** | **0.802** | **0.425** | **1.5ms** |

### Bid Optimization | 입찰 최적화
The bid optimization module uses Thompson sampling with a dynamic budget allocation strategy to maximize campaign performance (clicks or conversions) within budget constraints.

입찰 최적화 모듈은 예산 제약 내에서 캠페인 성과(클릭 또는 전환)를 최대화하기 위해 동적 예산 할당 전략과 함께 톰슨 샘플링을 사용합니다.

Key features | 주요 기능:
- Real-time budget pacing | 실시간 예산 페이싱
- Exploration-exploitation trade-off management | 탐색-활용 트레이드오프 관리
- Campaign-level performance optimization | 캠페인 수준 성능 최적화

### System Architecture | 시스템 아키텍처
The system is designed for scalable, real-time inference:

이 시스템은 확장 가능한 실시간 추론을 위해 설계되었습니다:

```
User Request → Feature Service → Prediction Service → Bid Optimizer → Auction
                     ↑               ↑                    ↑
                Feature Store ← Training Pipeline ← Data Collection

사용자 요청 → 특성 서비스 → 예측 서비스 → 입찰 최적화 → 경매
                  ↑            ↑              ↑
              특성 저장소 ← 훈련 파이프라인 ← 데이터 수집
```

### Installation and Usage | 설치 및 사용법
```bash
# Clone the repository | 저장소 복제
git clone https://github.com/username/rtb-prediction.git
cd rtb-prediction

# Setup environment | 환경 설정
pip install -r requirements.txt

# Download and prepare data | 데이터 다운로드 및 준비
python scripts/prepare_data.py --download --preprocess

# Train the model | 모델 훈련
python train.py --config configs/hybrid_model.yaml

# Run prediction server | 예측 서버 실행
python serve.py --model-path models/hybrid_model_v1.0
```

### Files Structure | 파일 구조
```
├── configs/            # Configuration files | 설정 파일
├── data/               # Data processing scripts | 데이터 처리 스크립트
├── models/             # Model implementations | 모델 구현
│   ├── feature_eng/    # Feature engineering modules | 특성 엔지니어링 모듈
│   ├── gbdt/           # LightGBM implementation | LightGBM 구현
│   ├── deep/           # Deep learning models | 딥러닝 모델
│   └── ensemble/       # Model ensembling | 모델 앙상블
├── bid_optimizer/      # Bid optimization algorithms | 입찰 최적화 알고리즘
├── evaluation/         # Evaluation metrics and tools | 평가 지표 및 도구
├── serving/            # Inference serving code | 추론 서빙 코드
└── notebooks/          # Analysis notebooks | 분석 노트북
```

### Future Work | 향후 작업
- Implement multi-task learning for joint CTR/CVR optimization | 공동 CTR/CVR 최적화를 위한 다중 작업 학습 구현
- Add recurrent models for user sequence modeling | 사용자 시퀀스 모델링을 위한 순환 모델 추가
- Implement transfer learning from large to small campaigns | 대규모에서 소규모 캠페인으로의 전이 학습 구현
- Add reinforcement learning for bid optimization | 입찰 최적화를 위한 강화 학습 추가

### References | 참고 문헌
1. Zhou, G., et al. (2018). Deep Interest Network for Click-Through Rate Prediction. KDD 2018.
2. McMahan, H.B., et al. (2013). Ad Click Prediction: a View from the Trenches. KDD 2013.
3. Chapelle, O. (2015). Offline Evaluation of Response Prediction in Online Advertising Auctions. WWW 2015.

### Contact | 연락처
For questions or collaboration, please open an issue or contact [your-email@example.com](mailto:hyunaeee@gmail.com).

질문이나 협업을 위해서는 이슈를 열거나 [your-email@example.com](mailto:hyunaeee@gmail.com)으로 문의해 주세요.
