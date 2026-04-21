# MovieLens AutoInt Recommendation Learning Project

MovieLens 1M 데이터를 사용해 **recommendation system**의 feature interaction 모델을 학습한 교육 과정 산출물입니다. 포트폴리오 대표작이라기보다는, 추천시스템과 딥러닝 모델 구현을 학습하며 남긴 **Learning Project**로 정리했습니다.

## Project Goal

- MovieLens 사용자, 영화, 평점 데이터를 추천 모델 입력 형태로 전처리합니다.
- AutoInt(Self-Attention based Feature Interaction) 구조를 이해하고 구현합니다.
- AutoInt 출력에 MLP를 결합한 AutoInt+ 형태를 실험합니다.
- Streamlit으로 간단한 추천 결과 확인 화면을 구성합니다.

## Tech Stack

| Area | Stack |
|---|---|
| Language | Python |
| Modeling | TensorFlow, AutoInt, MLP |
| Data | MovieLens 1M |
| Preprocessing | pandas, NumPy, scikit-learn |
| Demo | Streamlit |

## Repository Structure

```text
DS4-Main-Quest/
├── README.md
├── autointMLP.py
└── autoint/
    ├── README.md
    ├── autoint.py
    ├── show_st.py
    ├── submit.ipynb
    └── train_autoint_mlp.py
```

## What I Practiced

- 범주형 feature를 모델 입력으로 바꾸는 preprocessing flow
- embedding layer와 feature interaction의 역할
- AutoInt의 self-attention 기반 feature interaction 개념
- MLP branch를 결합해 모델을 확장하는 방식
- 학습 스크립트와 시각화 앱을 분리하는 프로젝트 구조
- 추천 결과를 Streamlit UI로 빠르게 확인하는 방식

## How To Run

가상환경을 만든 뒤 필요한 패키지를 설치합니다.

```bash
conda create -n recsys python=3.11
conda activate recsys
pip install tensorflow pandas numpy scikit-learn streamlit
```

AutoInt+ 학습:

```bash
python autoint/train_autoint_mlp.py
```

Streamlit demo 실행:

```bash
streamlit run autoint/show_st.py
```

## Notes

- 이 레포는 교육 과정 중 만든 학습 산출물입니다.
- 데이터와 모델 weight가 로컬 환경 기준으로 구성되어 있을 수 있어, 실행 전 경로와 의존성을 확인해야 합니다.
- 대표 포트폴리오보다는 추천시스템 학습 기록으로 두는 것이 적합합니다.

## Limitations

- 실험 결과와 metric tracking이 충분히 구조화되어 있지 않습니다.
- `requirements.txt`가 별도로 정리되어 있지 않아 재현성이 약합니다.
- 모델 성능 비교가 README에 정량적으로 정리되어 있지 않습니다.
- 실제 서비스형 추천 API나 배포 구조까지 포함하지는 않습니다.

## Next Steps

- `requirements.txt` 또는 `environment.yml` 추가
- train/eval 결과를 표로 정리
- NDCG, Hit Rate, AUC 등 metric을 일관되게 기록
- notebook 내용을 script 중심 구조로 정리
- 작은 샘플 데이터로 smoke test 가능하게 만들기
