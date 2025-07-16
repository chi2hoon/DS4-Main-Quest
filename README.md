# MovieLens Recommendation – AutoInt & AutoInt+

본 프로젝트는 MovieLens 1M 데이터셋을 기반으로 다음 두 가지 모델을 구현-학습-시각화한 예제입니다.

1. **AutoInt** : Self-Attention 기반의 Feature Interaction 모델
2. **AutoInt+ (AutoInt × MLP)** : AutoInt 출력과 MLP 출력을 합산해 성능을 향상시킨 확장 버전

---

## 1. 프로젝트 구조

```text
10_RecSys/
├─ autoint/                 # 모델·데이터·스트림릿 앱
│  ├─ autoint.py            # AutoInt 레이어/모델 구현
│  ├─ autointMLP.py         # AutoInt+ 레이어/모델 구현 (TensorFlow 2.x)
│  ├─ train_autoint_mlp.py  # AutoInt+ 학습·평가·가중치 저장 스크립트
│  ├─ show_st.py            # Streamlit 시각화 앱
│  ├─ data/                 # 전처리된 MovieLens 데이터 & field_dims.npy / label_encoders.pkl
│  └─ model/                # 학습된 가중치 (.h5)
└─ README.md                # ⬅ 현재 파일
```

---

## 2. 환경 세팅

```bash
conda create -n recsys python=3.11
conda activate recsys
pip install -r requirements.txt   # requirements.txt 를 직접 작성해도 좋습니다.
```
필수 주요 패키지
* TensorFlow >= 2.11
* Pandas, NumPy, scikit-learn
* Streamlit

---

## 3. 데이터 준비

`autoint/data/ml-1m/` 폴더에는 다음 전처리 파일이 포함되어 있습니다.

* `ratings_prepro.csv`, `movies_prepro.csv`, `users_prepro.csv`
* 각 범주형 컬럼의 cardinality 정보를 담은 `field_dims.npy`
* 학습·예측용 라벨 인코더 객체 `label_encoders.pkl`

전처리 스크립트가 필요하다면 별도 Notebook(`submit.ipynb`)을 참고해 주세요.

---

## 4. 모델 학습

### 4-1. AutoInt (기존)
사전 학습된 가중치 `autoint/model/autoInt_model.weights.h5` 가 이미 제공됩니다.

### 4-2. AutoInt+
```bash
python autoint/train_autoint_mlp.py
```
* MovieLens 1M 데이터 전체(약 1M rows)를 학습
* 기본 하이퍼파라미터: `embed_dim=16`, `dnn_hidden_units=(128,64)`, `epochs=5`, `batch_size=1024`
* 학습 완료 후 가중치가 `autoint/model/autoIntMLP_model.weights.h5` 로 저장됩니다.

하이퍼파라미터 튜닝을 위해 스크립트 상단의 **BATCH_SIZE / EPOCHS / learning_rate / hidden_units** 등을 자유롭게 변경해 성능(NDCG, Hit-Rate, AUC)을 향상시켜 볼 수 있습니다.

---

## 5. Streamlit 시각화

```bash
streamlit run autoint/show_st.py
```
* 사이드바에서 **AutoInt / AutoInt+** 모델 선택 가능
* 사용자 ID·타깃 연/월을 입력하면
  * 사용자 기본 정보 & 과거 시청 이력(평점 ≥ 4)
  * 모델이 예측한 **Top-10 추천 영화**를 동시에 확인할 수 있습니다.

_PTIP_: 가중치 변경 후 캐시를 초기화해야 할 때는 `R` 키(“Reload & Clear cache”) 또는 `streamlit cache clear` 명령을 사용하세요.

---

## 6. 성능 향상 실험 (AutoInt+)

다음 요소를 변경하며 A/B 테스트를 수행했습니다.

| 파라미터 | 기본값 | 실험 값 예시 | 효과 |
|----------|--------|--------------|------|
| Epochs | 5 | 10 / 15 | 과적합에 유의하며 AUC 개선 |
| Learning Rate | 1e-3 | 5e-4, 3e-4 | 수렴 안정성 확보 |
| Dropout | 0.4 | 0.3 / 0.5 | regularization |
| Embedding Dim | 16 | 32 | 표현력 ↑, 학습 시간 ↑ |
| DNN Units | (128,64) | (256,128,64) | 비선형 복잡도 증가 |

실험 로그 및 결과는 WandB / TensorBoard 등을 사용해 추적할 수 있으며, 기본 스크립트에서는 Validation AUC를 표준 출력으로 제공합니다.

---

## 7. 기여 & 라이선스

*본 레포지토리는 교육 목적으로 제작되었습니다. 자유롭게 fork 하여 개선 PR을 보내 주세요.*

MIT License 