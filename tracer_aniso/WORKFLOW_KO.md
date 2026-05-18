# Tracer Aniso — 뉴런 트레이싱 워크플로우

---

## 개요

형광 현미경으로 촬영한 3D TIFF 이미지에서 뉴런의 완전한 SWC 형태 파일을 생성하는 6단계 파이프라인입니다.

```
원시 TIFF
  ↓ step0  — 전처리
  ↓ step1  — Tubularity 맵 (OOF + Structure Tensor)
  ↓ step1b — 소마 검출 및 앵커링
  ↓ step2  — 다운샘플 및 메트릭 준비
  ↓ step3  — Riemannian FMM 트레이싱 → SWC
  ↓ step4  — 시각화 및 형태 분석
```

---

## Step 0 — 전처리 (`step0_preprocess.ipynb`)

**목표:** 원시 TIFF를 정규화된 isotropic float32 볼륨으로 변환합니다.

### 처리 순서
1. **로드** — 다중 슬라이스 TIFF 읽기; TIFF 태그에서 XY 픽셀 크기 자동 감지.
2. **다운샘플** — `DOWNSAMPLE_XY`를 자동 계산하여 출력 복셀 크기가 `TARGET_VOXEL_XY_UM`(기본 0.40 µm) 이하, XY 픽셀 수가 `MAX_XY_PX` 이하가 되도록 조정.
3. **정규화** — 하한: `threshold_triangle`(배경/신호 경계 자동 탐지). 상한: 99.9th percentile(핫픽셀 제거). 출력: float32 [0, 1].
4. **Z 리스케일** — `scipy.ndimage.zoom`으로 Z 방향을 XY와 동일한 복셀 크기로 보간하여 isotropic 볼륨 생성. 비등방성 < 1.2이면 생략.
5. **Noise2Void (선택)** — 정답 이미지 없이 단일 볼륨만으로 학습하는 self-supervised 노이즈 제거기. `N2V_ENABLE = True`로 활성화.

### 주요 출력
| 파일 | 내용 |
|------|------|
| `output/<sample>/stack_preprocessed.tif` | 정규화된 isotropic float32 볼륨 |
| `output/<sample>/preprocess_meta.npz` | `voxel_iso`, 클립 범위, 비등방성 비율 |

### 주요 파라미터
| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `TARGET_VOXEL_XY_UM` | 0.40 | 다운샘플 후 목표 XY 복셀 크기 (µm) |
| `MAX_XY_PX` | 1024 | XY 최대 픽셀 수 |
| `CLIP_HIGH_PERCENTILE` | 99.9 | 상한 클립 백분위수 (희소 신호면 99.99 권장) |
| `N2V_ENABLE` | False | Noise2Void 활성화 여부 |

---

## Step 1 — Tubularity 맵 (`step1_tubularity_oof_v2.ipynb`)

**목표:** 관형 구조(축삭, 수상돌기) 내부에서 높고 배경에서 낮은 스칼라 tubularity 맵 W(x) ∈ [0, 1]을 생성합니다.

### 방법: Guided Weighting

두 독립적인 검출기가 동일한 tube 축을 추정하며, 두 결과의 일치도로 반응을 강화하고 불일치(노이즈/artifact)를 억제합니다.

| 검출기 | 원리 | 출력 |
|--------|------|------|
| **OOF** (최적 방향 플럭스) | 구 표면의 법선 플럭스; 두 횡방향 고유값이 모두 음수일 때 관형 구조로 판단 | `I_OOF(x)`, `v_OOF(x)` |
| **Structure Tensor** | 적분 스케일에서 평활화된 기울기 외적; 최소 고유값 방향 = tube 축 | `v_ST(x)` |

**Guided weighting 수식:**
```
C_align(x) = |⟨v_OOF, v_ST⟩|          # 코사인 유사도, 0–1
W(x)       = I_OOF(x) · (1 + β · C_align(x))
```

OOF가 놓치는 tube 내부와 소마 shell을 채우기 위해 **LoG blob 반응**(`LAMBDA_BLOB`으로 가중)을 추가합니다.

### 멀티스케일
반경은 `TUBE_RADIUS_MIN_UM`에서 `TUBE_RADIUS_MAX_UM`까지 로그 균등하게 `N_RADII`개 샘플링. 복셀별로 최대 반응 스케일을 기록. GPU(MPS/CUDA) 가속을 지원합니다.

### 주요 출력
| 파일 | 내용 |
|------|------|
| `output/<sample>/tubularity.npz` | `T_combined`, `I_OOF_raw`, `orient_field`, `radius_map` |

### 주요 파라미터
| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `TUBE_RADIUS_MIN_UM` | 0.10 | 최소 tube 반경 (µm) |
| `TUBE_RADIUS_MAX_UM` | 2.0 | 최대 tube 반경 (µm) |
| `N_RADII` | 8 | 스케일 수 |
| `BETA` | 1.0 | 방향 일치 강화 세기 |
| `LAMBDA_BLOB` | 0.3 | LoG blob 가중치 |

---

## Step 1b — 소마 검출 (`step1b_soma.ipynb`)

**목표:** 소마를 분할하고, tubularity 맵에서 소마가 전역 최댓값이 되도록 앵커링합니다 — FMM의 진정한 최저 비용 출발점 확보.

### 처리 순서
1. **소마 위치 탐색** — 점수 = 밝기 × (1 − T_combined). 이 점수를 Gaussian 평활화한 후 피크 위치를 소마 후보로 선택. 경계 복셀은 edge artifact 방지를 위해 제외.
2. **소마 분할** — `SOMA_SEARCH_RADIUS_UM` 구체 내 국소 Otsu 임계값 적용. 내부가 빈 소마(hollow): 반복 morphological closing + `fill_holes`. morphological opening/closing/erosion으로 마스크 정제.
3. **소마 앵커링** — 소마 내부를 거리 변환(EDT) 기반 그라디언트로 대체하여 소마 중심이 T=1.0. 전체를 소마 최댓값으로 정규화 → tube 값은 ≤ ~0.67로 압축.

> **앵커링이 필요한 이유:** FMM 경로는 소마에서 출발해야 합니다. 앵커링 없이는 T=1.0으로 포화된 수상돌기 복셀이 소마와 경쟁하여 위상적으로 잘못된 트리가 생성될 수 있습니다.

### 주요 출력
| 파일 | 내용 |
|------|------|
| `output/<sample>/soma.npz` | `soma_mask`, `soma_centroid_vox`, 메시 정점/면 |
| `output/<sample>/soma.json` | `centroid_vox`, `radius_um` |
| `output/<sample>/tubularity_anchored.npz` | 소마를 전역 최대로 설정한 T_combined |

---

## Step 2 — 메트릭 준비 (`step2_prep_aniso.ipynb`)

**목표:** 모든 필드를 다운샘플하고 Riemannian 메트릭에 사용할 EDT 반경 맵을 계산합니다.

### 처리 순서
1. **다운샘플** — T는 max-pooling 사용(얇은 tube 신호가 이중선형 보간 시 소실되는 문제 방지). 방향 필드는 이중선형 보간 후 재정규화.
2. **경계 마스킹** — 비정상적으로 높은 T를 가진 상하단 Z 슬라이스를 0으로 설정(`BORDER_ARTIFACT_RATIO`로 자동 감지). FMM의 표면 노이즈 단축 경로 방지.
3. **EDT 반경** — 전경 마스크(T > `EDT_THRESHOLD`)의 거리 변환 후 Gaussian 평활화, `EDT_RADIUS_SCALE` 적용. SWC 출력의 tube 반경으로 사용.

### 주요 출력
| 파일 | 내용 |
|------|------|
| `output/<sample>/prep_riem.npz` | `T_down`, `orient_down`, `edt_down`, `soma_mask_down`, `voxel_down` |

### 주요 파라미터
| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `DOWNSAMPLE` | 4 | 공간 다운샘플 배율 |
| `EDT_THRESHOLD` | 0.20 | EDT 계산용 전경 임계값 |
| `EDT_RADIUS_SCALE` | 0.7 | EDT 반경 전체 스케일 |
| `BORDER_ARTIFACT_RATIO` | 1.5 | 엣지/내부 비율 임계값 |

---

## Step 3 — Riemannian FMM 트레이싱 (`step3_auto.ipynb`)

**목표:** tubularity 필드를 통해 소마로부터의 측지 거리를 계산하고, 모든 수상돌기 경로를 소마까지 역추적합니다.

### 처리 순서

#### 1. Riemannian metric tensor
각 복셀에 방향 의존적 비용 구성:

```
cost(x)   = exp(−α · T(x))          # 낮은 T → 높은 비용
σ_∥(x)   = σ_⊥ · (1 − γ · T(x))   # tube 축 방향 비용 감소
M(x)      = cost² · [σ_⊥·I + (σ_∥ − σ_⊥)·v⊗v]
```

- tube 축 `v` 방향: 비용 `σ_∥` (tube 내부에서 낮음).
- 축 수직 방향: 비용 `σ_⊥` (높음; 측면 이동 억제).
- T=1(tube 중심)에서 비등방성 비율 ≈ 20:1.

#### 2. Fast Marching (FileHFM)
`agd.Eikonal` (Riemann3 모델)로 모든 소마 복셀에서 동시에 측지 거리를 계산합니다.

#### 3. Tip 검출
T_down에 `peak_local_max` 적용 (`min_distance = MIN_DIST_UM / voxel`). T > `MIN_T_TIP`이고 소마에서 도달 가능한 tip만 유지.

#### 4. Traceback
측지 거리 필드에서 이산 기울기 하강. 각 tip이 감소하는 측지 거리 방향으로 이동하여 소마에 도달. 원위부 세그먼트 중 평균 T < `MIN_MEAN_T`이거나 T < `MIN_SEG_T`인 구간은 트리밍.

#### 5. 중복 병합
공간적으로 가깝고(`MERGE_DIST_UM`) 방향이 유사한(`MERGE_DOT_MIN = 0.92`) 1차 가지를 병합 — tip 수가 많은 가지를 유지.

### 주요 출력
| 파일 | 내용 |
|------|------|
| `output/<sample>/neurons_auto.swc` | 표준 SWC 형태 파일 |

### 주요 파라미터 (자동 감지)
| 파라미터 | 설정 방식 | 의미 |
|---------|----------|------|
| `ALPHA` | `log(COST_TARGET_RATIO)` | 비용 대비 (tube vs 배경) |
| `MIN_T_TIP` | T에 대한 Otsu | tip으로 인정할 최소 T값 |
| `GAMMA` | 0.95 (고정) | tube 축 비등방성 강도 |
| `MIN_DIST_UM` | 25.0 (사용자 설정) | tip 간 최소 거리 |

---

## Step 4 — 시각화 및 형태 분석 (`step4_viz.ipynb`)

**목표:** 트레이싱 결과를 검사하고 형태 통계를 계산합니다.

### 시각화 목록
| 패널 | 내용 |
|------|------|
| **3D branch order 플롯** | Plotly scatter3d; 색상이 분기 차수를 나타냄 (1차=빨강 → 5차+=파랑) |
| **MIP 오버레이** | XY/XZ/YZ 최대 강도 투영 위에 SWC 엣지 표시 |
| **3D tube mesh** | EDT 반경으로 swept tube 기하 생성; 같은 1차 가지는 동일 색상 |
| **Dendrogram** | 경로 길이 vs 가지 인덱스; tip당 1행 |
| **가지 분포** | tip 거리, 분기 차수, primary당 tip 수 히스토그램 |
| **반경 컬러맵** | 국소 tube 반경으로 색상화된 MIP (빨강=얇음, 초록=두꺼움) |

### 계산되는 형태 지표
- 소마 반경, 부피
- 1차 가지 수, 분기점 수, tip 수
- 전체 수상돌기 길이 (소마 표면 보정)
- tip 경로 길이 분포 (평균, 표준편차, 중앙값, 최댓값)
- 반경 통계 (최솟값, 중앙값, 평균, p99, 최댓값)
- 최대 분기 차수

### Axon / Apical / Basal 분류
- **Axon**: 평균 반경이 가장 작은 1차 가지
- **Apical**: pia 방향으로 뻗는 가지(dir_z < −0.1) 중 최대 경로 길이
- **Basal**: 나머지 1차 가지

---

## 파일 입출력 요약

```
data/<sample>.tif                          ← 원시 입력

output/<sample>/
  stack_preprocessed.tif                  ← step0 출력
  preprocess_meta.npz
  tubularity.npz                           ← step1 출력
  soma.npz                                 ← step1b 출력
  soma.json
  tubularity_anchored.npz
  prep_riem.npz                            ← step2 출력
  neurons_auto.swc                         ← step3 출력 (최종 결과)
```
