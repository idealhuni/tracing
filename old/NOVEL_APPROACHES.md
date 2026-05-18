# Novel Neuronal Tracing Approaches
## Beyond Rivuletpy / APP2 / Skeletonization

현재 파이프라인(orient_field 스트림라인 + GWDT burn)을 기반으로,  
기존 방법과 차별화되는 4가지 방향을 제안합니다.

---

## 현재 방식의 한계

| 한계 | 영향 |
|------|------|
| Greedy tip-by-tip 추적 | 앞 결정이 뒤 결과에 영향 → 전파 오류 |
| 결정론적 orient_field 추적 | 노이즈 영역에서 경로 이탈 |
| FMM (isotropic) | 1B voxel에서 2시간+ → 실용 불가 |
| 불확실도 없음 | 어느 가지가 신뢰할 수 있는지 알 수 없음 |

---

## Approach 1: Probabilistic Streamline Tracing ⭐ (먼저 구현)

### 핵심 아이디어

각 추적 step에서 방향을 **확률 분포로 모델링**.  
T값이 낮은 곳(불확실) = 넓은 분포, T값 높은 곳(확실) = 좁은 분포.

### 수학적 모델

각 voxel x에서 방향 d의 확률:

```
P(d | x) ~ von Mises-Fisher(μ=orient_field[x], κ=T[x] · κ_max)
```

- `μ`: orient_field의 단위 벡터 (평균 방향)
- `κ`: concentration parameter — T값에 비례
  - T=1.0 → κ=κ_max (매우 좁은 분포, 확실)
  - T=0.1 → κ=0.1·κ_max (넓은 분포, 불확실)
- von Mises-Fisher = 구면 위의 Gaussian

### 알고리즘

```
for each tip:
    for s in range(N_SAMPLES):          # N_SAMPLES개 경로 샘플링
        path_s = []
        pos = tip_position
        while not terminated:
            μ = orient_field[pos]
            κ = T[pos] * κ_max
            d = sample_vMF(μ, κ)        # 방향 샘플링
            pos = pos + d * step_size
            path_s.append(pos)
    
    # 샘플들의 평균 경로 = 최종 경로
    mean_path = average(all_samples)
    
    # 분산 = 불확실도 맵
    uncertainty = variance(all_samples)
```

### 출력물 (기존 대비 추가)

- `neurons.swc`: 기존과 동일 (평균 경로)
- **`neurons_uncertainty.nrrd`**: 각 voxel의 경로 불확실도 (0~1)
- **`neurons_samples.swc`**: 샘플 경로들 (confidence band 시각화)

### 논문 차별점

- DTI tractography에서는 표준 방법이나 **형광 현미경 뉴런 트레이싱에 적용된 논문 없음**
- Uncertainty map → 생물학적 해석 가능 (어느 가지가 real vs. artifact?)
- 기존 SWC에 confidence score 추가 → 새 표준 포맷 제안 가능

### 구현 복잡도: ★★☆☆☆

현재 `trace_streamline()` 함수를 N번 반복 + 샘플링 추가.  
새로운 외부 라이브러리 불필요.

---

## Approach 2: Riemannian Anisotropic Fast Marching

### 핵심 아이디어

isotropic FMM 대신 **orient_field로 정의된 anisotropic metric**으로 geodesic 계산.

### 수학적 모델

각 voxel x에서 metric tensor:

```
G(x) = v(x)v(x)^T + λ·(I - v(x)v(x)^T)

여기서:
  v(x) = orient_field[x]   (tube 방향 단위벡터)
  λ ≪ 1                    (수직 방향 패널티, 예: 0.01)
```

- tube 방향: 비용 1
- tube 수직 방향: 비용 1/√λ >> 1

Eikonal equation: `√(∇u · G(x)⁻¹ · ∇u) = 1`

### 장점 vs. isotropic FMM

| | isotropic FMM | Anisotropic FMM |
|---|---|---|
| 속도 | O(N log N), 느림 | 더 느리지만 결과 우월 |
| crossing 분리 | 약함 | 강함 (tube 방향 따라 분리) |
| tip 감지 | GWDT 극댓값 | anisotropic depth 극댓값 |

### 구현 참조

- Mirebeau (2014): "Anisotropic fast-marching on cartesian grids"
- Python: `AGD` 라이브러리 또는 직접 구현

### 구현 복잡도: ★★★★☆

anisotropic FMM은 구현 복잡도 높음. 기존 라이브러리 없으면 직접 구현 필요.

---

## Approach 3: Global Energy Minimization

### 핵심 아이디어

Greedy 대신 **모든 경로를 동시에 최적화**.

### 에너지 함수

```
E(Paths) = Σ_i E_data(path_i) + α·E_smooth(path_i) + β·E_coverage + γ·E_exclusion

E_data     = Σ_{x∈path} (1 - T(x))          # 경로가 tube 위에 있어야
E_smooth   = Σ_{x∈path} ||d_t - d_{t-1}||²  # 방향 변화 최소화
E_coverage = -|∪ paths ∩ ridges|             # 최대한 많은 foreground 커버
E_exclusion = |∩ paths_i ∩ paths_j|         # 경로 중복 방지
```

### 최적화 방법

- **Graph cuts** (Boykov & Kolmogorov): binary 변수 (각 voxel이 경로에 포함되는가)
- **Belief propagation**: 더 부드러운 최적화
- **Simulated annealing**: 구현 쉬움, 느림

### 논문 차별점

- 전역 최적해 → greedy artifact 없음
- 순서 의존성 없음 (어떤 tip 먼저 추적하느냐에 무관)

### 구현 복잡도: ★★★★★

에너지 정의 + 최적화 구현 모두 복잡. 연구 수준 구현 필요.

---

## Approach 4: Topological Data Analysis (Persistent Homology)

### 핵심 아이디어

T_combined를 scalar field로 보고 **Morse theory로 topology 추출**.

### Morse theory 적용

```
T_combined scalar field에서:

Critical points:
  - maximum    → tube tip (가지 끝)
  - 1-saddle   → tube junction (분기점)
  - minimum    → background

Gradient flow:
  - maximum → saddle → minimum 연결 = 가지 구조
```

Persistence diagram:
- (birth, death) pair → persistence = birth - death
- 낮은 persistence = 노이즈 → 임계값으로 필터링

### 출력

- **Reeb graph**: T_combined의 위상학적 skeleton
- **Persistence diagram**: 각 가지의 신뢰도
- SWC 변환 가능

### 논문 차별점

- 위상학적으로 보장된 tree 구조
- 노이즈 필터링이 수학적으로 엄밀
- 새로운 시각화 (persistence barcode)

### 구현 참조

- `gudhi` 라이브러리 (Python)
- `scikit-tda` 

### 구현 복잡도: ★★★★☆

gudhi로 기본 구현은 가능. 3D volume Morse theory는 추가 작업 필요.

---

## 구현 로드맵

### Phase 1: Probabilistic Tracing (현재)
- `step3_prob.ipynb` 신규 작성
- 기존 `trace_streamline()` 확장
- N_SAMPLES=10~50으로 uncertainty map 생성
- 검증: 같은 데이터에서 step3_soma_signal vs step3_prob 비교

### Phase 2: Riemannian FMM
- step2를 anisotropic FMM으로 교체
- orient_field를 metric tensor로 변환
- tip 감지 개선 확인

### Phase 3: 정량 평가
- BigNeuron 공개 데이터셋으로 기존 방법과 비교
- Diadem metric (spatial distance, topological score)
- 논문 작성

---

## 파일 구조 (예정)

```
rivuletpy/
├── step1.ipynb              ← Hessian tubularity (현재)
├── step2.ipynb              ← GWDT (현재)
├── step3_soma_signal.ipynb  ← 결정론적 트레이싱 (현재)
├── step3_prob.ipynb         ← Probabilistic tracing (Phase 1)
├── step3_riemannian.ipynb   ← Riemannian geodesic (Phase 2)
└── eval/
    ├── diadem_metric.py
    └── compare_methods.ipynb
```
