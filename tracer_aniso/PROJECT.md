# Tracer Aniso — Riemannian Geodesic 기반 뉴런 트레이서

## 핵심 아이디어

OOF + Structure Tensor로 계산한 tubularity와 tube 방향 벡터를 이용해  
**Riemannian metric tensor** M(x)를 정의하고, anisotropic eikonal equation을 풀어  
soma에서 각 tip까지의 방향 의존적 geodesic 거리를 구한다.  
Traceback은 이 geodesic distance 필드 위에서 discrete gradient descent로 수행한다.

```
step1: OOF + ST Guided Weighting → T_combined, v_OOF (orient_field)
             ↓
step1b: Soma segmentation + EDT anchor → soma_mask, T_anchored
             ↓
step2_prep_aniso: 4× 다운샘플 → prep_riem.npz
             ↓
step3_aniso:
  M(x) 구성  ←  v_OOF, T_combined, GAMMA, ALPHA
       ↓
  agd Riemann3 FMM  (soma_mask → 전 볼륨)
       ↓
  geodesic_dist (Riemannian 거리 필드)
       ↓
  Tip detection  (T_down local maxima)
       ↓
  Discrete gradient descent traceback (tip → soma)
       ↓
  Tree 구성 + primary merge
       ↓
  neurons_riem.swc
```

---

## 파이프라인 구조

```
tracer_aniso/
├── step0_preprocess.ipynb       → (symlink, tracer_tda와 공유)
├── step1_tubularity_oof.ipynb   → (symlink, tracer_tda와 공유)
├── step1b_soma.ipynb            → (symlink, tracer_tda와 공유)
├── step2_prep_aniso.ipynb       → output/prep_riem.npz
├── step3_aniso.ipynb            → output/neurons_riem.swc
├── step4_viz.ipynb              → 시각화
├── bin/
│   ├── FileHFM_Isotropic3       (컴파일된 agd 바이너리)
│   └── FileHFM_Riemann3
└── output/                      (symlink 포함)
    ├── tubularity_anchored.npz  (← tracer_tda/output symlink)
    ├── soma.json / soma.npz     (← tracer_tda/output symlink)
    ├── prep_riem.npz            (step2 출력)
    └── neurons_riem.swc         (step3 출력)
```

---

## Step 1 — OOF + Structure Tensor Guided Weighting

### OOF (Orientation Optimized Filter)

구 표면 위 균등 분포 방향 `{nᵢ}` (Fibonacci sphere, N=300)에서  
각 방향의 중심 차분 기울기를 outer product로 누적하여 Q 행렬 구성:

```
Q(x) = (1/N) Σᵢ [∂I/∂nᵢ · (nᵢ ⊗ nᵢ)]
```

Q의 eigenvalue λ₁ ≤ λ₂ ≤ λ₃ 에서:
- **Tubularity response**: λ₁ < 0 이고 λ₂ < 0 일 때 `I_OOF = |λ₁ + λ₂|`
- **Tube axis**: `v_OOF` = λ₃의 eigenvector (가장 큰 eigenvalue 방향 = tube 축)

OOF는 구 표면 flux 기반으로 edge-dominant하여 tube 중심은 hollow할 수 있음.

### Structure Tensor

Gaussian 미분으로 gradient 벡터 계산 후 outer product smoothing:

```
J(x) = G_ρ * (∇I · ∇Iᵀ)   (G_ρ: integration scale Gaussian)
```

J의 최소 eigenvalue의 eigenvector `v_ST` = tube 축 방향 (OOF와 독립적 추정).  
Scale matching: `σ_d = radius / 2` (OOF와 같은 scale에서 방향 추정).

### Guided Weighting

두 방법의 방향 일치도 (cosine similarity):

```
C_align(x) = |⟨v_OOF(x), v_ST(x)⟩|  ∈ [0, 1]
```

Blob response (LoG) — tube 중심 및 soma 채움:

```
B(x) = max(0, -Tr(H))   H: Hessian of I at scale r
```

최종 tubularity (Enhancement formula — 억제가 아닌 강화):

```
T_raw(x) = I_OOF(x) + λ_blob · B(x)    (λ_blob = 0.3)
W(x)     = T_raw(x) · (1 + β · C_align(x))   (β = 1.0)
```

- C_align ≈ 1 (두 방향 일치, 실제 tube): W ≈ 2 · T_raw → **강화**  
- C_align ≈ 0 (불일치, 노이즈): W ≈ T_raw → **그대로** (0으로 억제 안 함)

저장 결과: `T_combined = W`, `orient_field = v_OOF` (float16)

**ST의 M(x) 기여**: ST는 M(x)에 직접 들어가지 않음.  
C_align → W → cost² → M(x) 의 간접 경로로만 영향. 즉 ST는  
"tubularity confidence"로만 작용하며 metric의 방향은 v_OOF만 결정함.

---

## Step 1b — Soma Segmentation + Anchor

**Soma detection**: `soma_score = intensity × (1 - T_combined)` — 밝고 tubularity 낮은 곳.  
Local Otsu threshold + morphological refinement (ball opening/closing/erosion).

**EDT Anchor**: soma 내부에서 T_combined을 overwrite:

```
T_anchored(x) = dist_EDT(x) / max(dist_EDT) × (T_max + 0.5)   (x ∈ soma_mask)
T_anchored(x) = T_combined(x)                                   (x ∉ soma_mask)
```

soma 중심이 global maximum → FMM seed로 자연스럽게 선택됨.

---

## Step 2 — 다운샘플 전처리

4× 다운샘플 (voxel_iso 0.342 µm → voxel_down 1.368 µm):

- `T_down`, `radius_down`: trilinear zoom
- `soma_mask_down`: nearest-neighbor zoom (bool 보존)
- `orient_down`: 채널별 trilinear zoom → 재정규화 (unit vector 보존)

```
v_norm = √(vz² + vy² + vx²) + ε
orient_down = [vz, vy, vx] / v_norm
```

---

## Step 3 — Riemannian FMM + Traceback

### Metric Tensor 구성

각 voxel x에서 T-adaptive anisotropic metric M(x):

```
σ_parallel(x) = σ_perp · (1 - γ · T(x))

M(x) = cost(x)² · [σ_perp · I + (σ_parallel(x) - σ_perp) · v(x)⊗v(x)]

where  cost(x) = exp(-α · T(x))
       v(x) = orient_down (v_OOF, ZYX 성분)
```

**물리적 의미**:

| 위치 | T(x) | σ_parallel | anisotropy ratio | cost |
|---|---|---|---|---|
| Background | 0 | σ_perp (=1.0) | 1:1 (isotropic) | 1.0 |
| Weak tube | 0.5 | σ_perp·(1-γ/2) | ~2:1 | exp(-α/2) |
| Strong tube | 1.0 | σ_perp·(1-γ) | 1/(1-γ):1 | exp(-α) |

현재 파라미터: α=8, γ=0.95 → strong tube에서 **ratio=20:1, cost≈0.0003**

**Eikonal equation** (agd Riemann3):

```
√(∇u(x)ᵀ · M(x) · ∇u(x)) = 1,    u(soma) = 0
```

- M(x) 작으면 (tube 축 방향): ∇u 크게 허용 → 빠르게 전파
- M(x) 크면 (수직 방향): ∇u 작아야 함 → 느리게 전파

### FMM 실행

Multi-source: `soma_mask` 내 모든 voxel을 seed로 동시 출발.

```python
hfm = Eikonal.dictIn({
    'model':     'Riemann3',
    'metric':    Metrics.Riemann(M),
    'seeds':     soma_seeds,   # (N,3) z,y,x
    'gridScale': 1.0,
})
geodesic_dist = hfm.Run()['values']
```

### Tip Detection

`T_down`의 local maxima:

```python
tips = peak_local_max(T_down,
    min_distance  = MIN_DIST_VOX,    # 20 vox (≈27 µm)
    threshold_abs = MIN_T_TIP,       # 0.40
)
```

T 높은 순 정렬, reachable (finite geodesic_dist) 필터.

### Discrete Gradient Descent Traceback

agd의 continuous geodesic 대신 discrete 26-neighbor steepest descent 사용.  
이유: continuous tracing은 공유 trunk에서 voxel 불일치 → trunk deduplication 실패.

```python
def traceback(tip, geodesic_dist, soma_mask):
    cur = tip
    path = []
    while not soma_mask[cur]:
        path.append(cur)
        # 26-neighbor 중 geodesic_dist 최소 방향으로 이동
        best = min(26-neighbors, key=lambda n: geodesic_dist[n])
        cur = best
    path.append(cur)       # soma voxel 포함
    return path[::-1]      # soma → tip 순서
```

같은 trunk를 지나는 두 path는 **항상 동일한 integer voxel 시퀀스**로 수렴  
→ `node_id_map` deduplication이 정확히 작동 → 올바른 tree topology.

### Tree 구성

- soma → tip 순서로 처리, 이미 방문한 voxel은 기존 node에 연결
- Primary branch merge: soma 인접 branch 중 거리 ≤ 15 µm + cosine ≥ 0.92인 것 병합

---

## 현재 파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `ALPHA` | 8.0 | cost = exp(-α·T); 높을수록 tube/background 대비 강화 |
| `SIGMA_PERP` | 1.0 | 수직 방향 metric 기준값 |
| `GAMMA` | 0.95 | anisotropy 강도; T=1에서 ratio = 1/(1-γ) = 20:1 |
| `MIN_DIST_VOX` | 20 | tip 간 최소 거리 (≈27 µm) |
| `MIN_T_TIP` | 0.40 | tip 최소 T 값 (노이즈 제거) |
| `MAX_TIPS` | 300 | 안전망 상한 |
| `MIN_PATH_LEN_UM` | 5.0 µm | stub pruning 기준 |
| `DOWNSAMPLE` | 4 | voxel_down = 1.368 µm |

---

## tracer_tda (iso) 와의 비교

| 항목 | tracer_tda (iso) | tracer_aniso (Riemannian) |
|---|---|---|
| FMM cost | `exp(-α·T)` scalar | `M(x)` 3×3 tensor |
| 방향성 | 없음 | v_OOF 방향 선호 |
| Tubularity | OOF + blob | OOF + blob + ST confidence |
| Traceback | MCP_Geometric (discrete) | agd Riemann3 → discrete gradient descent |
| 논문 claim | isotropic geodesic | **Riemannian geodesic** |
| FMM 시간 | ~12s | ~37s |
| 출력 | neurons_iso.swc | neurons_riem.swc |

---

## 설계 선택 이유

| 결정 | 이유 |
|---|---|
| agd continuous geodesic 버리고 discrete traceback 사용 | continuous path가 공유 trunk에서 voxel 불일치 → trunk dedup 실패 |
| σ_parallel = σ_perp·(1-γ·T) adaptive | T=0 background는 isotropic으로 두어 잘못된 방향 강제 방지 |
| ST를 M(x)에 직접 넣지 않음 | v_OOF와 v_ST blend 시 추가 파라미터 필요; C_align → W 경로로 충분 |
| soma_mask 전체 voxel을 FMM seed로 | soma 경계 노이즈 영향 제거; wavefront가 경계에서 바로 출발 |
| EDT anchor (soma 내부 T 덮어쓰기) | OOF의 ring artifact 제거; soma가 FMM에서 자연스럽게 최고값 |

---

## 참고문헌

- Mirebeau, J.M. (2014) "Anisotropic fast-marching on cartesian grids using Voronoi's first reduction" — agd 라이브러리 수학적 기반
- Benmansour & Cohen (2011) "Tubular Structure Segmentation Based on Minimal Path Method and Anisotropic Enhancement"
- Sethian & Vladimirsky (2003) "Ordered Upwind Methods for Static Hamilton-Jacobi Equations"
- AGD 라이브러리: https://github.com/Mirebeau/AdaptiveGridDiscretizations
- FileHFM 바이너리: `/Users/lee/Tracer/tracer_aniso/bin/`
