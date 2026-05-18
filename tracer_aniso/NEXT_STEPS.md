# tracer_aniso — Next Steps & Performance Analysis

---

## 1. 다음 수정 우선순위 (Phase B)

### 1순위 — step3: soma_mask_down 빈 경우 조기 중단
**위험도: HIGH** | 파일: `step3_auto.ipynb` cell `load`

다운샘플(×4) 후 소마 마스크가 0 voxel로 축소되면 FMM seed가 없어 `geodesic_dist` 전체가 `inf` → `tip_coords_s` 전부 필터링 → SWC 빈 파일 저장. **무음 실패**.

```python
# load 셀 하단에 추가
assert soma_mask_down.sum() > 0, (
    f"soma_mask_down is empty after {DOWNSAMPLE}× downsampling! "
    f"Check DOWNSAMPLE ({DOWNSAMPLE}) or soma segmentation in step1b."
)
print(f'soma_mask_down: {soma_mask_down.sum():,} voxels  OK')
```

---

### 2순위 — step2: orient_field NaN/Inf 검사
**위험도: MEDIUM** | 파일: `step2_prep_aniso.ipynb` cell `load`

등방성 영역(배경)에서 norm=0 → NaN이 Riemannian metric tensor M(x) 전체를 오염시킴. FMM이 잘못된 geodesic 거리를 계산해도 에러 없이 진행됨.

```python
# orient_field 정규화 직후 추가
nan_count = (~np.isfinite(orient_down.astype(np.float32))).sum()
if nan_count > 0:
    print(f'WARNING: {nan_count} NaN/Inf in orient_down — replacing with 0')
    orient_down = np.where(np.isfinite(orient_down), orient_down, np.float16(0))
else:
    print('orient_down: no NaN/Inf  OK')
```

---

### 3순위 — step0: VOXEL_Z 유효성 검사
**위험도: HIGH** | 파일: `step0_preprocess.ipynb` cell `load`

`VOXEL_Z`는 유일하게 사용자가 수동 입력하는 물리 파라미터. 2배 오류(예: 0.5 대신 1.0) 시 Z 리스케일이 반전되어 이후 전 단계 결과가 왜곡됨. 경고 없음.

```python
# VOXEL_Z 사용 전 추가
assert 0.1 <= VOXEL_Z <= 10.0, f"VOXEL_Z={VOXEL_Z} out of plausible range [0.1, 10.0] µm"
if VOXEL_Z < 0.5:
    print(f'WARNING: VOXEL_Z={VOXEL_Z} µm very small — confirm slice thickness')
if VOXEL_Z > 3.0:
    print(f'WARNING: VOXEL_Z={VOXEL_Z} µm large — high anisotropy, Z rescaling will expand volume significantly')
```

---

### 4순위 — step3: MIN_TORTUOSITY를 Config로 이동
**위험도: LOW** | 파일: `step3_auto.ipynb`

현재 `MIN_TORTUOSITY = 1.02`가 config 셀에 있지만 자동 감지 셀에서는 재사용되지 않음. 층판 내 수직/수평 수상돌기(직선에 가까운 경로)가 이 조건에 걸려 소실될 수 있음.

```python
# config 셀 — 이미 있음, 값만 조정 권장
MIN_TORTUOSITY = 1.0   # 1.02 → 1.0: 직선형 수상돌기 보존
```

---

## 2. OOF 성능 분석 — 병렬화 시나리오별 메모리 위험도

### 현재 상황
- **실측 시간**: 7,006초 (~117분)
- **알고리즘**: 8 radii × 150 directions × 23 slabs, GPU(MPS) 직렬
- **현재 GPU 피크 메모리 (per slab)**: ~6.0 GB

| 구성요소 | 크기 |
|---------|------|
| slab_t (66z × 996y × 990x) | 0.33 GB |
| Q accumulator (6채널) | 1.99 GB |
| grid_fwd/bwd cache | 1.99 GB |
| Ip/Im temporaries | 0.66 GB |
| v_ST field | 0.99 GB |
| **합계** | **5.96 GB** |

---

### 시나리오 1 — N_RADII 병렬 (8 GPU workers)
**위험도: 🔴 CRITICAL**

각 radius worker가 독립 Q(1.99 GB) + grid cache(1.99 GB) 필요.

```
peak = slab_t(0.33) + 8 × (Q + grids)(3.98) = 32.1 GB
```

**M-series Mac (통합 메모리 36GB 기준) 89% 점유 → OOM 또는 극심한 메모리 스왑**

MPS는 CPU와 메모리를 공유하므로 OS + Python + 기타 프로세스까지 합산하면 사실상 불가능.

---

### 시나리오 2 — Slab 병렬 (N CPU workers, joblib)
**위험도: 🟡 MEDIUM**

GPU를 포기하고 CPU에서 여러 slab을 동시 처리.

```
per worker = slab(0.33) + Q(1.99) + Ip/Im(0.66) = 2.98 GB
4 workers  = 11.9 GB RAM
```

- 32GB 시스템에서 가능
- **단점**: CPU OOF는 GPU 대비 10~20× 느려 총 시간이 오히려 증가할 수 있음
- **결론**: 권장하지 않음

---

### 시나리오 3 — Direction 배치 (k방향 묶음)
**위험도: 🔴 HIGH (k=16)**

k방향을 한 번에 처리하면 grid가 (k, Z, Y, X, 3)으로 확장됨.

```
k=16: batch_grid = 16 × grid_cache = 31.8 GB  ← 사실상 불가능
k=4:  batch_grid = 4  × grid_cache = 8.0  GB  ← 간신히 가능
```

루프 오버헤드(Python → C++ 150회 호출) 제거 효과는 크지 않음 (~10–20% 개선).  
**결론**: 이득 대비 위험이 너무 큼.

---

### 시나리오 4 — N_SPHERE_PTS 감소 ✅ 권장
**위험도: 🟢 ZERO (메모리 변화 없음)**

Fibonacci sphere 샘플 수를 줄이면 방향당 GPU 연산이 정비례 감소.

| N_SPHERE_PTS | 이론 가속 | 예상 소요 시간 | 정확도 영향 |
|-------------|----------|--------------|-----------|
| 150 (현재) | 1.0× | ~117분 | baseline |
| 100 | 1.5× | ~78분 | 미미 |
| **75** | **2.0×** | **~58분** | 허용 수준 |
| 50 | 3.0× | ~39분 | 약간 저하 |

Fibonacci sphere는 75 pts에서도 균등 커버리지 유지. 얇은 구조 감지 정확도에 미치는 영향 최소.

**권장: `N_SPHERE_PTS = 75` → 약 2× 가속, 무위험**

---

### 시나리오 5 — INPUT_DOWNSAMPLE = 2
**위험도: 🟡 MEDIUM (얇은 구조 소실 가능)**

`INPUT_DOWNSAMPLE = 2`로 설정하면 입력 해상도 절반 → **이론 8× 가속** (3D 볼륨 감소 × 방향 계산 감소).

```
현재:  1064 × 996 × 990 × float32 = 4.2 GB
2× ds:  532 × 498 × 495 × float32 = 0.5 GB
예상 시간: ~14분
```

**단점**: 0.342 µm → 0.684 µm/vox. 직경 < 1.4 µm (≈ 2 voxel) 이하 얇은 axon 소실 가능.  
**권장**: 검증용 빠른 실행 또는 두꺼운 수상돌기 위주 데이터에만 사용.

---

## 3. 권장 액션 플랜

```
즉시 (안전) ──────────────────────────────────────────────────
  fix2: step3 soma_mask_down empty assert
  fix3: step2 orient_field NaN replace
  fix4: step0 VOXEL_Z range assert
  perf: N_SPHERE_PTS 150 → 75  (2× 가속, 무위험)

신중하게 ──────────────────────────────────────────────────────
  perf: INPUT_DOWNSAMPLE = 2  (빠른 검증 실행용, 별도 설정으로)
  fix5: MIN_TORTUOSITY 1.02 → 1.0

장기 ──────────────────────────────────────────────────────────
  step1 Cardano eigensolver norm clamp (np.maximum)
  step1b hollow soma 실패 시 진단 출력 강화
  step3 ALPHA 데이터 적응형 계산
```

## 4. 파일 현황

| 폴더 | 내용 |
|------|------|
| `./` | 원본 (수정 금지) |
| `fix1/` | Phase A 수정: step3 half-index 버그 수정 |
| `Final/` | 영어 번역본 6개 노트북 |
| `WORKFLOW_EN.md` | 영어 워크플로우 문서 |
| `WORKFLOW_KO.md` | 한국어 워크플로우 문서 |
| `NEXT_STEPS.md` | 이 파일 |
