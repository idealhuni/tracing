# Pipeline Improvement Ideas

## step1b_soma — Soma Detection

### [1] Intensity blur-first 방식으로 변경
**문제:** 현재 `soma_score = stack_norm * (1 - T_combined)` 후 Gaussian blur.  
hollow soma (빈 내부)에서 중심이 어두워 score가 낮게 나옴.

**개선:**
```python
stack_blurred = gaussian_filter(stack_norm, sigma=SOMA_SIGMA)  # 먼저 blur
soma_score    = stack_blurred * (1 - T_combined)               # 그 다음 억제
```
- hollow soma: ring을 soma 크기로 blur → 중심도 밝아짐
- filled soma: 중심이 원래 밝으니 동일하게 작동

### [2] SOMA_MAX_TUBULARITY → 소프트 페널티로 교체
**문제:** 하드 cutoff라서 내부가 라벨된 데이터에서 soma 전체가 candidate 제외됨.

**개선:**
```python
SOMA_TUBULARITY_PENALTY = 1.0  # 0=tubularity 무시, 1=현재 동작, 0.5=절충
soma_score = stack_blurred * (1 - T_combined * SOMA_TUBULARITY_PENALTY)
```

---

## step3_aniso — Path Length / Tortuosity

### [3] Path length 과소평가 버그 수정
**문제:** `traceback_discrete`가 26-connectivity (대각선 포함)로 이동하는데  
`len(path) * voxel_down`으로 계산하면 대각선 step을 1 voxel로 처리함.

| 이동 방향 | 실제 거리 | 현재 계산 |
|---|---|---|
| 축 방향 (±1,0,0) | 1.00 × voxel_down | 1.0 × voxel_down ✓ |
| 면 대각선 (±1,±1,0) | 1.41 × voxel_down | 1.0 × voxel_down ✗ |
| 공간 대각선 (±1,±1,±1) | 1.73 × voxel_down | 1.0 × voxel_down ✗ |

평균 20–40% 과소평가 → tortuosity도 함께 틀림.

**개선:**
```python
path_arr    = np.array(path, dtype=np.float32)
diffs       = np.diff(path_arr, axis=0)
path_len_um = float(np.linalg.norm(diffs, axis=1).sum()) * voxel_down
```

### [4] 외곽 노이즈 가지 과다 감지
**문제:** 중심 구조는 잘 잡히나 외곽 노이즈가 가지로 많이 감지됨.  
원인: 노이즈 tip이 tip detection에서 걸러지지 않고, 낮은 T 경로도 허용됨.

**튜닝 순서 (config에서 조정):**

| 파라미터 | 현재값 | 방향 | 효과 |
|---|---|---|---|
| `MIN_T_TIP` | 0.50 | ↑ 0.60~0.70 | 약한 tubularity tip 제거 |
| `MIN_MEAN_T` | 0.10 | ↑ 0.15~0.25 | 낮은 T 경로 전체 제거 |
| `MIN_PATH_LEN_UM` | 5.0 | ↑ 10~15 | 짧은 노이즈 가지 제거 |
| `ALPHA` | 16.0 | ↑ 24~32 | 저T 구간 통과 비용 증가 |

**추가 개선 — tip radius 필터 (미구현):**  
외곽 노이즈는 EDT radius가 작음 → tip 위치의 `edt_down`으로 최소 반경 필터링.
```python
MIN_TIP_RADIUS_UM = 0.3   # tip 위치 EDT 반경이 이 값 미만이면 제거
tip_edt = edt_down[tip_coords_s[:,0], tip_coords_s[:,1], tip_coords_s[:,2]]
tip_coords_s = tip_coords_s[tip_edt >= MIN_TIP_RADIUS_UM]
```
