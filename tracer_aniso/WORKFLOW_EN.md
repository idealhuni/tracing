# Tracer Aniso — Neuron Tracing Workflow

---

## Overview

A 6-step pipeline that takes a raw 3D fluorescence TIFF of a neuron and produces a fully reconstructed SWC morphology file, ready for morphological analysis and visualization.

```
Raw TIFF
  ↓ step0  — Preprocessing
  ↓ step1  — Tubularity Map (OOF + Structure Tensor)
  ↓ step1b — Soma Detection & Anchoring
  ↓ step2  — Downsampling & Metric Preparation
  ↓ step3  — Riemannian FMM Tracing → SWC
  ↓ step4  — Visualization & Morphology Analysis
```

---

## Step 0 — Preprocessing (`step0_preprocess.ipynb`)

**Goal:** Convert a raw TIFF into a normalized, isotropic float32 volume.

### Pipeline
1. **Load** — Read multi-slice TIFF; auto-detect XY voxel size from TIFF tags.
2. **Downsample** — `DOWNSAMPLE_XY` is auto-computed so that the output voxel size stays ≤ `TARGET_VOXEL_XY_UM` (default 0.40 µm) and the XY dimension stays ≤ `MAX_XY_PX`.
3. **Normalize** — Lower bound: `threshold_triangle` (auto background/signal split). Upper bound: 99.9th percentile (hot-pixel removal). Output: float32 [0, 1].
4. **Z Rescaling** — `scipy.ndimage.zoom` along Z to produce isotropic voxels (`zoom_factor = voxel_Z / voxel_XY`). Skipped if anisotropy < 1.2.
5. **Noise2Void (optional)** — Self-supervised 3D denoiser. Requires no clean reference image. Set `N2V_ENABLE = True` to activate.

### Key outputs
| File | Content |
|------|---------|
| `output/<sample>/stack_preprocessed.tif` | Normalized, isotropic float32 volume |
| `output/<sample>/preprocess_meta.npz` | `voxel_iso`, clip bounds, anisotropy |

### Key parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TARGET_VOXEL_XY_UM` | 0.40 | Target XY voxel size after downsampling (µm) |
| `MAX_XY_PX` | 1024 | Maximum XY pixel dimension |
| `CLIP_HIGH_PERCENTILE` | 99.9 | Upper clip bound (99.99 for sparse signals) |
| `N2V_ENABLE` | False | Enable Noise2Void denoising |

---

## Step 1 — Tubularity Map (`step1_tubularity_oof_v2.ipynb`)

**Goal:** Produce a scalar tubularity map W(x) ∈ [0, 1] that is high inside tube-like structures (axons, dendrites) and low elsewhere.

### Method: Guided Weighting

Two independent detectors estimate the same tube axis; their agreement amplifies the response while noise/artifacts are suppressed.

| Detector | Principle | Output |
|----------|-----------|--------|
| **OOF** (Optimally Oriented Flux) | Outward normal flux across a sphere surface; both transverse eigenvalues negative → tubular | `I_OOF(x)`, `v_OOF(x)` |
| **Structure Tensor** | Gradient outer product smoothed at integration scale; smallest eigenvalue direction = tube axis | `v_ST(x)` |

**Guided weighting formula:**
```
C_align(x) = |⟨v_OOF, v_ST⟩|          # cosine similarity, 0–1
W(x)       = I_OOF(x) · (1 + β · C_align(x))
```

A **LoG blob response** (scaled by `LAMBDA_BLOB`) is added to fill tube interiors and soma shells that OOF misses.

### Multi-scale
Radii are sampled log-uniformly from `TUBE_RADIUS_MIN_UM` to `TUBE_RADIUS_MAX_UM` (`N_RADII` scales). The max-response scale is kept per voxel. GPU (MPS/CUDA) acceleration is used when available.

### Key outputs
| File | Content |
|------|---------|
| `output/<sample>/tubularity.npz` | `T_combined`, `I_OOF_raw`, `orient_field`, `radius_map` |

### Key parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TUBE_RADIUS_MIN_UM` | 0.10 | Minimum tube radius (µm) |
| `TUBE_RADIUS_MAX_UM` | 2.0 | Maximum tube radius (µm) |
| `N_RADII` | 8 | Number of scales |
| `BETA` | 1.0 | Alignment enhancement strength |
| `LAMBDA_BLOB` | 0.3 | LoG blob weight |

---

## Step 1b — Soma Detection (`step1b_soma.ipynb`)

**Goal:** Segment the soma, then re-anchor the tubularity map so that the soma becomes the global maximum — making it the true lowest-cost origin for the FMM.

### Pipeline
1. **Soma localization** — Score = intensity × (1 − T_combined). The peak of a Gaussian-smoothed version of this score is the soma candidate. Border voxels are excluded to avoid edge artifacts.
2. **Soma segmentation** — Local Otsu threshold within a sphere of radius `SOMA_SEARCH_RADIUS_UM`. Hollow soma: iterative morphological closing + `fill_holes`. Morphological opening/closing/erosion refine the mask.
3. **Soma anchoring** — Inside the soma, replace T with a distance-transform gradient (EDT) that peaks at the soma center. Scale all T values so the soma center is the global max (T=1.0), compressing tube values to ≤ ~0.67.

> **Why anchor?** The FMM paths must originate from the soma. Without anchoring, saturated dendrite voxels (T=1.0) can "compete" with the soma as a path origin, producing topologically incorrect trees.

### Key outputs
| File | Content |
|------|---------|
| `output/<sample>/soma.npz` | `soma_mask`, `soma_centroid_vox`, mesh vertices/faces |
| `output/<sample>/soma.json` | `centroid_vox`, `radius_um` |
| `output/<sample>/tubularity_anchored.npz` | T_combined with soma as global max |

---

## Step 2 — Metric Preparation (`step2_prep_aniso.ipynb`)

**Goal:** Downsample all fields to a manageable size and compute the EDT-based radius map used in the Riemannian metric.

### Pipeline
1. **Downsample** — T uses max-pooling (preserves thin tube signals that would vanish with bilinear interpolation). Orientation field is bilinearly downsampled and renormalized.
2. **Border masking** — Top/bottom Z slices with anomalously high T are zeroed out (auto-detected via `BORDER_ARTIFACT_RATIO`). Prevents surface-noise shortcuts in FMM.
3. **EDT radius** — Distance transform of the foreground mask (T > `EDT_THRESHOLD`), Gaussian-smoothed and scaled by `EDT_RADIUS_SCALE`. Used as the tube radius in the SWC output.

### Key outputs
| File | Content |
|------|---------|
| `output/<sample>/prep_riem.npz` | `T_down`, `orient_down`, `edt_down`, `soma_mask_down`, `voxel_down` |

### Key parameters
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `DOWNSAMPLE` | 4 | Spatial downsampling factor |
| `EDT_THRESHOLD` | 0.20 | Foreground threshold for EDT computation |
| `EDT_RADIUS_SCALE` | 0.7 | Global scale applied to EDT radius |
| `BORDER_ARTIFACT_RATIO` | 1.5 | Edge/interior ratio above which slices are masked |

---

## Step 3 — Riemannian FMM Tracing (`step3_auto.ipynb`)

**Goal:** Compute geodesic distances from the soma through the tubularity field, then trace all dendrite paths back to the soma.

### Pipeline

#### 1. Riemannian metric tensor
A direction-dependent cost is constructed at each voxel:

```
cost(x)   = exp(−α · T(x))          # low T → high cost
σ_∥(x)   = σ_⊥ · (1 − γ · T(x))   # reduced cost along tube axis
M(x)      = cost² · [σ_⊥·I + (σ_∥ − σ_⊥)·v⊗v]
```

- Along the tube axis `v`: cost is `σ_∥` (low inside tubes).
- Perpendicular to axis: cost is `σ_⊥` (high; discourages lateral movement).
- At T=1 (tube center), the anisotropy ratio is `σ_⊥/σ_∥ = 1/(1−γ)` ≈ 20:1.

#### 2. Fast Marching (FileHFM)
`agd.Eikonal` (Riemannian3 model) computes geodesic distance from all soma voxels simultaneously.

#### 3. Tip detection
`peak_local_max` on T_down with `min_distance = MIN_DIST_UM / voxel`. Only tips with T > `MIN_T_TIP` (auto-set to Otsu threshold on T) and reachable from the soma are kept.

#### 4. Traceback
Discrete gradient descent on the geodesic distance field. Each tip walks toward decreasing geodesic distance until it hits the soma. Distal path segments with mean T < `MIN_MEAN_T` or T < `MIN_SEG_T` are trimmed.

#### 5. Duplicate merging
Primary branches that are close in space (`MERGE_DIST_UM`) and highly co-linear (`MERGE_DOT_MIN = 0.92`) are merged, keeping the branch with more tips.

### Key outputs
| File | Content |
|------|---------|
| `output/<sample>/neurons_auto.swc` | Standard SWC morphology file |

### Key parameters (auto-detected)
| Parameter | How set | Meaning |
|-----------|---------|---------|
| `ALPHA` | `log(COST_TARGET_RATIO)` | Cost contrast (tube vs background) |
| `MIN_T_TIP` | Otsu on T | Minimum T for a peak to be a tip |
| `GAMMA` | 0.95 (fixed) | Tube-axis anisotropy strength |
| `MIN_DIST_UM` | 25.0 (user) | Minimum inter-tip distance |

---

## Step 4 — Visualization & Morphology (`step4_viz.ipynb`)

**Goal:** Inspect the tracing result and compute morphological statistics.

### Visualizations
| Panel | What it shows |
|-------|--------------|
| **3D branch-order plot** | Plotly scatter3d; color encodes branch order (1st=red → 5th+=blue) |
| **MIP overlay** | XY/XZ/YZ max-intensity projections with SWC edges overlaid |
| **3D tube mesh** | Swept tube geometry (radius from EDT); same-primary branches share color |
| **Dendrogram** | Path-length vs branch index; one row per tip |
| **Branch distributions** | Tip distance, branch order, tips-per-primary histograms |
| **Radius colormap** | MIP colored by local tube radius (red=thin, green=thick) |

### Morphology metrics computed
- Soma radius, volume
- Primary branch count, branch points, tip count
- Total dendritic length (soma-surface corrected)
- Tip path-length distribution (mean, std, median, max)
- Radius statistics (min, median, mean, p99, max)
- Max branch order

### Axon / Apical / Basal classification
- **Axon**: primary branch with the smallest mean radius
- **Apical**: among branches pointing pia-ward (dir_z < −0.1), the one with the longest path
- **Basal**: all remaining primary branches

---

## File I/O Summary

```
data/<sample>.tif                          ← raw input

output/<sample>/
  stack_preprocessed.tif                  ← step0 output
  preprocess_meta.npz
  tubularity.npz                           ← step1 output
  soma.npz                                 ← step1b output
  soma.json
  tubularity_anchored.npz
  prep_riem.npz                            ← step2 output
  neurons_auto.swc                         ← step3 output (final result)
```
