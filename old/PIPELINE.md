# Neuronal Tracer Pipeline — /Users/lee/Tracer/rivuletpy/

## Overview

Skeleton-free volume-based Rivuletpy tracer using orient_field streamlines + GWDT burn.  
No Lee thinning. No binarization. Sub-voxel accuracy throughout.

---

## File Structure

```
rivuletpy/
├── step1.ipynb          ← Multi-scale Hessian → tubularity + orient_field
├── step2.ipynb          ← FMM → GWDT depth map + ridges foreground mask
├── step3.ipynb          ← Streamline tracing → neuron tree SWC
└── output/
    ├── stack_iso.tif        (isotropic resampled stack)
    ├── T_combined.tif       (tubularity visualization)
    ├── tubularity.npz       (step1 output)
    ├── gwdt.npz             (step2 output)
    └── neurons.swc          (step3 output — final result)
```

---

## Pipeline Dependency

```
step1.ipynb  →  tubularity.npz  (T_combined, orient_field, radius_map, voxel_iso)
     ↓
step2.ipynb  →  gwdt.npz        (gwdt_arr, ridges, radius_map, voxel_iso)
     ↓
step3.ipynb  →  neurons.swc     (neuron tree in physical µm coordinates)
```

---

## Step 1 — Tubularity (step1.ipynb)

**Input**: raw image stack (TIF)  
**Output**: `output/tubularity.npz`

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| T_combined | Z×Y×X | float32 | Tubularity ∈ [0,1] (Frangi + Meijering combined) |
| orient_field | Z×Y×X×3 | float16 | Tube-axis eigenvector v₁ at each voxel |
| radius_map | Z×Y×X | float32 | Estimated tube radius (µm) from Hessian scale |
| scale_idx_m | Z×Y×X | int8 | Index of best scale |
| sigmas | 1-D | float32 | Gaussian scales used |
| voxel_iso | scalar | float32 | µm/voxel after isotropic resampling |

Also writes `stack_iso.tif` and `T_combined.tif`.

---

## Step 2 — GWDT (step2.ipynb)

**Input**: `output/tubularity.npz`  
**Output**: `output/gwdt.npz`

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| gwdt_arr | Z×Y×X | float32 | GWDT depth (FMM arrival time from background seeds) — tube centres have highest value |
| ridges | Z×Y×X | bool | Foreground mask (thresholded T_combined) |
| radius_map | Z×Y×X | float32 | Passed through from step1 |
| voxel_iso | scalar | float32 | Passed through from step1 |
| orient_field | Z×Y×X×3 | float32 | Optional — also passed through if present |

**Note**: Lee thinning / skeleton step was removed. Step2 does GWDT only.

---

## Step 3 — Streamline Tracer (step3.ipynb)

**Input**: `output/gwdt.npz` + `output/tubularity.npz`  
**Output**: `output/neurons.swc`, `output/neuron_tree.png`, `output/neuron_overlay.png`, `output/neuron_slices.png`

### Algorithm

1. **Soma detection** — largest CC in `T_combined >= SOMA_T_THRESH` within ridges; burn sphere around it
2. **Tip detection** — GWDT local maxima in ridges using sphere footprint (`TIP_SEARCH_RADIUS`); sort by GWDT descending
3. **Streamline tracing** — for each tip, follow orient_field toward soma:
   - Direction consistency: flip eigenvector polarity if `dot(v1, to_soma) < 0`
   - Additional consistency with previous step: flip if `dot(v1, prev_dir) < 0`
   - Stop when: `gwdt_work < BURN_THRESH` | `T < MIN_TRACE_T` | near soma | MAX_STEPS exceeded
4. **Burn** — after each node, set sphere of radius `r_vox × BURN_RADIUS_FACTOR` to 0 in `gwdt_work`
5. **node_vol** — each burned voxel stores the tree node index that burned it → O(1) parent lookup at join points
6. **Pruning** — discard paths where `mean_T < MIN_PATH_T` or `length < MIN_PATH_LEN_UM`
7. **Radius refinement** — EDT on ridges mask gives inscribed-sphere radius per node
8. **SWC export** — coordinates in µm (x,y,z), type=1 soma / type=3 neurites

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| STEP_SIZE_UM | 0.5 µm | Integration step size |
| MAX_STEPS | 8000 | Hard limit per streamline |
| TIP_SEARCH_RADIUS | 8 vox | Local-max footprint for tip detection |
| MIN_TIP_T | 0.05 | Minimum T at a valid tip |
| BURN_RADIUS_FACTOR | 1.5 | Burn sphere = factor × tube radius |
| BURN_THRESH | 0.01 | Stop tracing when gwdt_work falls below this |
| MIN_TRACE_T | 0.03 | Stop tracing when T falls below this |
| MIN_PATH_T | 0.05 | Prune branch if mean T below this |
| MIN_PATH_LEN_UM | 3.0 µm | Prune branch shorter than this |
| SOMA_T_THRESH | 0.3 | T threshold for soma CC detection |
| SOMA_BURN_FACTOR | 2.5 | Soma burn sphere multiplier |

### Helper Functions

- `_sphere_offsets(r_vox)` — integer voxel offsets within sphere
- `_interp3(vol, pos)` — trilinear scalar interpolation
- `_interp3_vec(vec_vol, pos)` — trilinear vector interpolation (returns unit vector)
- `_burn_mark(gwdt_work, node_vol, shape, cz, cy, cx, r_vox, node_idx)` — burn sphere
- `trace_streamline(...)` — integrate orient_field from tip to soma

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| No skeleton (Lee thinning removed) | Lee thinning causes crossing re-merge artifacts; GWDT ridges are sufficient |
| orient_field streamlines instead of Dijkstra | Sub-voxel accuracy, no binarization, respects tube orientation |
| GWDT burn prevents crossing reuse | Structural prevention — no need for explicit crossing detection |
| node_vol for parent lookup | O(1) join point identification without graph traversal |
| Tips = GWDT local maxima | Replaces degree-1 skeleton nodes; larger footprint avoids shaft tips |
| Sub-voxel positions throughout | No post-refinement step needed (was step9 in old pipeline) |

---

## SWC Output Format

```
# id  type  x  y  z  radius  parent
1  1  x_um  y_um  z_um  r_um  -1     ← soma (root)
2  3  x_um  y_um  z_um  r_um   1     ← neurite
...
```

Load in: Vaa3D, neuroglancer, neuTube, or any SWC-compatible viewer.
