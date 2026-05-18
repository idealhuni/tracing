#!/usr/bin/env python3
"""Step 1: OOF + Structure Tensor guided weighting -> tubularity.npz"""
import argparse
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import maximum_filter

warnings.filterwarnings('ignore')

# ── Config (fixed) ───────────────────────────────────────────
TUBE_RADIUS_MIN_UM  = 0.15
TUBE_RADIUS_MAX_UM  = 3.5
TUBE_RADIUS_MIN_VOX = 1.0
N_RADII             = 10
N_SPHERE_PTS        = 100
INPUT_DOWNSAMPLE    = 1
ST_SIGMA_RATIO      = 2.0
ST_SIGMA_INTEG      = 2.0
BETA                = 1.0
LAMBDA_BLOB         = 0.3
RIDGE_FILL_KERNEL   = 3


# ── Sphere sampling ───────────────────────────────────────────
def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n, dtype=np.float64)
    theta  = np.arccos(np.clip(1 - 2*(i + 0.5)/n, -1, 1))
    phi    = 2 * np.pi * i / golden
    return np.stack([
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
    ], axis=1).astype(np.float32)


# ── Cardano eigensolver (CPU) ─────────────────────────────────
def _cardano_eigs_cpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx):
    p1  = Mzy**2 + Mzx**2 + Myx**2
    q   = (Mzz + Myy + Mxx) / 3.0
    p2  = (Mzz-q)**2 + (Myy-q)**2 + (Mxx-q)**2 + 2*p1
    p   = np.sqrt(np.maximum(p2/6, 0))
    ps  = np.where(p > 1e-12, p, 1.0)
    B00 = np.where(p>1e-12, (Mzz-q)/ps, 0.)
    B11 = np.where(p>1e-12, (Myy-q)/ps, 0.)
    B22 = np.where(p>1e-12, (Mxx-q)/ps, 0.)
    B01 = np.where(p>1e-12, Mzy/ps, 0.)
    B02 = np.where(p>1e-12, Mzx/ps, 0.)
    B12 = np.where(p>1e-12, Myx/ps, 0.)
    r   = np.clip((B00*(B11*B22-B12**2) - B01*(B01*B22-B12*B02)
                   + B02*(B01*B12-B11*B02)) / 2, -1, 1)
    phi = np.arccos(r) / 3
    tp  = 2*np.pi/3
    e1  = (q + 2*p*np.cos(phi)).astype(np.float32)
    e2  = (q + 2*p*np.cos(phi + tp)).astype(np.float32)
    e3  = (q + 2*p*np.cos(phi + 2*tp)).astype(np.float32)
    eigs = np.sort(np.stack([e1, e2, e3], axis=-1), axis=-1)
    return eigs[...,0], eigs[...,1], eigs[...,2]


def _eigvec_cpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx, lam):
    r0 = np.stack([Mzz - lam, Mzy, Mzx], axis=-1)
    r1 = np.stack([Mzy, Myy - lam, Myx], axis=-1)
    v  = np.stack([
        r0[...,1]*r1[...,2] - r0[...,2]*r1[...,1],
        r0[...,2]*r1[...,0] - r0[...,0]*r1[...,2],
        r0[...,0]*r1[...,1] - r0[...,1]*r1[...,0],
    ], axis=-1)
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


# ── CPU implementations ───────────────────────────────────────
def oof_slab_cpu(slab, radius, dirs):
    from scipy.ndimage import map_coordinates
    Z, Y, X = slab.shape
    N       = dirs.shape[0]
    zg, yg, xg = np.mgrid[0:Z, 0:Y, 0:X]
    base = np.stack([zg.ravel(), yg.ravel(), xg.ravel()], axis=0).astype(np.float32)
    Q = np.zeros((6, Z*Y*X), dtype=np.float32)
    for nz, ny, nx_ in dirs:
        off = radius * np.array([[nz],[ny],[nx_]], dtype=np.float32)
        Ip  = map_coordinates(slab, base + off, order=1, mode='nearest', prefilter=False)
        Im  = map_coordinates(slab, base - off, order=1, mode='nearest', prefilter=False)
        gn  = (Ip - Im) / (2.0 * radius + 1e-8)
        Q[0] += gn * nz  * nz;  Q[1] += gn * ny  * ny;  Q[2] += gn * nx_ * nx_
        Q[3] += gn * nz  * ny;  Q[4] += gn * nz  * nx_; Q[5] += gn * ny  * nx_
    Q /= N
    sh = (Z, Y, X)
    Qzz,Qyy,Qxx = Q[0].reshape(sh), Q[1].reshape(sh), Q[2].reshape(sh)
    Qzy,Qzx,Qyx = Q[3].reshape(sh), Q[4].reshape(sh), Q[5].reshape(sh)
    lam1, lam2, lam3 = _cardano_eigs_cpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx)
    both_neg = (lam1 < 0) & (lam2 < 0)
    response = np.where(both_neg, np.abs(lam1 + lam2), 0.0).astype(np.float32)
    v_oof    = _eigvec_cpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx, lam3)
    return response, v_oof


def structure_tensor_cpu(slab, sigma_d, sigma_i):
    sigma_d = max(0.5, sigma_d)
    gz = ndimage.gaussian_filter(slab, sigma=sigma_d, order=(1,0,0), output=np.float32)
    gy = ndimage.gaussian_filter(slab, sigma=sigma_d, order=(0,1,0), output=np.float32)
    gx = ndimage.gaussian_filter(slab, sigma=sigma_d, order=(0,0,1), output=np.float32)
    Jzz = ndimage.gaussian_filter(gz*gz, sigma=sigma_i).astype(np.float32)
    Jyy = ndimage.gaussian_filter(gy*gy, sigma=sigma_i).astype(np.float32)
    Jxx = ndimage.gaussian_filter(gx*gx, sigma=sigma_i).astype(np.float32)
    Jzy = ndimage.gaussian_filter(gz*gy, sigma=sigma_i).astype(np.float32)
    Jzx = ndimage.gaussian_filter(gz*gx, sigma=sigma_i).astype(np.float32)
    Jyx = ndimage.gaussian_filter(gy*gx, sigma=sigma_i).astype(np.float32)
    del gz, gy, gx; gc.collect()
    lam1, _, _ = _cardano_eigs_cpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx)
    v_st = _eigvec_cpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx, lam1)
    return v_st


def blob_response_cpu(slab, sigma):
    sigma = max(0.5, sigma)
    Hzz = ndimage.gaussian_filter(slab, sigma=sigma, order=(2,0,0)).astype(np.float32)
    Hyy = ndimage.gaussian_filter(slab, sigma=sigma, order=(0,2,0)).astype(np.float32)
    Hxx = ndimage.gaussian_filter(slab, sigma=sigma, order=(0,0,2)).astype(np.float32)
    return np.maximum(0.0, -(Hzz + Hyy + Hxx))


# ── GPU setup (optional) ──────────────────────────────────────
DEVICE  = None
USE_GPU = False
_oof_cache = {}

try:
    import torch
    import torch.nn.functional as _F
    if torch.backends.mps.is_available():
        DEVICE  = torch.device('mps')
        USE_GPU = True
        print(f'Backend: MPS  PyTorch {torch.__version__}')
    elif torch.cuda.is_available():
        DEVICE  = torch.device('cuda')
        USE_GPU = True
        print(f'Backend: CUDA  PyTorch {torch.__version__}')
    else:
        print('No GPU — CPU fallback')
except ImportError:
    print('PyTorch not installed — CPU fallback')


if USE_GPU:
    def _cardano_eigs_gpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx):
        p1  = Mzy**2 + Mzx**2 + Myx**2
        q   = (Mzz + Myy + Mxx) / 3.0
        p2  = (Mzz-q)**2 + (Myy-q)**2 + (Mxx-q)**2 + 2*p1
        p   = torch.sqrt(torch.clamp(p2/6, min=0))
        ps  = p.clamp(min=1e-12)
        B00 = (Mzz-q)/ps; B11 = (Myy-q)/ps; B22 = (Mxx-q)/ps
        B01 = Mzy/ps;      B02 = Mzx/ps;      B12 = Myx/ps
        r   = torch.clamp((B00*(B11*B22-B12**2) - B01*(B01*B22-B12*B02)
                           + B02*(B01*B12-B11*B02)) / 2, -1, 1)
        phi = torch.acos(r) / 3
        tp  = float(2*np.pi/3)
        e1  = q + 2*p*torch.cos(phi)
        e2  = q + 2*p*torch.cos(phi + tp)
        e3  = q + 2*p*torch.cos(phi + 2*tp)
        lam, _ = torch.sort(torch.stack([e1, e2, e3], dim=-1), dim=-1)
        return lam[...,0], lam[...,1], lam[...,2]

    def _eigvec_gpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx, lam):
        r0 = torch.stack([Mzz - lam, Mzy, Mzx], dim=-1)
        r1 = torch.stack([Mzy, Myy - lam, Myx], dim=-1)
        v  = torch.stack([
            r0[...,1]*r1[...,2] - r0[...,2]*r1[...,1],
            r0[...,2]*r1[...,0] - r0[...,0]*r1[...,2],
            r0[...,0]*r1[...,1] - r0[...,1]*r1[...,0],
        ], dim=-1)
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-8))

    def _gk(sigma, order):
        r  = int(np.ceil(4.0 * sigma))
        x  = torch.arange(-r, r+1, dtype=torch.float32, device=DEVICE)
        g  = torch.exp(-0.5 * (x / sigma) ** 2)
        g  = g / g.sum()
        if order == 1: g = (-x / sigma**2) * g
        if order == 2: g = ((x**2 / sigma**4) - 1.0 / sigma**2) * g
        return g

    def _c1d(v, k, axis):
        pad = k.shape[0] // 2
        Z, Y, X = v.shape
        kv = k.view(1, 1, -1)
        if axis == 0:
            u = v.permute(1,2,0).contiguous().reshape(Y*X, 1, Z)
            u = torch.nn.functional.pad(u, (pad, pad), mode='replicate')
            return torch.nn.functional.conv1d(u, kv).reshape(Y, X, Z).permute(2,0,1)
        elif axis == 1:
            u = v.permute(0,2,1).contiguous().reshape(Z*X, 1, Y)
            u = torch.nn.functional.pad(u, (pad, pad), mode='replicate')
            return torch.nn.functional.conv1d(u, kv).reshape(Z, X, Y).permute(0,2,1)
        else:
            u = v.contiguous().reshape(Z*Y, 1, X)
            u = torch.nn.functional.pad(u, (pad, pad), mode='replicate')
            return torch.nn.functional.conv1d(u, kv).reshape(Z, Y, X)

    def structure_tensor_gpu(slab_t, sigma_d, sigma_i):
        sigma_d = max(0.5, sigma_d)
        k0d = _gk(sigma_d, 0); k1d = _gk(sigma_d, 1)
        k0i = _gk(sigma_i, 0)
        gz = _c1d(_c1d(_c1d(slab_t, k1d, 0), k0d, 1), k0d, 2)
        gy = _c1d(_c1d(_c1d(slab_t, k0d, 0), k1d, 1), k0d, 2)
        gx = _c1d(_c1d(_c1d(slab_t, k0d, 0), k0d, 1), k1d, 2)
        sm = lambda v: _c1d(_c1d(_c1d(v, k0i, 0), k0i, 1), k0i, 2)
        Jzz = sm(gz*gz); Jyy = sm(gy*gy); Jxx = sm(gx*gx)
        Jzy = sm(gz*gy); Jzx = sm(gz*gx); Jyx = sm(gy*gx)
        del gz, gy, gx
        lam1, _, _ = _cardano_eigs_gpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx)
        v_st = _eigvec_gpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx, lam1)
        del Jzz, Jyy, Jxx, Jzy, Jzx, Jyx, lam1
        return v_st

    def blob_response_gpu(slab_t, sigma):
        sigma = max(0.5, sigma)
        k2 = _gk(sigma, 2); k0 = _gk(sigma, 0)
        Hzz = _c1d(_c1d(_c1d(slab_t, k2, 0), k0, 1), k0, 2)
        Hyy = _c1d(_c1d(_c1d(slab_t, k0, 0), k2, 1), k0, 2)
        Hxx = _c1d(_c1d(_c1d(slab_t, k0, 0), k0, 1), k2, 2)
        return torch.clamp(-(Hzz + Hyy + Hxx), min=0.0)

    def oof_slab_gpu(slab_t, radius, dirs_t):
        Z, Y, X = slab_t.shape
        N       = dirs_t.shape[0]
        vol5    = slab_t.unsqueeze(0).unsqueeze(0)
        key = (Z, Y, X)
        if key not in _oof_cache:
            gz = torch.linspace(-1, 1, Z, device=DEVICE)
            gy = torch.linspace(-1, 1, Y, device=DEVICE)
            gx = torch.linspace(-1, 1, X, device=DEVICE)
            gz3, gy3, gx3 = torch.meshgrid(gz, gy, gx, indexing='ij')
            bg = torch.stack([gx3, gy3, gz3], dim=-1)
            _oof_cache[key] = dict(
                base_grid = bg,
                grid_fwd  = bg.clone(),
                grid_bwd  = bg.clone(),
            )
        base_grid = _oof_cache[key]['base_grid']
        grid_fwd  = _oof_cache[key]['grid_fwd']
        grid_bwd  = _oof_cache[key]['grid_bwd']
        dz_n = radius * 2.0 / max(Z - 1, 1)
        dy_n = radius * 2.0 / max(Y - 1, 1)
        dx_n = radius * 2.0 / max(X - 1, 1)
        scale    = torch.tensor([dx_n, dy_n, dz_n], device=DEVICE)
        offs_xyz = dirs_t[:, [2, 1, 0]] * scale
        nz_v, ny_v, nx_v = dirs_t[:,0], dirs_t[:,1], dirs_t[:,2]
        coeffs_np = torch.stack([
            nz_v*nz_v, ny_v*ny_v, nx_v*nx_v,
            nz_v*ny_v, nz_v*nx_v, ny_v*nx_v,
        ], dim=1).cpu().numpy()
        Q     = torch.zeros(6, Z, Y, X, dtype=torch.float32, device=DEVICE)
        denom = 2.0 * radius + 1e-8
        for k in range(N):
            torch.add(base_grid, offs_xyz[k], out=grid_fwd)
            torch.sub(base_grid, offs_xyz[k], out=grid_bwd)
            Ip = _F.grid_sample(vol5, grid_fwd.unsqueeze(0), mode='bilinear',
                                padding_mode='border', align_corners=True).view(Z, Y, X)
            Im = _F.grid_sample(vol5, grid_bwd.unsqueeze(0), mode='bilinear',
                                padding_mode='border', align_corners=True).view(Z, Y, X)
            Ip.sub_(Im).div_(denom)
            c = coeffs_np[k]
            Q[0].add_(Ip, alpha=float(c[0])); Q[1].add_(Ip, alpha=float(c[1]))
            Q[2].add_(Ip, alpha=float(c[2])); Q[3].add_(Ip, alpha=float(c[3]))
            Q[4].add_(Ip, alpha=float(c[4])); Q[5].add_(Ip, alpha=float(c[5]))
        Q /= N
        Qzz, Qyy, Qxx = Q[0], Q[1], Q[2]
        Qzy, Qzx, Qyx = Q[3], Q[4], Q[5]
        lam1, lam2, lam3 = _cardano_eigs_gpu(Qzz, Qyy, Qxx, Qzy, Qzx, Qyx)
        both_neg = (lam1 < 0) & (lam2 < 0)
        response = torch.where(both_neg, torch.abs(lam1 + lam2), torch.zeros_like(lam1))
        v_oof    = _eigvec_gpu(Qzz, Qyy, Qxx, Qzy, Qzx, Qyx, lam3)
        return response, v_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',   required=True)
    ap.add_argument('--slab-size', type=int, default=48)
    args = ap.parse_args()

    out_dir   = Path(args.out_dir)
    SLAB_SIZE = args.slab_size

    PREPROCESSED_TIF  = out_dir / 'stack_preprocessed.tif'
    PREPROCESSED_META = out_dir / 'preprocess_meta.npz'
    OUT_TIF = out_dir / 'T_combined.tif'
    OUT_NPZ = out_dir / 'tubularity.npz'

    # ── Load from step0 ──────────────────────────────────────────
    meta      = np.load(str(PREPROCESSED_META))
    voxel_iso = float(meta['voxel_iso'])
    stack_iso = tifffile.imread(str(PREPROCESSED_TIF)).astype(np.float32)

    if INPUT_DOWNSAMPLE > 1:
        from scipy.ndimage import zoom
        stack_iso = zoom(stack_iso, 1.0 / INPUT_DOWNSAMPLE, order=1).astype(np.float32)
        voxel_iso = voxel_iso * INPUT_DOWNSAMPLE

    print(f'Loaded: {PREPROCESSED_TIF}')
    print(f'Shape : {stack_iso.shape}  voxel_iso={voxel_iso:.4f} um')
    print(f'Memory: {stack_iso.nbytes / 1e9:.2f} GB')

    NZ, NY, NX = stack_iso.shape

    r_min = max(TUBE_RADIUS_MIN_UM, TUBE_RADIUS_MIN_VOX * voxel_iso)
    r_max = TUBE_RADIUS_MAX_UM
    radii_vox = np.logspace(
        np.log10(r_min / voxel_iso),
        np.log10(r_max / voxel_iso),
        N_RADII,
    ).astype(np.float32)
    OVERLAP = int(np.ceil(float(radii_vox[-1]) * 3))

    print(f'Radii (vox): {radii_vox.round(2)}')
    print(f'Overlap    : {OVERLAP} vox')
    print(f'Slab size  : {SLAB_SIZE}')

    SPHERE_DIRS = fibonacci_sphere(N_SPHERE_PTS)

    if USE_GPU:
        dirs_t = torch.from_numpy(SPHERE_DIRS).to(DEVICE)

    # ── Main loop ────────────────────────────────────────────────
    W_combined   = np.zeros((NZ, NY, NX), np.float32)
    I_OOF_raw    = np.zeros((NZ, NY, NX), np.float32)
    orient_field = np.zeros((NZ, NY, NX, 3), np.float16)
    scale_idx    = np.zeros((NZ, NY, NX), np.uint8)

    n_slabs = int(np.ceil(NZ / SLAB_SIZE))
    backend = 'GPU' if USE_GPU else 'CPU'
    print(f'Stack: {stack_iso.shape}  Slabs: {n_slabs}  Backend: {backend}')
    t0 = time.time()

    for slab_i, z0 in enumerate(range(0, NZ, SLAB_SIZE)):
        z1     = min(z0 + SLAB_SIZE, NZ)
        z0p    = max(0, z0 - OVERLAP)
        z1p    = min(NZ, z1 + OVERLAP)
        core_s = z0 - z0p
        core_e = z1 - z0p
        slab   = stack_iso[z0p:z1p]
        cZ     = z1 - z0

        best_W    = np.zeros((cZ, NY, NX), np.float32)
        best_IOOF = np.zeros((cZ, NY, NX), np.float32)
        best_si   = np.zeros((cZ, NY, NX), np.uint8)
        best_v    = np.zeros((cZ, NY, NX, 3), np.float32)

        for ri, radius in enumerate(radii_vox):
            sigma_d = float(radius) / ST_SIGMA_RATIO

            if USE_GPU:
                slab_t         = torch.from_numpy(slab).to(DEVICE)
                resp_t, voof_t = oof_slab_gpu(slab_t, float(radius), dirs_t)
                resp_c         = resp_t[core_s:core_e].cpu().numpy()
                voof_c         = voof_t[core_s:core_e].cpu().numpy()
                del resp_t, voof_t
            else:
                resp_full, voof_full = oof_slab_cpu(slab, float(radius), SPHERE_DIRS)
                resp_c  = resp_full[core_s:core_e]
                voof_c  = voof_full[core_s:core_e]
                del resp_full, voof_full

            if USE_GPU:
                core_blob = slab_t[core_s:core_e]
                blob_t    = blob_response_gpu(core_blob, float(radius))
                blob_c    = blob_t.cpu().numpy()
                del core_blob, blob_t
            else:
                blob_c = blob_response_cpu(slab[core_s:core_e], float(radius))
            blob_c *= float(radius) ** 2

            if USE_GPU:
                core_t   = slab_t[core_s:core_e]
                vst_t    = structure_tensor_gpu(core_t, sigma_d, ST_SIGMA_INTEG)
                vst_full = vst_t.cpu().numpy()
                del core_t, vst_t, slab_t
                if DEVICE.type == 'mps': torch.mps.empty_cache()
            else:
                vst_full = structure_tensor_cpu(slab[core_s:core_e], sigma_d, ST_SIGMA_INTEG)

            c_align = np.abs((voof_c * vst_full).sum(axis=-1))
            T_raw   = resp_c + LAMBDA_BLOB * blob_c
            W       = T_raw * (1.0 + BETA * c_align)

            improved = W > best_W
            best_W[improved]    = W[improved]
            best_IOOF[improved] = resp_c[improved]
            best_si[improved]   = ri
            best_v[improved]    = voof_c[improved]

            del resp_c, voof_c, vst_full, c_align, blob_c, T_raw, W, improved
            gc.collect()

        W_combined[z0:z1]   = best_W
        I_OOF_raw[z0:z1]    = best_IOOF
        orient_field[z0:z1] = best_v.astype(np.float16)
        scale_idx[z0:z1]    = best_si

        del best_W, best_IOOF, best_si, best_v; gc.collect()
        print(f'  slab {slab_i+1}/{n_slabs}  z={z0}-{z1}  {time.time()-t0:.0f}s')

    # ── Ridge filling + normalize ────────────────────────────────
    W_combined = maximum_filter(W_combined, size=RIDGE_FILL_KERNEL)
    W_combined /= (W_combined.max() + 1e-10)
    I_OOF_raw  /= (I_OOF_raw.max()  + 1e-10)
    radius_map  = radii_vox[scale_idx] * voxel_iso

    print(f'Done in {time.time()-t0:.0f}s')
    print(f'W_combined: max={W_combined.max():.4f}  mean={W_combined.mean():.6f}')

    # ── Save ────────────────────────────────────────────────────
    p999       = np.percentile(W_combined[W_combined > 0], 99.5)
    T_combined = np.clip(W_combined / p999, 0, 1)

    tifffile.imwrite(str(OUT_TIF), T_combined)
    np.savez(str(OUT_NPZ),
        T_combined   = T_combined,
        I_OOF_raw    = I_OOF_raw,
        orient_field = orient_field,
        radius_map   = radius_map,
        scale_idx    = scale_idx,
        radii        = radii_vox,
        voxel_iso    = np.float32(voxel_iso),
    )
    print(f'Saved: {OUT_NPZ}')
    print(f'  T_combined   {T_combined.shape} float32')
    print(f'  orient_field {orient_field.shape} float16')
    print(f'  radius_map   {radius_map.shape} float32')


if __name__ == '__main__':
    main()
