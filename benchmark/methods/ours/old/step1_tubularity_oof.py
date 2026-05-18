# ── Config ──────────────────────────────────────────────────
import argparse as _ap, os
_a = _ap.ArgumentParser()
_a.add_argument('--out-dir', default='output')
_a.add_argument('--slab-size', type=int, default=None)
_a = _a.parse_args()
OUT_DIR = _a.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

PREPROCESSED_TIF  = f'{OUT_DIR}/stack_preprocessed.tif'
PREPROCESSED_META = f'{OUT_DIR}/preprocess_meta.npz'

# OOF
TUBE_RADIUS_MIN_UM = 0.17
TUBE_RADIUS_MAX_UM = 1.71
N_RADII            = 8
N_SPHERE_PTS       = 300
SLAB_SIZE          = _a.slab_size if _a.slab_size is not None else 48

# Structure Tensor: sigma_d = radius / ST_SIGMA_RATIO  (per-scale matching)
ST_SIGMA_RATIO = 2.0
ST_SIGMA_INTEG = 2.0

# Guided Weighting
BETA             = 1.0
LAMBDA_BLOB      = 0.3
RIDGE_FILL_KERNEL = 3

# ── Imports ─────────────────────────────────────────────────
import numpy as np
import tifffile
from scipy import ndimage
from tqdm import tqdm
import gc, time, warnings
warnings.filterwarnings('ignore')

print('Imports ready.')

# ── Load from step0 ──────────────────────────────────────────
meta      = np.load(PREPROCESSED_META)
voxel_iso = float(meta['voxel_iso'])
stack_iso = tifffile.imread(PREPROCESSED_TIF).astype(np.float32)

print(f'Loaded   : {PREPROCESSED_TIF}')
print(f'Shape    : {stack_iso.shape}  (Z, Y, X)')
print(f'Range    : {stack_iso.min():.4f} - {stack_iso.max():.4f}')
print(f'Memory   : {stack_iso.nbytes / 1e9:.2f} GB')
print(f'voxel_iso: {voxel_iso:.4f} um')

NZ, NY, NX = stack_iso.shape

radii_vox = np.logspace(
    np.log10(TUBE_RADIUS_MIN_UM / voxel_iso),
    np.log10(TUBE_RADIUS_MAX_UM / voxel_iso),
    N_RADII,
).astype(np.float32)
OVERLAP = int(np.ceil(float(radii_vox[-1]) * 3))

print(f'Radii (vox): {radii_vox.round(2)}')
print(f'Overlap    : {OVERLAP} vox')

# ── Device detection ─────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    if torch.backends.mps.is_available():
        DEVICE  = torch.device('mps')
        USE_GPU = True
        print(f'Backend : MPS (Apple Metal)  PyTorch {torch.__version__}')
    elif torch.cuda.is_available():
        DEVICE  = torch.device('cuda')
        USE_GPU = True
        print(f'Backend : CUDA  PyTorch {torch.__version__}')
    else:
        DEVICE  = torch.device('cpu')
        USE_GPU = False
        print('No GPU - CPU fallback')
except ImportError:
    DEVICE = None; USE_GPU = False
    print('PyTorch not installed - CPU fallback')

# ── Sphere sampling (Fibonacci) ──────────────────────────────
def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) unit vectors in ZYX order, uniform on sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(n, dtype=np.float64)
    theta  = np.arccos(np.clip(1 - 2*(i + 0.5)/n, -1, 1))
    phi    = 2 * np.pi * i / golden
    return np.stack([
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
    ], axis=1).astype(np.float32)  # (N, 3)  ZYX

SPHERE_DIRS = fibonacci_sphere(N_SPHERE_PTS)
print(f'Sphere dirs: {SPHERE_DIRS.shape}')

# ── Shared: 3x3 symmetric Cardano eigensolver (CPU) ──────────
def _cardano_eigs_cpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx):
    """Eigenvalues of symmetric 3x3, sorted ascending. Returns (lam1, lam2, lam3)."""
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
    eigs    = np.stack([e1, e2, e3], axis=-1)
    eigs    = np.sort(eigs, axis=-1)
    return eigs[...,0], eigs[...,1], eigs[...,2]   # lam1 <= lam2 <= lam3


def _eigvec_cpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx, lam):
    """Eigenvector of eigenvalue lam via cross-product of two deflated rows."""
    r0 = np.stack([Mzz - lam, Mzy, Mzx], axis=-1)
    r1 = np.stack([Mzy, Myy - lam, Myx], axis=-1)
    v  = np.stack([
        r0[...,1]*r1[...,2] - r0[...,2]*r1[...,1],
        r0[...,2]*r1[...,0] - r0[...,0]*r1[...,2],
        r0[...,0]*r1[...,1] - r0[...,1]*r1[...,0],
    ], axis=-1)
    return (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)

print('Cardano eigensolver ready.')

# ── Shared: 3x3 symmetric Cardano eigensolver (GPU) ──────────
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
        lam, _ = torch.sort(torch.stack([e1,e2,e3], dim=-1), dim=-1)
        return lam[...,0], lam[...,1], lam[...,2]  # ascending

    def _eigvec_gpu(Mzz, Myy, Mxx, Mzy, Mzx, Myx, lam):
        r0 = torch.stack([Mzz - lam, Mzy, Mzx], dim=-1)
        r1 = torch.stack([Mzy, Myy - lam, Myx], dim=-1)
        v  = torch.stack([
            r0[...,1]*r1[...,2] - r0[...,2]*r1[...,1],
            r0[...,2]*r1[...,0] - r0[...,0]*r1[...,2],
            r0[...,0]*r1[...,1] - r0[...,1]*r1[...,0],
        ], dim=-1)
        return v / (torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-8))

    print('Cardano eigensolver (GPU) ready.')

# ── GPU helpers: 1D separable Gaussian ──────────────────────
if USE_GPU:
    def _gk(sigma, order):
        r = int(np.ceil(4.0 * sigma))
        x = torch.arange(-r, r+1, dtype=torch.float32, device=DEVICE)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
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

    print('GPU Gaussian helpers ready.')

# ── Structure Tensor: GPU ────────────────────────────────────
if USE_GPU:
    def structure_tensor_gpu(slab_t, sigma_d, sigma_i):
        sigma_d = max(0.5, sigma_d)
        k0d = _gk(sigma_d, 0); k1d = _gk(sigma_d, 1)
        k0i = _gk(sigma_i, 0)
        g0d = lambda v, ax: _c1d(v, k0d, ax)
        g1d = lambda v, ax: _c1d(v, k1d, ax)
        g0i = lambda v, ax: _c1d(v, k0i, ax)

        gz = g0d(g0d(g1d(slab_t, 0), 1), 2)
        gy = g0d(g1d(g0d(slab_t, 0), 1), 2)
        gx = g1d(g0d(g0d(slab_t, 0), 1), 2)

        sm = lambda v: g0i(g0i(g0i(v, 0), 1), 2)
        Jzz = sm(gz * gz); Jyy = sm(gy * gy); Jxx = sm(gx * gx)
        Jzy = sm(gz * gy); Jzx = sm(gz * gx); Jyx = sm(gy * gx)
        del gz, gy, gx

        lam1, _, _ = _cardano_eigs_gpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx)
        v_st = _eigvec_gpu(Jzz, Jyy, Jxx, Jzy, Jzx, Jyx, lam1)
        del Jzz, Jyy, Jxx, Jzy, Jzx, Jyx, lam1
        return v_st  # (Z,Y,X,3) float32

    print('Structure tensor (GPU) ready.')

# ── OOF: CPU ─────────────────────────────────────────────────
from scipy.ndimage import map_coordinates

def oof_slab_cpu(slab, radius, dirs):
    Z, Y, X = slab.shape
    N       = dirs.shape[0]
    zg, yg, xg = np.mgrid[0:Z, 0:Y, 0:X]
    base = np.stack([zg.ravel(), yg.ravel(), xg.ravel()], axis=0).astype(np.float32)

    Q = np.zeros((6, Z*Y*X), dtype=np.float32)
    for nz, ny, nx_ in dirs:
        off  = radius * np.array([[nz],[ny],[nx_]], dtype=np.float32)
        Ip   = map_coordinates(slab, base + off, order=1, mode='nearest', prefilter=False)
        Im   = map_coordinates(slab, base - off, order=1, mode='nearest', prefilter=False)
        gn   = (Ip - Im) / (2.0 * radius + 1e-8)
        Q[0] += gn * nz  * nz
        Q[1] += gn * ny  * ny
        Q[2] += gn * nx_ * nx_
        Q[3] += gn * nz  * ny
        Q[4] += gn * nz  * nx_
        Q[5] += gn * ny  * nx_
    Q /= N

    sh = (Z, Y, X)
    Qzz,Qyy,Qxx = Q[0].reshape(sh),Q[1].reshape(sh),Q[2].reshape(sh)
    Qzy,Qzx,Qyx = Q[3].reshape(sh),Q[4].reshape(sh),Q[5].reshape(sh)

    lam1, lam2, lam3 = _cardano_eigs_cpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx)
    both_neg = (lam1 < 0) & (lam2 < 0)
    response = np.where(both_neg, np.abs(lam1 + lam2), 0.0).astype(np.float32)
    v_oof = _eigvec_cpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx, lam3)
    return response, v_oof

# ── OOF: GPU ─────────────────────────────────────────────────
if USE_GPU:
    def oof_slab_gpu(slab_t, radius, dirs_t):
        Z, Y, X = slab_t.shape
        N       = dirs_t.shape[0]
        vol5    = slab_t.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)

        gz = torch.linspace(-1, 1, Z, device=DEVICE)
        gy = torch.linspace(-1, 1, Y, device=DEVICE)
        gx = torch.linspace(-1, 1, X, device=DEVICE)
        gz3, gy3, gx3 = torch.meshgrid(gz, gy, gx, indexing='ij')
        base_grid = torch.stack([gx3, gy3, gz3], dim=-1)  # (Z,Y,X,3) in xyz

        dz_n = radius * 2.0 / max(Z-1, 1)
        dy_n = radius * 2.0 / max(Y-1, 1)
        dx_n = radius * 2.0 / max(X-1, 1)

        Q = torch.zeros(6, Z*Y*X, dtype=torch.float32, device=DEVICE)

        for k in range(N):
            nz, ny, nx_ = dirs_t[k,0], dirs_t[k,1], dirs_t[k,2]
            off = torch.stack([nx_*dx_n, ny*dy_n, nz*dz_n])  # xyz order
            Ip  = F.grid_sample(vol5, (base_grid + off).unsqueeze(0),
                                mode='bilinear', padding_mode='border',
                                align_corners=True).squeeze().ravel()
            Im  = F.grid_sample(vol5, (base_grid - off).unsqueeze(0),
                                mode='bilinear', padding_mode='border',
                                align_corners=True).squeeze().ravel()
            gn  = (Ip - Im) / (2.0 * radius + 1e-8)
            Q[0] += gn * nz  * nz
            Q[1] += gn * ny  * ny
            Q[2] += gn * nx_ * nx_
            Q[3] += gn * nz  * ny
            Q[4] += gn * nz  * nx_
            Q[5] += gn * ny  * nx_
        Q /= N

        sh = (Z, Y, X)
        Qzz,Qyy,Qxx = Q[0].reshape(sh),Q[1].reshape(sh),Q[2].reshape(sh)
        Qzy,Qzx,Qyx = Q[3].reshape(sh),Q[4].reshape(sh),Q[5].reshape(sh)

        lam1, lam2, lam3 = _cardano_eigs_gpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx)
        both_neg = (lam1 < 0) & (lam2 < 0)
        response = torch.where(both_neg, torch.abs(lam1 + lam2),
                               torch.zeros_like(lam1)).float()
        v_oof = _eigvec_gpu(Qzz,Qyy,Qxx,Qzy,Qzx,Qyx, lam3)
        return response, v_oof

# ── Structure Tensor: CPU ────────────────────────────────────
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

    lam1, _, _ = _cardano_eigs_cpu(Jzz,Jyy,Jxx,Jzy,Jzx,Jyx)
    v_st = _eigvec_cpu(Jzz,Jyy,Jxx,Jzy,Jzx,Jyx, lam1)
    return v_st

print('Structure tensor (CPU) ready.')

# ── Blob response (LoG) ──────────────────────────────────────
def blob_response_cpu(slab, sigma):
    sigma = max(0.5, sigma)
    Hzz = ndimage.gaussian_filter(slab, sigma=sigma, order=(2,0,0)).astype(np.float32)
    Hyy = ndimage.gaussian_filter(slab, sigma=sigma, order=(0,2,0)).astype(np.float32)
    Hxx = ndimage.gaussian_filter(slab, sigma=sigma, order=(0,0,2)).astype(np.float32)
    return np.maximum(0.0, -(Hzz + Hyy + Hxx))

if USE_GPU:
    def blob_response_gpu(slab_t, sigma):
        sigma = max(0.5, sigma)
        k2 = _gk(sigma, 2)
        k0 = _gk(sigma, 0)
        Hzz = _c1d(_c1d(_c1d(slab_t, k2, 0), k0, 1), k0, 2)
        Hyy = _c1d(_c1d(_c1d(slab_t, k0, 0), k2, 1), k0, 2)
        Hxx = _c1d(_c1d(_c1d(slab_t, k0, 0), k0, 1), k2, 2)
        return torch.clamp(-(Hzz + Hyy + Hxx), min=0.0)

print('Blob response ready.')

# ── Main loop: Guided Weighting ──────────────────────────────
W_combined   = np.zeros((NZ, NY, NX), np.float32)
I_OOF_raw    = np.zeros((NZ, NY, NX), np.float32)
orient_field = np.zeros((NZ, NY, NX, 3), np.float16)
scale_idx    = np.zeros((NZ, NY, NX), np.uint8)
radius_map   = np.zeros((NZ, NY, NX), np.float32)

if USE_GPU:
    dirs_t = torch.from_numpy(SPHERE_DIRS).to(DEVICE)

n_slabs = int(np.ceil(NZ / SLAB_SIZE))
backend = 'GPU' if USE_GPU else 'CPU'
print(f'Stack: {stack_iso.shape}  Slabs: {n_slabs}  Backend: {backend}')
print(f'Radii: {N_RADII}  Sphere pts: {N_SPHERE_PTS}')
t0 = time.time()

slab_bar = tqdm(range(0, NZ, SLAB_SIZE), desc='slabs', unit='slab', total=n_slabs)
for slab_i, z0 in enumerate(slab_bar):
    z1     = min(z0 + SLAB_SIZE, NZ)
    z0p    = max(0, z0 - OVERLAP)
    z1p    = min(NZ, z1 + OVERLAP)
    core_s = z0 - z0p
    core_e = z1 - z0p
    slab   = stack_iso[z0p:z1p]
    cZ     = z1 - z0

    slab_bar.set_postfix(z=f'{z0}-{z1}', elapsed=f'{time.time()-t0:.0f}s')

    best_W    = np.zeros((cZ, NY, NX), np.float32)
    best_IOOF = np.zeros((cZ, NY, NX), np.float32)
    best_si   = np.zeros((cZ, NY, NX), np.uint8)
    best_v    = np.zeros((cZ, NY, NX, 3), np.float32)

    radii_bar = tqdm(enumerate(radii_vox), desc='  radii', total=N_RADII,
                     unit='r', leave=False)
    for ri, radius in radii_bar:
        radii_bar.set_postfix(r_um=f'{radius*voxel_iso:.2f}')
        sigma_d = float(radius) / ST_SIGMA_RATIO

        # ── OOF ──────────────────────────────────────────────────
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

        # ── Blob response (LoG) ──────────────────────────────────
        if USE_GPU:
            core_blob  = slab_t[core_s:core_e]
            blob_t     = blob_response_gpu(core_blob, float(radius))
            blob_c     = blob_t.cpu().numpy()
            del core_blob, blob_t
        else:
            blob_c = blob_response_cpu(slab[core_s:core_e], float(radius))
        blob_c *= float(radius) ** 2

        # ── Structure Tensor ─────────────────────────────────────
        if USE_GPU:
            core_t = slab_t[core_s:core_e]
            vst_t  = structure_tensor_gpu(core_t, sigma_d, ST_SIGMA_INTEG)
            vst_full = vst_t.cpu().numpy()
            del core_t, vst_t, slab_t
            if DEVICE.type == 'mps': torch.mps.empty_cache()
        else:
            vst_full = structure_tensor_cpu(slab[core_s:core_e], sigma_d, ST_SIGMA_INTEG)

        # ── C_align & Guided Weighting ────────────────────────────
        c_align = np.abs((voof_c * vst_full).sum(axis=-1))
        T_raw   = resp_c + LAMBDA_BLOB * blob_c
        W       = T_raw * (1.0 + BETA * c_align)

        improved = W > best_W
        best_W[improved]    = W[improved]
        best_IOOF[improved] = resp_c[improved]
        best_si[improved]   = ri
        best_v[improved]    = voof_c[improved]

        del resp_c, voof_c, vst_full, c_align, blob_c, T_raw, W, improved; gc.collect()

    W_combined[z0:z1]   = best_W
    I_OOF_raw[z0:z1]    = best_IOOF
    orient_field[z0:z1] = best_v.astype(np.float16)
    scale_idx[z0:z1]    = best_si

    del best_W, best_IOOF, best_si, best_v; gc.collect()

# ── Ridge filling ─────────────────────────────────────────────
from scipy.ndimage import maximum_filter
W_combined = maximum_filter(W_combined, size=RIDGE_FILL_KERNEL)

W_combined /= (W_combined.max() + 1e-10)
I_OOF_raw  /= (I_OOF_raw.max()  + 1e-10)
radius_map  = radii_vox[scale_idx] * voxel_iso

print(f'\nDone in {time.time()-t0:.0f}s')
print(f'W_combined : max={W_combined.max():.4f}  mean={W_combined.mean():.6f}')
print(f'I_OOF_raw  : max={I_OOF_raw.max():.4f}  mean={I_OOF_raw.mean():.6f}')

# ── Save ────────────────────────────────────────────────────
p999 = np.percentile(W_combined[W_combined > 0], 99.5)
T_combined = np.clip(W_combined / p999, 0, 1)

tifffile.imwrite(f'{OUT_DIR}/T_combined.tif', T_combined)
np.savez_compressed(f'{OUT_DIR}/tubularity.npz',
    T_combined   = T_combined,
    I_OOF_raw    = I_OOF_raw,
    orient_field = orient_field,
    radius_map   = radius_map,
    scale_idx    = scale_idx,
    radii        = radii_vox,
    voxel_iso    = np.float32(voxel_iso),
)
print(f'Saved: {OUT_DIR}/tubularity.npz')
print(f'  T_combined (W)  {W_combined.shape} float32')
print(f'  I_OOF_raw       {I_OOF_raw.shape} float32')
print(f'  orient_field    {orient_field.shape} float16')
print(f'  radius_map      {radius_map.shape} float32')
