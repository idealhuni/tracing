import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--data-path', required=True)
_p.add_argument('--voxel-xy', type=float, default=None)
_p.add_argument('--voxel-z', type=float, default=1.0)
_p.add_argument('--out-dir', default='output')
_p.add_argument('--downsample-xy', type=int, default=None)
_a = _p.parse_args()

# ── Config ──────────────────────────────────────────────────
import os
OUT_DIR = _a.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH     = _a.data_path
VOXEL_Z       = _a.voxel_z
DOWNSAMPLE_XY = _a.downsample_xy  # set by run_batch.py; fallback computed after voxel_xy is known
DOWNSAMPLE_Z  = 1

# ── Noise2Void ────────────────────────────────────────────────
N2V_ENABLE     = False
N2V_TRAIN      = True
N2V_MODEL_DIR  = f'{OUT_DIR}/n2v_model'
N2V_EPOCHS     = 50
N2V_PATIENCE   = 10
N2V_BATCH_SIZE = 8
N2V_PATCH_XY   = 32
N2V_PATCH_Z    = 8

# ── Imports ─────────────────────────────────────────────────
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_triangle
import gc, warnings
warnings.filterwarnings('ignore')

print('Imports ready.')

# ── Load ────────────────────────────────────────────────────
print(f'Loading: {DATA_PATH}')
stack_raw = tifffile.imread(DATA_PATH)
if stack_raw.ndim == 2:
    stack_raw = stack_raw[np.newaxis]
print(f'  Shape  : {stack_raw.shape}  (Z, Y, X)')
print(f'  dtype  : {stack_raw.dtype}')
print(f'  Memory : {stack_raw.nbytes / 1e9:.2f} GB')
print(f'  Range  : {int(stack_raw.min())} - {int(stack_raw.max())}')

if _a.voxel_xy is not None:
    voxel_xy = _a.voxel_xy
    print(f'  XY voxel: {voxel_xy:.7f} um/px  (from --voxel-xy)')
else:
    voxel_xy = None
    with tifffile.TiffFile(DATA_PATH) as tif:
        xres = tif.pages[0].tags.get('XResolution')
        runit = tif.pages[0].tags.get('ResolutionUnit')
        if xres is not None:
            val = xres.value
            res = (val[0] / val[1]) if isinstance(val, tuple) and val[1] != 0 else float(val)
            unit = runit.value if runit is not None else 2
            # unit=1: no unit, unit=2: inch, unit=3: cm — only trust if clearly not DPI
            if res > 0 and unit == 1:
                voxel_xy = 1.0 / res
    if voxel_xy is None or voxel_xy <= 0:
        voxel_xy = 1.0
        print('WARNING: XY voxel size not found in TIFF and --voxel-xy not given. Defaulting to 1.0 um/px.')
    else:
        print(f'  XY voxel: {voxel_xy:.7f} um/px  (from TIFF tag)')
print(f'  Z  voxel: {VOXEL_Z:.4f} um/slice')
print(f'  Anisotropy (Z/XY): {VOXEL_Z / voxel_xy:.2f}x')

# ── Compute DOWNSAMPLE_XY if not provided ────────────────────
TARGET_VOXEL_XY_UM = 0.55
if DOWNSAMPLE_XY is None:
    DOWNSAMPLE_XY = max(1, int(TARGET_VOXEL_XY_UM / voxel_xy))
    print(f'  DOWNSAMPLE_XY auto: {DOWNSAMPLE_XY}  (target ≤{TARGET_VOXEL_XY_UM} µm, raw={voxel_xy:.4f} µm)')
else:
    print(f'  DOWNSAMPLE_XY: {DOWNSAMPLE_XY}  (from --downsample-xy)')

# ── Downsample → float32 → Normalize ────────────────────────
stack_ds = stack_raw[::DOWNSAMPLE_Z, ::DOWNSAMPLE_XY, ::DOWNSAMPLE_XY]
vxy   = voxel_xy * DOWNSAMPLE_XY
vz    = VOXEL_Z  * DOWNSAMPLE_Z
aniso = vz / vxy
print(f'After downsample : {stack_ds.shape}  aniso={aniso:.2f}x')

stack_f = stack_ds.astype(np.float32)
p_low   = float(threshold_triangle(stack_f))
p_high  = float(np.percentile(stack_f, 99.9))
print(f'Full range  : {stack_f.min():.0f} - {stack_f.max():.0f}')
print(f'Clip range  : {p_low:.1f} - {p_high:.1f}  (triangle / 99.9th pct)')

stack_norm = np.clip(stack_f, p_low, p_high)
stack_norm = (stack_norm - p_low) / (p_high - p_low)
del stack_f; gc.collect()
print(f'Normalized  : {stack_norm.min():.4f} - {stack_norm.max():.4f}')

# ── Isotropic Z-rescaling ────────────────────────────────────
if aniso > 1.2:
    print(f'Rescaling Z by {aniso:.2f}x ...')
    stack_iso = ndimage.zoom(stack_norm, (aniso, 1.0, 1.0), order=1, prefilter=False)
    voxel_iso = vxy
    print(f'  Before : {stack_norm.shape}')
    print(f'  After  : {stack_iso.shape}')
    print(f'  Memory : {stack_iso.nbytes / 1e9:.2f} GB')
else:
    stack_iso = stack_norm.copy()
    voxel_iso = vxy
    print('Near-isotropic - Z rescaling skipped.')

del stack_norm; gc.collect()
print(f'Isotropic voxel size: {voxel_iso:.4f} um')

# ── N2V: check install ───────────────────────────────────────
import platform

if not N2V_ENABLE:
    N2V_AVAILABLE = False
    print('N2V_ENABLE=False -> N2V 생략')
else:
    try:
        from n2v.models import N2VConfig, N2V
        from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
        import tensorflow as tf
        if platform.system() == 'Darwin':
            tf.config.set_visible_devices([], 'GPU')
        print(f'n2v ready  | TF {tf.__version__}')
        print(f'Devices: {tf.config.list_logical_devices()}')
        N2V_AVAILABLE = True
    except ImportError as e:
        print(f'n2v not installed: {e}')
        N2V_AVAILABLE = False

# ── N2V: Training ────────────────────────────────────────────
if N2V_AVAILABLE and N2V_TRAIN:
    datagen = N2V_DataGenerator()
    vol_n2v = stack_iso[np.newaxis, ..., np.newaxis]
    patches = datagen.generate_patches_from_list(
        [vol_n2v],
        shape=(N2V_PATCH_Z, N2V_PATCH_XY, N2V_PATCH_XY),
    )
    n_val   = max(1, int(len(patches) * 0.1))
    X_train = patches[n_val:]
    X_val   = patches[:n_val]
    print(f'Patches: train={len(X_train)}  val={len(X_val)}  shape={X_train.shape[1:]}')

    config = N2VConfig(
        X_train,
        unet_kern_size          = 3,
        train_steps_per_epoch   = 400,
        train_epochs            = N2V_EPOCHS,
        train_loss              = 'mse',
        batch_norm              = False,
        train_batch_size        = N2V_BATCH_SIZE,
        n2v_perc_pix            = 0.198,
        n2v_patch_shape         = (N2V_PATCH_Z, N2V_PATCH_XY, N2V_PATCH_XY),
        n2v_manipulator         = 'uniform_withCP',
        n2v_neighborhood_radius = 3,
        unet_n_depth            = 1,
        unet_n_first            = 32,
    )
    model = N2V(config, 'n2v_3d', basedir=N2V_MODEL_DIR)

    from tensorflow.keras.callbacks import EarlyStopping
    model.prepare_for_training()
    model.callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=N2V_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ))
    history = model.train(X_train, X_val)
    print(f'Stopped at epoch {len(history.history["loss"])} / {N2V_EPOCHS}')
    print(f'Model saved -> {N2V_MODEL_DIR}/n2v_3d/')

elif N2V_AVAILABLE and not N2V_TRAIN:
    model = N2V(config=None, name='n2v_3d', basedir=N2V_MODEL_DIR)
    print(f'Loaded model from {N2V_MODEL_DIR}/n2v_3d/')

else:
    print('N2V skipped.')

# ── N2V: Prediction ─────────────────────────────────────────
if N2V_AVAILABLE:
    import time
    print('Denoising ...')
    t0 = time.time()
    stack_denoised = model.predict(stack_iso, axes='ZYX')
    stack_denoised = np.clip(stack_denoised, 0.0, 1.0).astype(np.float32)
    print(f'Done in {time.time()-t0:.1f}s')
    print(f'Range: {stack_denoised.min():.4f} - {stack_denoised.max():.4f}')
else:
    stack_denoised = stack_iso
    print('Using stack_iso (N2V skipped).')

# ── Save ────────────────────────────────────────────────────
OUT_TIF  = f'{OUT_DIR}/stack_preprocessed.tif'
OUT_META = f'{OUT_DIR}/preprocess_meta.npz'

tifffile.imwrite(OUT_TIF, stack_denoised)
np.savez(OUT_META,
    voxel_iso = np.float32(voxel_iso),
    p_low     = np.float32(p_low),
    p_high    = np.float32(p_high),
    aniso     = np.float32(aniso),
    n2v_used  = np.bool_(N2V_AVAILABLE),
)

print(f'Saved: {OUT_TIF}  {stack_denoised.shape}  {stack_denoised.dtype}')
print(f'Saved: {OUT_META}')
print()
print('Summary')
print(f'  voxel_iso  : {voxel_iso:.4f} um')
print(f'  clip range : {p_low:.1f} - {p_high:.1f}  (raw units)')
print(f'  anisotropy : {aniso:.2f}x  (Z/XY before rescaling)')
print(f'  N2V used   : {N2V_AVAILABLE}')
print(f'  Output     : {stack_denoised.shape}  float32 [0,1]')
