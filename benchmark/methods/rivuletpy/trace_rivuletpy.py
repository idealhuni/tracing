import sys
import numpy as np
import tifffile
from scipy.ndimage import zoom
from rivuletpy.trace import R2Tracer

input_path, output_path = sys.argv[1], sys.argv[2]

img = tifffile.imread(input_path)
if img.ndim == 4:
    img = img[0]
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# Downsample so longest XY dimension <= 512 (rivuletpy is very slow on large images)
MAX_XY = 512
zy, zx = img.shape[1], img.shape[2]
if max(zy, zx) > MAX_XY:
    factor = MAX_XY / max(zy, zx)
    img = zoom(img, (1.0, factor, factor), order=1)
    print(f'  downsampled to {img.shape} (factor={factor:.2f})')

print(f'  tracing {img.shape} ...')
tracer = R2Tracer(speed=True, clean=True, silent=False)
swc, soma = tracer.trace(img, threshold=0.0)
swc.save(output_path)
print(f'  saved: {output_path}')
