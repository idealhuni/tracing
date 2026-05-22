"""
Batch runner for BigNeuron benchmark.

Usage:
    python run_batch.py --method ours|vaa3d [--limit N]

For 'ours': runs tracer_aniso pipeline (step0→step1→step1b→step2→step3_auto).
For 'vaa3d': calls methods/vaa3d/run_vaa3d.sh.

Output SWCs land in results/<method>/<sample_name>.swc
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "images"
GOLD_DIR    = ROOT / "data" / "gold_standard"
RESULTS_DIR = ROOT / "results"


def parse_voxel_sizes(stem: str) -> tuple[float, float]:
    """Return (voxel_xy, voxel_z) in µm from gold_standard/{stem}.pixelsize.txt."""
    pxsz = GOLD_DIR / f"{stem}.pixelsize.txt"
    if not pxsz.exists():
        pxsz = GOLD_DIR / f"{stem}.tif.pixelsize.txt"
    if not pxsz.exists():
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"WARNING: no pixelsize.txt for '{stem}'", file=sys.stderr)
        print(f"  Expected: {pxsz}", file=sys.stderr)
        print(f"  Defaulting to voxel_xy=1.0 µm, voxel_z=1.0 µm", file=sys.stderr)
        print(f"  → OOF radius range will be wrong (radii_vox_max=1.71)", file=sys.stderr)
        print(f"  Create the file with:", file=sys.stderr)
        print(f'  echo "Voxel size: X x Y x Z micron" > {pxsz}', file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        return 1.0, 1.0
    text = pxsz.read_text()
    m = re.search(r'Voxel size:\s*([\d.]+)x([\d.]+)x([\d.]+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(3))
    print(f"\nWARNING: '{pxsz}' exists but format not recognized.", file=sys.stderr)
    print(f"  Expected: 'Voxel size: XxYxZ micron'  got: {text.strip()!r}", file=sys.stderr)
    print(f"  Defaulting to voxel_xy=1.0 µm, voxel_z=1.0 µm\n", file=sys.stderr)
    return 1.0, 1.0


def run_py_script(script: Path, work_dir: Path, extra_args=None):
    """Run a Python script with real-time stdout streaming."""
    cmd = [sys.executable, str(script)] + (extra_args or [])
    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        tqdm.write(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, script)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


STEP_SENTINELS = {
    "step0_preprocess":    "preprocess_meta.npz",
    "step1_tubularity_oof":"tubularity.npz",
    "step1b_soma":         "soma.npz",
    "step2_prep_aniso":    "prep_riem.npz",
    "step3_auto":          "neurons_auto.swc",
}


TARGET_VOXEL_ISO_UM = 0.35   # isotropic voxel target for step0 fractional zoom
XY_MAX              = 4096   # XY hard cap (극단적 케이스 안전망, 물리 기반 700M cap이 주 제어)
TUBE_RADIUS_MAX_UM  = 3.5    # must match step1_tubularity_oof.py
MAX_OUTPUT_VOXELS   = 700_000_000  # must match step0_preprocess.py


def _resolve_voxel_iso(tif: Path, voxel_xy: float, voxel_z: float) -> float:
    """Compute the actual voxel_iso step0 will produce (mirrors step0 logic)."""
    import tifffile as _tf
    voxel_iso = TARGET_VOXEL_ISO_UM if voxel_xy <= TARGET_VOXEL_ISO_UM else voxel_xy
    with _tf.TiffFile(str(tif)) as t:
        nZ = len(t.pages)
        nY, nX = t.pages[0].shape
    # XY max cap
    voxel_iso = max(voxel_iso, voxel_xy * max(nY, nX) / XY_MAX)
    zoom_xy = min(1.0, voxel_xy / voxel_iso)
    zoom_z  = voxel_z  / voxel_iso
    # Volume cap
    if int(nZ * zoom_z * nY * zoom_xy * nX * zoom_xy) > MAX_OUTPUT_VOXELS:
        viso_min  = (nZ * nY * nX * voxel_z * voxel_xy**2 / MAX_OUTPUT_VOXELS) ** (1/3)
        voxel_iso = max(voxel_iso, viso_min)
    return voxel_iso


def _slab_size_for(tif: Path, voxel_xy: float, voxel_z: float,
                   base: int = 48, gpu_gb: float = 24.0) -> int:
    """Compute slab size from GPU memory budget.

    Peak usage during OOF: ~25 single-channel tensors simultaneously
    (slab_t×1 + cache_grids×9 + Q×6 + cardano_intermediates×9).
    """
    import math, tifffile as _tf
    voxel_iso = _resolve_voxel_iso(tif, voxel_xy, voxel_z)
    overlap   = int(math.ceil(TUBE_RADIUS_MAX_UM / voxel_iso * 3))

    # Prefer preprocessed TIF dimensions (exact); fall back to estimating from original
    work_dir       = ROOT / "methods" / "ours"
    preprocessed   = work_dir / "output" / tif.stem / "stack_preprocessed.tif"
    if preprocessed.exists():
        with _tf.TiffFile(str(preprocessed)) as t:
            page = t.pages[0]
            y, x = page.shape[0], page.shape[1]
    else:
        zoom_xy = voxel_xy / voxel_iso
        with _tf.TiffFile(str(tif)) as t:
            page = t.pages[0]
            y = max(1, int(page.shape[0] * zoom_xy))
            x = max(1, int(page.shape[1] * zoom_xy))

    # Use total system RAM × 0.75 as GPU budget (MPS shares system memory)
    try:
        import subprocess
        r = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        gpu_gb = int(r.stdout.strip()) / 1e9 * 0.75
    except Exception:
        pass

    PEAK_TENSORS = 25
    eff_z_budget = int(gpu_gb * 1e9 / (y * x * 4 * PEAK_TENSORS))
    return max(4, min(base, eff_z_budget - 2 * overlap))


def run_ours(tif: Path, out_swc: Path, sample_label: str = ""):
    out_swc.parent.mkdir(parents=True, exist_ok=True)
    work_dir = ROOT / "methods" / "ours"
    out_dir  = work_dir / "output" / tif.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_xy, voxel_z = parse_voxel_sizes(tif.stem)
    voxel_iso = _resolve_voxel_iso(tif, voxel_xy, voxel_z)
    slab_size = _slab_size_for(tif, voxel_xy, voxel_z)
    tqdm.write(f"  {tif.stem}: voxel_xy={voxel_xy:.4f}µm  → viso={voxel_iso:.4f}µm  slab={slab_size}")

    steps = ["step0_preprocess", "step1_tubularity_oof", "step1b_soma",
             "step2_prep_aniso", "step3_auto"]

    sample_start = time.time()
    with tqdm(steps, desc=f"  {sample_label}", unit="step", leave=False,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as sbar:
        for step_name in sbar:
            sentinel = out_dir / STEP_SENTINELS[step_name]
            if sentinel.exists():
                sbar.set_postfix_str(f"{step_name} (cached)")
                tqdm.write(f"  skip {step_name} (output exists)")
                continue

            sbar.set_postfix_str(step_name)
            t0 = time.time()

            py_script = work_dir / f"{step_name}.py"
            extra = ["--out-dir", str(out_dir)]
            if step_name == "step0_preprocess":
                extra += ["--data-path", str(tif),
                          "--voxel-xy", str(voxel_xy),
                          "--voxel-z", str(voxel_z),
                          "--target-voxel-iso", str(TARGET_VOXEL_ISO_UM),
                          "--xy-max", str(XY_MAX)]
            if step_name == "step1_tubularity_oof":
                extra += ["--slab-size", str(slab_size)]
                tqdm.write(f"  slab_size={slab_size} (image {tif.stem})")

            run_py_script(py_script, work_dir, extra_args=extra)

            step_elapsed = time.time() - t0
            sbar.set_postfix_str(f"{step_name} ({_fmt(step_elapsed)})")

    sample_elapsed = time.time() - sample_start
    swc_out = out_dir / "neurons_auto.swc"
    if swc_out.exists():
        shutil.copy(swc_out, out_swc)
        tqdm.write(f"[ours] {tif.stem} → {out_swc}  (total {_fmt(sample_elapsed)})")
    else:
        tqdm.write(f"[ours] WARNING: neurons_auto.swc not found for {tif.stem}", file=sys.stderr)


def run_shell(method: str, tif: Path, out_swc: Path):
    script = ROOT / "methods" / method / f"run_{method}.sh"
    out_swc.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subprocess.run(["bash", str(script), str(tif), str(out_swc)], check=True)
    tqdm.write(f"[{method}] {tif.stem} → {out_swc}  ({_fmt(time.time() - t0)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["ours", "vaa3d"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=str, default=None,
                        help="Comma-separated sample stems, e.g. neuron2,neuron4")
    args = parser.parse_args()

    if args.samples:
        stems = [s.strip() for s in args.samples.split(",")]
        tifs = [DATA_DIR / f"{s}.tif" for s in stems]
        tifs = [t for t in tifs if t.exists()]
        if not tifs:
            sys.exit(f"No matching .tif files found")
    else:
        tifs = sorted(DATA_DIR.glob("*.tif"))
        if not tifs:
            sys.exit(f"No .tif files in {DATA_DIR}")
        if args.limit:
            tifs = tifs[:args.limit]

    print(f"Method: {args.method}  |  Images: {len(tifs)}")

    batch_start = time.time()
    with tqdm(tifs, unit="sample",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
        for tif in pbar:
            pbar.set_postfix_str(tif.stem)
            out_swc = RESULTS_DIR / args.method / f"{tif.stem}.swc"
            if out_swc.exists():
                tqdm.write(f"  skip (exists): {tif.stem}")
                continue
            try:
                if args.method == "ours":
                    run_ours(tif, out_swc, sample_label=tif.stem)
                else:
                    run_shell(args.method, tif, out_swc)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"  ERROR on {tif.stem}: {e}", file=sys.stderr)

    print(f"Batch done.  Total: {_fmt(time.time() - batch_start)}")


if __name__ == "__main__":
    main()
