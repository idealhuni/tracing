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


TARGET_VOXEL_XY_UM = 0.55  # keep post-downsample XY voxel below this threshold


def _compute_downsample_xy(voxel_xy: float) -> int:
    return max(1, int(TARGET_VOXEL_XY_UM / voxel_xy))


def _slab_size_for(tif: Path, downsample_xy: int, base: int = 48, ref_xy: int = 1024) -> int:
    """Scale slab size based on effective XY size after step0 downsampling."""
    import tifffile as _tf
    with _tf.TiffFile(str(tif)) as t:
        page = t.pages[0]
        y, x = page.shape[0] // downsample_xy, page.shape[1] // downsample_xy
    ratio = (ref_xy * ref_xy) / (y * x)
    return max(4, min(base, int(base * ratio)))


def run_ours(tif: Path, out_swc: Path, sample_label: str = ""):
    out_swc.parent.mkdir(parents=True, exist_ok=True)
    work_dir = ROOT / "methods" / "ours"
    out_dir  = work_dir / "output" / tif.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    voxel_xy, voxel_z = parse_voxel_sizes(tif.stem)
    downsample_xy = _compute_downsample_xy(voxel_xy)
    slab_size = _slab_size_for(tif, downsample_xy)
    tqdm.write(f"  {tif.stem}: voxel_xy={voxel_xy:.4f}µm  ds_xy={downsample_xy}  "
               f"→ viso={voxel_xy*downsample_xy:.4f}µm  slab={slab_size}")

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
                          "--downsample-xy", str(downsample_xy)]
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
