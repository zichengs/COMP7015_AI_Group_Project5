"""Lab smoke test: run both notebooks top to bottom on a tiny subset, in minutes instead of hours.

Usage (from the repository root, in the environment that will run the notebooks):

    python scripts/smoke_test.py            # or: uv run python scripts/smoke_test.py
    python scripts/smoke_test.py --keep     # keep outputs_smoke/ and cache_smoke/ for inspection

Every cell runs, including training, Grad-CAM / attention rollout, the robustness tests and the PlantDoc
download. Settings are tiny: MAX_LEAVES leaves per class, one seed, minimal epochs. Results go to
outputs_smoke/ and cache_smoke/; the real outputs/ and cache/ are never touched, and the notebooks
in the repository are not modified (executed copies are saved to outputs_smoke/executed/).
Exit code 0 = every notebook passed.
"""

# Packages are imported inside the functions on purpose: preflight must report a missing
# package before anything imports it.
# ruff: noqa: PLC0415

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = {"01": "01_classification.ipynb", "02": "02_robustness.ipynb"}
SMOKE_OUT, SMOKE_CACHE = "outputs_smoke", "cache_smoke"

# Run in a separate process by preflight(): packages, GPU, pre-trained weights. Exit code 1 = a
# package is missing.
PACKAGE_CHECK = r"""
import importlib, importlib.metadata, os, sys, urllib.request

def report(status, message):
    print(f"  [{status:^4}] {message}", flush=True)

# module -> distribution name (for the version)
PACKAGES = {
    "torch": "torch", "torchvision": "torchvision", "torchinfo": "torchinfo",
    "sklearn": "scikit-learn", "scipy": "scipy", "pandas": "pandas",
    "matplotlib": "matplotlib", "pydantic": "pydantic", "PIL": "pillow",
    "pytorch_grad_cam": "grad-cam", "nbformat": "nbformat", "nbclient": "nbclient",
    "ipykernel": "ipykernel",
}
missing = False
for module, dist in PACKAGES.items():
    try:
        importlib.import_module(module)
        report("OK", f"{dist} {importlib.metadata.version(dist)}")
    except Exception as e:
        report("FAIL", f"{dist}: {e}")
        missing = True
if missing:
    sys.exit(1)

import torch
from torchvision import models

if torch.cuda.is_available():
    report("OK", f"CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    report("WARN", "no CUDA GPU: everything runs on the CPU (much slower)")

# ImageNet weights of ResNet18 and ViT-B/16: cached, or downloadable on first use
for weights in (models.ResNet18_Weights.IMAGENET1K_V1, models.ViT_B_16_Weights.IMAGENET1K_V1):
    file = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))
    if os.path.exists(file):
        report("OK", f"pre-trained weights cached: {os.path.basename(file)}")
        continue
    try:
        urllib.request.urlopen(urllib.request.Request(weights.url, method="HEAD"), timeout=10)
        report("OK", f"pre-trained weights downloadable: {os.path.basename(file)}")
    except Exception as e:
        report("WARN", f"cannot download {weights.url} ({e}): Sections 7-10 need it")
"""


def report(status, message):
    # ASCII only: lab consoles may not print Unicode check marks
    print(f"  [{status:^4}] {message}", flush=True)


# ----------------------------------------------------------------------------
# Preflight: everything the notebooks need, checked in seconds
# ----------------------------------------------------------------------------
def preflight():
    print("Preflight", flush=True)
    ok = True
    report("OK", f"Python {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 12):  # noqa: UP036 (the lab interpreter may be older)
        report("WARN", "the project targets Python >= 3.12")

    # packages, GPU and pre-trained weights are checked in a separate process, so that this process
    # never loads torch / CUDA (it would hold memory while the notebooks run)
    check = subprocess.run([sys.executable, "-c", PACKAGE_CHECK], check=False)  # noqa: S603
    if check.returncode:
        print(
            "\nMissing packages: run `uv sync`, or `pip install` the packages listed in README.md."
        )
        return False

    report(
        "OK" if shutil.which("git") else "FAIL",
        "git on PATH (needed to clone PlantVillage)",
    )
    data_root = os.environ.get("DATA_ROOT", "./PlantVillage-Dataset")
    if os.path.isdir(os.path.join(data_root, "raw", "color")):
        report("OK", f"PlantVillage found in {data_root}")
    else:
        report(
            "WARN", f"PlantVillage not in {data_root}: Section 2 clones it (about 2 GB)"
        )
        ok = ok and bool(shutil.which("git"))

    try:
        urllib.request.urlopen("https://github.com", timeout=10)
        report("OK", "reachable: https://github.com (PlantVillage, PlantDoc)")
    except Exception as e:
        report(
            "WARN", f"github.com not reachable ({e}): the dataset downloads may fail"
        )

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    report("OK" if free_gb > 5 else "WARN", f"free disk space: {free_gb:.0f} GB")
    try:
        os.makedirs(SMOKE_OUT, exist_ok=True)
        report("OK", f"write access to {os.path.abspath(SMOKE_OUT)}")
    except OSError as e:
        report("FAIL", f"cannot write {SMOKE_OUT}: {e}")
        ok = False
    return ok


# ----------------------------------------------------------------------------
# Running a notebook
# ----------------------------------------------------------------------------
def make_kernel_manager():
    # Start the kernel with this interpreter, whatever "python3" kernel the machine has registered.
    from jupyter_client.kernelspec import KernelSpec
    from jupyter_client.manager import KernelManager

    km = KernelManager()
    km._kernel_spec = KernelSpec(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="smoke test",
        language="python",
    )
    return km


def failing_cell(nb):
    # (index, section heading, cell) of the first cell with an error output
    heading = "(before the first heading)"
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                if line.startswith("#"):
                    heading = line.lstrip("# ").strip()
                    break
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return i, heading, cell
    return None


def run_notebook(name, timeout):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(os.path.join(ROOT, name), as_version=4)
    client = NotebookClient(
        nb,
        km=make_kernel_manager(),
        timeout=timeout,
        resources={"metadata": {"path": ROOT}},
    )
    print(f"\n=== {name} ({len(nb.cells)} cells) ===")
    tic = time.perf_counter()
    error = None
    try:
        client.execute()
    except Exception as e:
        error = e
    finally:
        # nbclient leaves a kernel it did not create running: stop it, so that its memory (images,
        # models, GPU) is free before the next notebook starts
        if client.km is not None and client.km.has_kernel:
            client.km.shutdown_kernel(now=True)
    minutes = (time.perf_counter() - tic) / 60

    executed_dir = os.path.join(SMOKE_OUT, "executed")
    os.makedirs(executed_dir, exist_ok=True)
    nbformat.write(nb, os.path.join(executed_dir, name))

    if error is None:
        print(f"PASS  {name} in {minutes:.1f} min")
        return True
    print(f"FAIL  {name} after {minutes:.1f} min")
    found = failing_cell(nb)
    if found:
        i, heading, cell = found
        print(f"  failing cell {i}, section: {heading}")
        print("  " + "\n  ".join(cell.source.splitlines()[:8]))
        for output in cell.outputs:
            if output.get("output_type") == "error":
                print(f"  {output['ename']}: {output['evalue']}")
    else:
        print(f"  {type(error).__name__}: {str(error)[:2000]}")
    print(f"  executed copy with all outputs: {os.path.join(executed_dir, name)}")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--leaves", type=int, default=12, help="leaves per class (default 12)"
    )
    parser.add_argument(
        "--only",
        choices=sorted(NOTEBOOKS),
        help="run one notebook (02 needs a kept 01 run)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep outputs_smoke/ and cache_smoke/"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="seconds per cell (default 3600)"
    )
    args = parser.parse_args()
    # show progress at once, also when the output goes to a file or pipe
    sys.stdout.reconfigure(line_buffering=True)

    os.chdir(ROOT)
    # tiny settings, separate output and cache folders (read by plantdisease/config.py in the kernels)
    os.environ.update(
        {
            "OUT_DIR": f"./{SMOKE_OUT}",
            "CACHE_DIR": f"./{SMOKE_CACHE}",
            "MAX_LEAVES": str(args.leaves),
            "SEEDS": "62",
            "EPOCHS_CNN": "1",
            "EPOCHS_RESNET": "2",
            "EPOCHS_VIT": "4",  # minimum: the ViT trains only its head for 3 epochs
        }
    )
    if not preflight():
        print("\nPreflight failed: fix the items marked FAIL above.")
        return 1

    names = [NOTEBOOKS[args.only]] if args.only else list(NOTEBOOKS.values())
    tic = time.perf_counter()
    results = {}
    for name in names:
        results[name] = run_notebook(name, args.timeout)
        if not results[name]:
            break  # 02 depends on the runs and final_models.json of 01

    print(f"\n=== Summary ({(time.perf_counter() - tic) / 60:.1f} min) ===")
    for name in names:
        status = {True: "PASS", False: "FAIL"}.get(results.get(name), "SKIPPED")
        print(f"  {status:7} {name}")
    passed = len(results) == len(names) and all(results.values())
    if passed and not args.keep:
        for folder in (SMOKE_OUT, SMOKE_CACHE):
            shutil.rmtree(folder, ignore_errors=True)
        print(f"  removed {SMOKE_OUT}/ and {SMOKE_CACHE}/ (use --keep to inspect them)")
    elif not passed:
        print(f"  kept {SMOKE_OUT}/ and {SMOKE_CACHE}/ for inspection")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
