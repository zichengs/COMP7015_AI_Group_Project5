# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP7015 group project: classify PlantVillage tomato leaves (healthy, bacterial spot, early blight, late blight) with a custom SE-CNN, a fine-tuned ResNet18 and a fine-tuned ViT-B/16. Then test how robust the models are to the background, colour, lighting, image quality and field photos. The work is two notebooks backed by the `plantdisease/` package. The written report is in `report/` (Typst).

## Commands

```bash
uv sync                                   # Python 3.12 env from pyproject.toml / uv.lock (CUDA wheels on win/linux)
uv add <pkg>                              # add a dependency (updates pyproject.toml and uv.lock)
uvx ruff check plantdisease               # lint (config in ruff.toml)
uvx ruff format plantdisease              # format

# execute a notebook headlessly, saving outputs in place (run from the repo root)
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 01_classification.ipynb

# quick end-to-end smoke run into a separate output folder (EPOCHS_VIT must be >= 4: 3 head-only epochs)
EPOCHS_CNN=1 EPOCHS_RESNET=2 EPOCHS_VIT=4 SEEDS=62 OUT_DIR=./outputs_quick uv run jupyter lab

typst compile --root . report/report.typ report/report.pdf
```

There is no test suite. To check a change, run the affected function in a short `uv run python -c "..."` script on a few images or checkpoints, or execute the affected notebook.

Ruff baseline: `uvx ruff check plantdisease` already reports some errors in files you haven't touched (`download.py`, `leafmap.py`, `train.py`). Don't add new ones. RUF003 flags `×` in comments, so write `x` instead.

## Notebook pipeline

Run the two notebooks in order:
- `01_classification.ipynb`: Part 1, data (Sections 1–6); Part 2, CNN and ResNet18 (7–9); Part 3, ViT (10); Part 4, test evaluation, compute and Grad-CAM / Score-CAM (11–13); Part 5, summary tables (21) and discussion 22.1–22.3.
- `02_robustness.ipynb`: Sections 14–20, then its own summary tables (21) and discussion 22.4–22.6 (robustness, limitations, conclusion).

Section numbers continue across both notebooks, and the code comments refer to them ("Section 18"). Refer to sections, not notebook numbers. Notebooks only call the package and plot; put logic in `plantdisease/`. `01_classification` runs in one kernel, so watch for variable names reused across parts (`images` is `data.images`, for example).

The notebooks pass work through files in `OUT_DIR` (default `./outputs`):
- Sections 9.3 and 10.3 write the selected models to `final_models.json` via `save_selected`. Section 11 and `02_robustness` read them with `load_final_models()`.
- Both notebooks write their tables (`.csv`) and figures (`.png`) to `OUT_DIR`. The Section 21 cells only read those CSVs.

**Cost:** with all runs in `outputs/runs/`, re-executing either notebook only loads them. On a fresh output folder, `01_classification` trains 38 runs and `02_robustness` 24 more. Check `outputs/runs/` before executing a notebook, and ask before starting hours of GPU training.

## Architecture (the parts that span several files)

- **Data protocol:** `data/dataset.py: prepare()` is the one entry point that every notebook calls. It filters to photos with an official leaf id (`records.load_records`) and loads or builds the uint8 image cache (`images.build_cache`). It then makes the leaf-group 70/15/15 split with a leakage assert (`records.make_split`). It returns a `Data` object with indices, labels, `class_weights` and lazily built segmented / grayscale / leaf-mask versions. The split seed (`DATA_CONFIG["split_seed"]`) is separate from the training seeds, so every model shares one split.
- **Runs are identified by name.** `experiments.py` defines `RunConfig`, which is pydantic, frozen and rejects extra fields. It also defines the base recipes `BASE_CNN` / `BASE_RESNET` / `BASE_VIT`, each experiment as `base.variant(...)`, and derived runs named `"<experiment> + <suffix>"`. The suffixes are `INPUT_TRAINING` (Section 15) and `COUNTERMEASURES` (Section 18). `config_of(name)` resolves any name to its recipe. To add an experiment, add an entry to one of these dicts; don't build configs in notebooks.
- **Training:** `train.py: run(name, data, seeds)` trains or loads one run per seed into `outputs/runs/<safe_name>__seed<seed>/`, saving `best.pt` and `history.json`. A run is reused only if both the saved config and the data fingerprint (`records.data_fingerprint`) match. So changing a recipe or the training photos retrains automatically, and changing anything else does not. Pre-trained models use two stages: `head_epochs` with only the head trainable, then fine-tuning chosen by `trainable_blocks` in `models.set_trainable`. Early stopping only counts epochs of the last stage.
- **Predictions:** `predictions.py: predict_probs / predict_labels(names, {set_name: uint8 images})` returns `probs[set][model]` with shape `(seeds, N, K)`, or labels with shape `(seeds, N)`. The seed axis follows `TRAIN_CONFIG["seeds"]`. Results are cached as `runs/<run>/preds/<set>_<md5 of images>.npy`, and a retrain deletes that folder. `evaluate.py` works on this layout: `seed_scores`, `mean_sd`, `compare_seeds` (same-seed paired comparison), `stable_errors`.
- **Explanations:** `gradcam.py` connects the pytorch-grad-cam library (`GradCAM`, `ScoreCAM`) to our models:
  - It picks the target layer per architecture. For the ViT this is `encoder.layers[-1].ln_1`, with `vit_reshape` turning the 196 patch tokens into a 14x14 grid.
  - It holds the leaf-share analysis (`leaf_attention`, `leaf_share_table`). Per-image results are cached in `runs/<run>/preds/leafshare-lib_*.npy`.
  - If you change how the maps are computed, change that cache prefix too, or stale cached values will be reused.
- **Backgrounds** (`data/backgrounds.py`): leaf masks come from PlantVillage's segmented photos. They are used for the Section 14 test variants, the same-leaf / other-background tests (Section 16) and `BackgroundSwapDataset` (Section 18). `train.train_dataset` switches to that dataset when `cfg.background_swap > 0`.
- **Determinism:** `device.py` sets `CUBLAS_WORKSPACE_CONFIG` before importing torch and enables deterministic algorithms. It picks CUDA (with AMP), then MPS, then CPU. Use `set_seed` and `synchronize` (needed for correct timings) from there.

## Conventions

- **Leave `pseudocode/` and `database/` alone.** They are side material and don't matter for the project. Don't update them when code, classes or notebooks change, and don't read them for context, unless the user asks for it explicitly.

- **Evaluation protocol:** tune and select on the validation set only. The test set is evaluated once, after selection. Report results as mean ± sample sd over seeds 62/63/64, and compare models seed by seed ("higher in k of 3 seeds").
- **Settings:** they live in the four dicts in `config.py` (`PATHS_CONFIG`, `DATA_CONFIG`, `PLANTDOC_CONFIG`, `TRAIN_CONFIG`), and environment variables override them (`OUT_DIR`, `DATA_ROOT`, `CACHE_DIR`, `SEEDS`, `EPOCHS_*`, `PATIENCE_*`, `NUM_WORKERS`). Build output paths with `out_path(name)` and `run_path(name, seed)`.
- **Code style:** function bodies explain themselves with short `#` comments, not docstrings. Module docstrings name the notebook sections they serve. Keep that style.
- **Git-ignored and regenerated by the code:** `PlantVillage-Dataset/` (git clone, about 2 GB), `PlantDoc-Dataset/`, `cache/`, `outputs/runs/`, `database/*.db` and `docs/` (course material and archived notebooks). `outputs/*.csv` and `outputs/*.png` are committed.
