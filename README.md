# COMP7015 Group Project – Topic 1: Plant Disease Classification

Tomato leaf images from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) are classified into **healthy, early blight, leaf mold and spider mites** (two-spotted spider mite damage). We compare a custom CNN with Squeeze-and-Excitation attention, a fine-tuned ResNet18 and a fine-tuned ViT-B/16. Then we investigate how robust the models are to the background, colour, lighting and image quality, and to photos taken outside the lab.

## Setup

```bash
uv sync                      # Python 3.12 environment from pyproject.toml / uv.lock
source .venv/bin/activate
jupyter lab
```

Without uv: `pip install torch torchvision torchinfo scikit-learn scipy pandas matplotlib jupyter`.

The code runs on an NVIDIA GPU (CUDA, with mixed precision), on Apple silicon (MPS) and on the CPU; the device is chosen automatically.

The dataset is downloaded by the first notebook: a sparse git checkout of the four tomato folders of `raw/color` and `raw/segmented`, plus `leaf-map.json`. Section 19.2 also downloads the matching folders of [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset). It is skipped if the download fails.

## Running

Run the notebooks in order, from the repository root:

| Notebook | Sections | Content |
|---|---|---|
| `01_data.ipynb` | 1–6 | configuration, download, leaf grouping, image cache, leaf-group split and leakage audit, background-only baseline, augmentation |
| `02_cnn_resnet.ipynb` | 7–9 | custom CNN and ResNet18, training engine, tuning and ablations, selection with all seeds |
| `03_vit.ipynb` | 10 | ViT-B/16 fine-tuning strategies (head only, last two blocks, full) and augmentation ablation, selection |
| `04_test_evaluation.ipynb` | 11–13 | test results of the final models over 3 seeds, compute trade-offs (inference and training cost), error analysis, Grad-CAM |
| `05_robustness.ipynb` | 14–20 | robustness to background removal, grayscale, lighting, blur and noise; ViT retrained on grayscale and background-removed photos; the same leaf on other backgrounds; leaf attention; countermeasure (random backgrounds); left-out photo series and PlantDoc field photos; confidence and abstention |
| `06_discussion.ipynb` | 21–22 | all key tables in one place, discussion |

Every training run is saved under `outputs/runs/<run>__seed<seed>/`: the best checkpoint, the history (including training time, peak memory and whether it stopped early) and the configuration. Re-running a notebook loads finished runs instead of retraining them, and it reuses cached predictions. So an interrupted run can simply be resumed, and every notebook can be re-run on its own once the earlier ones have run. Notebooks 02 and 03 write the selected models to `outputs/final_models.json`.

A GPU is recommended: the full run trains about 70 models (21 of them ViTs). For a quick end-to-end test, write to a separate output folder:

```bash
EPOCHS_CNN=1 EPOCHS_RESNET=2 EPOCHS_VIT=4 SEEDS=62 OUT_DIR=./outputs_quick jupyter lab
```

`EPOCHS_VIT` must be at least 4, because the ViT trains only its head for the first 3 epochs.

## Method

- **Data.** One photo series per class with official leaf ids (`RS_HL`, `RS_Erly.B`, `Crnl_L.Mold`, `Com.G_SpM_FL`). The official leaf ids of every class are split 70/15/15 with a fixed seed, and the audit checks that no leaf id crosses splits. The healthy series `GH_HL` has no leaf ids and is used only as a left-out series test.
- **Preprocessing.** Every photo is resized to 224×224 once and cached. Training augmentation: flips, 90° rotations, ±10% brightness/contrast.
- **ViT** (Zhouyang's protocol). 3 epochs train only the new head (lr 1e-3). Then the head (lr 1e-4) and the unfrozen backbone (lr 1e-5) train for up to 12 more epochs. AdamW with weight decay 0.01, constant learning rate, batch 16, no class weights. Three strategies: head only, partial (last 2 blocks + final LayerNorm) and full fine-tuning.
- **Early stopping.** Every run stops when the validation macro-F1 has not improved for `patience` epochs of its last training stage (5 epochs for every model, `patience` of the recipes in notebooks 02 and 03); the checkpoint of the best epoch is kept.
- **Uncertainty.** Every model is trained with seeds 62, 63 and 64 on one fixed split. Results are the mean ± sample standard deviation over the seeds, and models trained with the same seeds are compared seed by seed.
- **Compute.** Parameters, checkpoint size and inference latency; trainable parameters, epochs trained, training time without validation, and peak memory during training (exact on CUDA, sampled on MPS).

## Configuration

The settings live in `plantdisease/config.py`, in four dictionaries:

- `PATHS_CONFIG`: where the data, the image cache and the results live;
- `DATA_CONFIG`: the PlantVillage protocol (classes, training series, image size, split seed and fractions, loader workers);
- `PLANTDOC_CONFIG`: the PlantDoc field photos (folder, repository URL, folder of each class);
- `TRAIN_CONFIG`: training seeds and the background-swap probability. The maximum epochs and the early-stopping patience are part of each model recipe (notebooks 02 and 03).

The model recipes (learning rate, batch size, freezing, epochs, ...) are defined in notebooks 02 and 03, right before the experiments are run; the settings a recipe can have are defined by `RunConfig` in `plantdisease/recipe.py`. All settings are saved to `OUT_DIR/config.json`. These environment variables override the defaults:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./PlantVillage-Dataset` | PlantVillage checkout |
| `CACHE_DIR` | `./cache` | decoded and resized images (uint8 tensors) |
| `OUT_DIR` | `./outputs` | runs, tables (`.csv`) and figures (`.png`) |
| `FIELD_ROOT` | `./PlantDoc-Dataset` | PlantDoc checkout (Section 19.2) |
| `SEEDS` | `62,63,64` | training seeds |
| `EPOCHS_CNN`, `EPOCHS_RESNET`, `EPOCHS_VIT` | `50`, `16`, `15` | maximum training epochs |
| `PATIENCE_CNN`, `PATIENCE_RESNET`, `PATIENCE_VIT` | `5`, `5`, `5` | early stopping: epochs without a better validation macro-F1 (0 = off) |
| `NUM_WORKERS` | `2` | data loader workers |

Runs are deterministic: fixed seeds, deterministic algorithms, and a split seed that is separate from the training seeds.

## Code

The notebooks show the steps and the results. The shared code is in the `plantdisease/` package:

| Module | Content |
|---|---|
| `config.py` | settings (`PATHS_CONFIG`, `DATA_CONFIG`, `PLANTDOC_CONFIG`, `TRAIN_CONFIG`) |
| `device.py` | device, mixed precision, determinism, `set_seed`, `synchronize` |
| `leafmap.py` | leaf ids from `leaf-map.json` (also used by `database/build_db.py`) |
| `data.py` | download, leaf grouping, image cache, split, segmented and grayscale versions, transforms, loaders, `prepare()` |
| `backgrounds.py` | leaf masks, test variants, same-leaf background tests, background-swap dataset |
| `models.py` | custom CNN with SE attention, ResNet18, ViT-B/16, freezing for the fine-tuning strategies |
| `recipe.py` | `RunConfig`, the pydantic model of one training run (which settings exist, their types and allowed values) |
| `experiments.py` | registry of the experiments defined in notebooks 02 and 03 (`outputs/experiments.json`), derived runs of Sections 15 and 18, the selected models |
| `train.py` | training engine (AdamW, schedules, two-stage fine-tuning with separate head/backbone learning rates, checkpoints, training cost), cached predictions |
| `evaluate.py` | macro-F1, mean ± sd over seeds, same-seed comparisons, stable errors, Grad-CAM, latency and training cost, calibration |
| `plots.py` | plotting helpers |

`plant_disease_classification2.ipynb` and `plant_disease_classification3_julia.ipynb` are earlier single-notebook versions with different classes (bacterial spot and late blight instead of leaf mold and spider mites); they are kept for reference.

## Acknowledgements

- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/) (models, pre-trained ImageNet weights of ResNet18 and ViT-B/16, transforms)
- [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/)
- PlantVillage dataset: Hughes & Salathé (2015); Mohanty, Hughes & Salathé (2016), <https://github.com/spMohanty/PlantVillage-Dataset>
- PlantDoc dataset: Singh et al. (2020), <https://github.com/pratikkayal/PlantDoc-Dataset>
- Background bias in PlantVillage: Noyan (2022)
- Squeeze-and-Excitation: Hu et al. (2018); Grad-CAM: Selvaraju et al. (2017); ViT: Dosovitskiy et al. (2021)
- ViT fine-tuning protocol and input ablation: Zhouyang (team member), `docs/zhouyang/`
