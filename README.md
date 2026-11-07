# COMP7015 Group Project – Topic 1: Plant Disease Classification

Tomato leaf images from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) are classified into **healthy, bacterial spot, early blight and late blight**. We compare a custom CNN with Squeeze-and-Excitation attention, a fine-tuned ResNet18 and a fine-tuned ViT-B/16. Then we ask how much the models rely on the photo background, and whether training with random backgrounds helps.

## Setup

```bash
uv sync                      # Python 3.12 environment from pyproject.toml / uv.lock
source .venv/bin/activate
jupyter lab
```

Without uv: `pip install torch torchvision scikit-learn scipy pandas matplotlib jupyter`.

The dataset is downloaded by the first notebook: a sparse git checkout of the four tomato folders of `raw/color` and `raw/segmented`, plus `leaf-map.json`. Section 15.2 also downloads the matching folders of [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset). It is skipped if the download fails.

## Running

Run the notebooks in order, from the repository root:

| Notebook | Sections | Content |
|---|---|---|
| `01_data.ipynb` | 1–6 | configuration, download, leaf grouping, image cache, near-duplicates, leaf-group split and leakage audit, background-only baseline, augmentation |
| `02_models_and_tuning.ipynb` | 7–9 | models, training engine, tuning and ablations, model selection with all seeds |
| `03_test_evaluation.ipynb` | 10–12 | test results over 3 seeds, bootstrap intervals, paired comparisons, compute trade-offs, error analysis, Grad-CAM |
| `04_background_bias.ipynb` | 13–14 | robustness, the same leaf on different backgrounds, leaf attention, countermeasure (random backgrounds) |
| `05_generalisation.ipynb` | 15–17 | left-out photo series, PlantDoc field photos, confidence and abstention, discussion |

Every training run is saved under `outputs/runs/<run>__seed<seed>/`: the best checkpoint, the history and the configuration. Re-running a notebook loads finished runs instead of retraining them, and it reuses cached test predictions. So an interrupted run can simply be resumed, and notebooks 03–05 can be re-run on their own. Notebook 02 writes the selected models to `outputs/final_models.json`.

A GPU is recommended: the full run trains about 40 models. For a quick end-to-end test, write to a separate output folder:

```bash
EPOCHS_CNN=1 EPOCHS_RESNET=2 EPOCHS_VIT=2 SEEDS=42 OUT_DIR=./outputs_quick jupyter lab
```

## Configuration

All settings live in one `CFG` dictionary in `plantdisease/config.py`. The configuration is saved to `OUT_DIR/config.json`. These environment variables override the defaults:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./PlantVillage-Dataset` | PlantVillage checkout |
| `CACHE_DIR` | `./cache` | decoded and resized images (uint8 tensors) |
| `OUT_DIR` | `./outputs` | runs, tables (`.csv`) and figures (`.png`) |
| `FIELD_ROOT` | `./PlantDoc-Dataset` | PlantDoc checkout (Section 15.2) |
| `SEEDS` | `42,43,44` | training seeds of the final models |
| `EPOCHS_CNN`, `EPOCHS_RESNET`, `EPOCHS_VIT` | `50`, `16`, `12` | training epochs |
| `NUM_WORKERS` | 4 on Linux, else 0 | data loader workers |

Runs are deterministic: fixed seeds, deterministic algorithms, and a split seed that is separate from the training seeds.

## Code

The notebooks show the steps and the results. The shared code is in the `plantdisease/` package:

| Module | Content |
|---|---|
| `config.py` | `CFG`, seeds, device |
| `data.py` | download, leaf grouping, image cache, near-duplicate merging, split, transforms, loaders, `prepare()` |
| `backgrounds.py` | leaf masks, test variants, same-leaf background tests, background-swap dataset |
| `models.py` | custom CNN with SE attention, ResNet18, ViT-B/16, two-stage freezing |
| `experiments.py` | the configurations of Sections 9 and 14, the selected models |
| `train.py` | training engine (AdamW, cosine schedule, two-stage fine-tuning, checkpoints), cached predictions |
| `evaluate.py` | macro-F1, bootstrap over leaf groups, paired comparisons, Grad-CAM, latency, calibration |
| `plots.py` | plotting helpers |

## Acknowledgements

- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/) (models, pre-trained ImageNet weights of ResNet18 and ViT-B/16, transforms)
- [scikit-learn](https://scikit-learn.org/), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/)
- PlantVillage dataset: Hughes & Salathé (2015); Mohanty, Hughes & Salathé (2016), <https://github.com/spMohanty/PlantVillage-Dataset>
- PlantDoc dataset: Singh et al. (2020), <https://github.com/pratikkayal/PlantDoc-Dataset>
- Background bias in PlantVillage: Noyan (2022)
- Squeeze-and-Excitation: Hu et al. (2018); Grad-CAM: Selvaraju et al. (2017); ViT: Dosovitskiy et al. (2021)
