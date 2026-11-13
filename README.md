# COMP7015 Group Project – Topic 1: Plant Disease Classification

We classify [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) tomato leaves as **healthy, bacterial spot, early blight or late blight** with three models: a custom CNN with Squeeze-and-Excitation attention, a fine-tuned ResNet18 and a fine-tuned ViT-B/16. We then test how robust they are to the background, colour, lighting, image quality and photos taken outside the lab. The written report is in `report/`.

## Setup

```bash
uv sync                      # Python 3.12 environment from pyproject.toml / uv.lock
source .venv/bin/activate
jupyter lab
```

Without uv: `pip install torch torchvision torchinfo grad-cam pydantic scikit-learn scipy pandas matplotlib jupyter`.

The device is chosen automatically: NVIDIA GPU (CUDA, mixed precision), Apple silicon (MPS) or CPU. A GPU is recommended, because a full run trains 62 models (24 of them ViTs).

The data is downloaded by the notebooks. PlantVillage is a `git clone --depth 1` (about 2 GB; git is required), skipped if `PlantVillage-Dataset/raw/color` exists. The matching [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) folders are downloaded for the field-photo test, which is skipped if that download fails.

## Running

Run the two notebooks in order, from the repository root:

| Notebook | Sections | Content |
|---|---|---|
| `01_classification.ipynb` | 1–14 | data and leaf-group split (1–6), custom CNN and ResNet18 (7–9), ViT-B/16 (10), test results, compute and heat maps (11–13), discussion (14) |
| `02_robustness.ipynb` | 15–22 | background, colour, lighting and quality changes; retraining on other inputs; same leaf on other backgrounds; where the models look; random-background countermeasure; PlantDoc field photos; confidence (15–21), discussion (22) |

Every training run is saved under `outputs/runs/<run>__seed<seed>/` (best checkpoint, history, configuration), and predictions are cached. Re-running a notebook loads finished runs instead of retraining them, so an interrupted run can be resumed, and `02_robustness.ipynb` can be re-run on its own once `01_classification.ipynb` has run.

## Configuration

Settings live in `plantdisease/config.py`, and environment variables of the same names override them (for example `OUT_DIR`, `SEEDS`, `EPOCHS_*`). The model recipes are in `plantdisease/experiments.py`: `BASE_CNN`, `BASE_RESNET` and `BASE_VIT`, and every experiment is a base recipe with one change. Runs are deterministic: fixed seeds, deterministic algorithms, and a split seed separate from the training seeds.

## Code

The notebooks call the `plantdisease/` package:

| Module | Content |
|---|---|
| `config.py`, `device.py` | settings; device, mixed precision and determinism |
| `data/` | downloads, leaf ids, image cache, leaf-group split, loaders and augmentation, background tests (`prepare()` runs the whole data protocol) |
| `models.py` | SE-CNN, ResNet18, ViT-B/16 and the freezing for fine-tuning |
| `experiments.py` | the recipe of every run (`RunConfig`, `config_of(name)`) and the selected models |
| `train.py` | `run(name, data)`: training with every seed, early stopping, checkpoints |
| `predictions.py`, `evaluate.py` | cached predictions; metrics, seed-by-seed comparisons, compute cost, calibration |
| `gradcam.py` | heat maps (Grad-CAM for the CNNs, gradient-weighted attention for the ViT) and the leaf share |
| `plots.py` | plotting helpers |

## Declaration of AI usage

We used Claude Code (Anthropic) to refactor code, draft helper scripts, debug, and edit the wording of this README and the report. The research questions, experimental design, experiments and conclusions are our own. All reported numbers come from our own training runs. Library code is credited below.

## Acknowledgements

**Libraries.**
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/): models, ImageNet weights of ResNet18 and ViT-B/16, transforms
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam): Grad-CAM
- [torchinfo](https://github.com/TylerYep/torchinfo): model summaries
- [scikit-learn](https://scikit-learn.org/): metrics, random forest
- [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/): numerics, tables and plots
- [Pillow](https://python-pillow.org/): image decoding
- [pydantic](https://docs.pydantic.dev/): run recipes
- [Jupyter](https://jupyter.org/): notebooks
- [uv](https://docs.astral.sh/uv/): environment

**Data.**
- PlantVillage: D. P. Hughes, M. Salathé. *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv, 2015. [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)
- PlantVillage: S. P. Mohanty, D. P. Hughes, M. Salathé. *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science 7:1419, 2016. [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)
- PlantDoc: D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, N. Batra. *PlantDoc: a dataset for visual plant disease detection.* CoDS-COMAD, 2020. [arXiv:1911.10317](https://arxiv.org/abs/1911.10317)
- Background bias in PlantVillage: M. A. Noyan. *Uncovering bias in the PlantVillage dataset.* arXiv, 2022. [arXiv:2206.04374](https://arxiv.org/abs/2206.04374)

**Models and methods.**
- ResNet: K. He, X. Zhang, S. Ren, J. Sun. *Deep residual learning for image recognition.* CVPR, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- ViT: A. Dosovitskiy et al. *An image is worth 16x16 words: transformers for image recognition at scale.* ICLR, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Grad-CAM: R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra. *Grad-CAM: visual explanations from deep networks via gradient-based localization.* ICCV, 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Gradient-weighted attention: H. Chefer, S. Gur, L. Wolf. *Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers.* ICCV, 2021. [arXiv:2103.15679](https://arxiv.org/abs/2103.15679)
