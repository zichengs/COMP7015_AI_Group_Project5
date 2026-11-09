"""Plant disease classification on PlantVillage (COMP7015 group project, Topic 1).

Shared code of the notebooks 01-05:

- ``config``      settings (PATHS_CONFIG, DATA_CONFIG, PLANTDOC_CONFIG, TRAIN_CONFIG)
- ``device``      device, mixed precision, determinism, set_seed, synchronize
- ``data``        download, leaf grouping, image cache, near-duplicates, split, loaders
- ``backgrounds`` leaf masks, test variants, same-leaf background tests, background swap
- ``models``      custom CNN with SE attention, ResNet18, ViT-B/16
- ``recipe``      ``RunConfig``: the settings of one training run (pydantic model)
- ``experiments`` registry of the experiments defined in notebooks 02 and 03, derived runs (Sections 15, 18), selected models
- ``train``       training engine and cached predictions
- ``evaluate``    metrics, bootstrap, paired comparisons, Grad-CAM, compute
- ``plots``       plotting helpers
"""
