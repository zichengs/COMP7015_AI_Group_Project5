"""Plant disease classification on PlantVillage (COMP7015 group project, Topic 1).

Shared code of the notebooks 01-06:

- ``config``        settings (PATHS_CONFIG, DATA_CONFIG, PLANTDOC_CONFIG, TRAIN_CONFIG)
- ``device``        device, mixed precision, determinism, set_seed, synchronize
- ``data/``
    - ``download``     PlantVillage (git clone) and PlantDoc field photos
    - ``leafmap``      leaf ids from leaf-map.json
    - ``records``      photo table, leaf-group split with leakage audit, data fingerprint
    - ``images``       image cache, segmented and grayscale versions
    - ``loaders``      preprocessing, augmentation, datasets and loaders
    - ``backgrounds``  leaf masks, test variants, background tests, background swap
    - ``dataset``      the Data object (``prepare()``)
- ``models``        custom CNN with SE attention, ResNet18, ViT-B/16
- ``recipe``        ``RunConfig`` (the settings of one run) and ``RunHistory`` (pydantic models)
- ``experiments``   registry of the experiments of notebooks 02 and 03, derived runs (Sections 15, 18), selected models
- ``train``         training engine: one run, early stopping, checkpoints, saved history
- ``predictions``   rebuild a model from a finished run, predictions with an on-disk cache
- ``evaluate``      metrics, seed statistics, compute, Grad-CAM
- ``plots``         plotting helpers
"""
