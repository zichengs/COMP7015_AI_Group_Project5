"""Plant disease classification on PlantVillage (COMP7015 group project, Topic 1).

Shared code of the notebooks 01_classification and 02_robustness:

- ``config``        settings (PATHS_CONFIG, DATA_CONFIG, PLANTDOC_CONFIG, TRAIN_CONFIG)
- ``device``        device, mixed precision, determinism, set_seed, synchronize
- ``data/``
    - ``download``     PlantVillage (git clone) and PlantDoc field photos
    - ``leafmap``      leaf ids from leaf-map.json
    - ``records``      photo table, leaf-group split with leakage audit, data fingerprint
    - ``images``       image cache, segmented and grayscale versions, leaf masks
    - ``loaders``      preprocessing, augmentation, datasets and loaders
    - ``dataset``      the Data object (``prepare()``)
    - ``backgrounds``  test variants, background tests, background swap
- ``experiments``   what is trained: ``RunConfig`` (the settings of one run), the recipes of all experiments, derived runs (Sections 17, 20), selected models
- ``models``        custom CNN with SE attention, ResNet18, ViT-B/16
- ``train``         training engine: ``run(name, data)``, early stopping, checkpoints, saved history
- ``predictions``   rebuild a model from a finished run, predictions with an on-disk cache
- ``evaluate``      metrics, seed statistics, compute cost, confidence
- ``gradcam``       Grad-CAM / Score-CAM (pytorch-grad-cam) and the attention on the leaf
- ``plots``         plotting helpers
"""
