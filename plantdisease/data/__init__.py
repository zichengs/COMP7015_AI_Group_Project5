"""Data pipeline: download, photo table and split, image cache, loaders, and the Data object (Sections 2-6, 14-18).

Import from the submodules (nothing is imported here, so `leafmap` stays free of torch):

- ``download``    PlantVillage (git sparse checkout) and PlantDoc (file by file) downloads
- ``leafmap``     leaf ids of the photos from the dataset's leaf-map.json (no torch)
- ``records``     photo table, series, leaf-group split with audit, data fingerprint
- ``images``      decode and resize once (image cache), segmented and grayscale versions
- ``loaders``     transforms, CachedDataset, data loaders
- ``dataset``     the ``Data`` object and ``prepare()``: everything above in one call
- ``backgrounds`` leaf masks, test variants, background swap (Sections 14, 16, 18)
"""
