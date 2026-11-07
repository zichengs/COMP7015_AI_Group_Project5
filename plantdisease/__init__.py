"""Plant disease classification on PlantVillage (COMP7015 group project, Topic 1).

Shared code of the notebooks 01-05:

- ``config``      CFG dictionary, seeds, device
- ``data``        download, leaf grouping, image cache, near-duplicates, split, loaders
- ``backgrounds`` leaf masks, test variants, same-leaf background tests, background swap
- ``models``      custom CNN with SE attention, ResNet18, ViT-B/16
- ``experiments`` the configurations of Sections 9 and 14 and the selected models
- ``train``       training engine and cached predictions
- ``evaluate``    metrics, bootstrap, paired comparisons, Grad-CAM, compute
- ``plots``       plotting helpers
"""
