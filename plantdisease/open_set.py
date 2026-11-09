"""Notebook 07: the classes 'Other disease' and 'Other crop' next to the four tomato classes."""

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import DATA_CONFIG, SHORT_NAMES
from .data import (
    DATA_ROOT,
    IMAGE_SUFFIXES,
    Data,
    build_cache,
    data_fingerprint,
    download_plantvillage,
    field_records,
    load_segmented,
    segmented_path,
    split_leaves,
)
from .leafmap import image_key, leaf_id

# ----------------------------------------------------------------------------
# Design: labels 0-3 are the four tomato classes of notebooks 01-05 (same photos, same split),
# label 4 collects other tomato diseases and label 5 other crops (healthy and diseased leaves).
# Some diseases and crops are held out of training completely: they test photos of a kind the
# models have never seen.
# ----------------------------------------------------------------------------
OTHER_DISEASE, OTHER_CROP = len(DATA_CONFIG["classes"]), len(DATA_CONFIG["classes"]) + 1
TOMATO_LABELS = list(range(len(DATA_CONFIG["classes"])))
CLASS_NAMES = [*SHORT_NAMES, "Other_disease", "Other_crop"]
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_FOLDERS = {
    OTHER_DISEASE: [
        "Tomato___Bacterial_spot",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ],
    OTHER_CROP: [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
    ],
}
HELD_OUT_FOLDERS = {
    OTHER_DISEASE: ["Tomato___Late_blight", "Tomato___Tomato_mosaic_virus"],
    OTHER_CROP: [
        "Blueberry___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
    ],
}
# PlantDoc folders outside the four classes -> (label, kind seen in training?)
FIELD_FOLDERS = {
    "Tomato leaf bacterial spot": (OTHER_DISEASE, True),
    "Tomato Septoria leaf spot": (OTHER_DISEASE, True),
    "Tomato leaf yellow virus": (OTHER_DISEASE, True),
    "Tomato leaf late blight": (OTHER_DISEASE, False),
    "Tomato leaf mosaic virus": (OTHER_DISEASE, False),
    "Apple Scab Leaf": (OTHER_CROP, True),
    "Apple leaf": (OTHER_CROP, True),
    "Apple rust leaf": (OTHER_CROP, True),
    "Bell_pepper leaf": (OTHER_CROP, True),
    "Bell_pepper leaf spot": (OTHER_CROP, True),
    "Cherry leaf": (OTHER_CROP, True),
    "Peach leaf": (OTHER_CROP, True),
    "Potato leaf early blight": (OTHER_CROP, True),
    "Potato leaf late blight": (OTHER_CROP, True),
    "Strawberry leaf": (OTHER_CROP, True),
    "grape leaf": (OTHER_CROP, True),
    "grape leaf black rot": (OTHER_CROP, True),
    "Blueberry leaf": (OTHER_CROP, False),
    "Corn Gray leaf spot": (OTHER_CROP, False),
    "Corn leaf blight": (OTHER_CROP, False),
    "Corn rust leaf": (OTHER_CROP, False),
    "Raspberry leaf": (OTHER_CROP, False),
    "Soyabean leaf": (OTHER_CROP, False),
    "Squash Powdery mildew leaf": (OTHER_CROP, False),
}
PHOTOS_PER_CLASS = 1000  # about the size of the tomato classes
PHOTOS_PER_LEAF = 2  # at most: the new classes cover as many leaves as possible
HELD_OUT_PER_FOLDER = 150  # photos of every held-out folder (test only)


# ----------------------------------------------------------------------------
# Download and records
# ----------------------------------------------------------------------------
def download_open_set() -> None:
    # raw/color and raw/segmented (background removed) of every training and held-out folder.
    folders = [
        f"raw/{kind}/{f}"
        for kind in ("color", "segmented")
        for groups in (TRAIN_FOLDERS, HELD_OUT_FOLDERS)
        for fs in groups.values()
        for f in fs
    ]
    download_plantvillage(folders)
    missing = [f for f in folders if not os.path.isdir(os.path.join(DATA_ROOT, f))]
    assert not missing, f"not downloaded: {missing}"


def folder_photos(folder: str) -> list[str]:
    # Photos of a folder that also have a background-removed version (the models train on those).
    folder_dir = os.path.join(DATA_ROOT, "raw", "color", folder)
    return [
        f
        for f in sorted(os.listdir(folder_dir))
        if f.lower().endswith(IMAGE_SUFFIXES)
        and os.path.exists(segmented_path(os.path.join(folder_dir, f)))
    ]


def sample_folder(
    leaf_map: dict[str, list[str]],
    folder: str,
    label: int,
    n_photos: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    # Whole leaves of one folder in random order, at most PHOTOS_PER_LEAF photos each, until
    # n_photos are reached. The leaves are split 70/15/15 like the tomato classes (make_split).
    leaf_of = {}  # photo -> official leaf id (leaf-map.json only)
    for f in folder_photos(folder):
        leaf, source = leaf_id(leaf_map, folder, f)
        if source == "leaf map":
            leaf_of[f] = leaf
    photos = pd.Series(leaf_of)
    chosen = {}  # leaf id -> its photos, in the random order of the leaves
    for leaf in rng.permutation(sorted(photos.unique())):
        chosen[leaf] = rng.permutation(photos.index[photos == leaf].to_numpy())[
            :PHOTOS_PER_LEAF
        ]
        if sum(map(len, chosen.values())) >= n_photos:
            break
    split_of = split_leaves(list(chosen))
    rows = []
    for leaf, files in chosen.items():
        for f in sorted(files):
            key = image_key(f)
            rows.append(
                {
                    "path": os.path.join(DATA_ROOT, "raw", "color", folder, f),
                    "file_name": f,
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                    "folder": folder,
                    "key": key,
                    "series": key.split()[0],
                    "leaf_id": leaf,
                    "source": "leaf map",
                    "split": split_of[leaf],
                }
            )
    return rows


def open_set_records() -> pd.DataFrame:
    # Photos of the two new classes: about PHOTOS_PER_CLASS per class, spread evenly over the folders.
    with open(os.path.join(DATA_ROOT, "leaf-map.json")) as f:
        leaf_map = json.load(f)
    rng = np.random.default_rng(DATA_CONFIG["split_seed"])
    rows = []
    for label, folders in TRAIN_FOLDERS.items():
        per_folder = int(np.ceil(PHOTOS_PER_CLASS / len(folders)))
        for folder in folders:
            rows += sample_folder(leaf_map, folder, label, per_folder, rng)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Six-class data: the tomato photos and split of notebook 01 plus the new classes
# ----------------------------------------------------------------------------
def prepare_open_set(data: Data) -> Data:
    # data: the four-class data of prepare(). Returns Data with labels 0-5; the first photos are
    # those of `data`, with the same split.
    new_df = open_set_records()
    tomato_df = data.df.assign(folder=data.df["class_name"])
    df = pd.concat([tomato_df, new_df], ignore_index=True)
    n = len(tomato_df)
    images = torch.cat([data.images, build_cache(new_df["path"], "open_set")])
    new_idx = {s: n + np.where(new_df.split == s)[0] for s in ["train", "val", "test"]}
    train_idx = np.concatenate([data.train_idx, new_idx["train"]])
    data6 = Data(
        df=df,
        other_df=data.other_df,
        images=images,
        labels=torch.tensor(df["label"].values),
        train_idx=train_idx,
        val_idx=np.concatenate([data.val_idx, new_idx["val"]]),
        test_idx=np.concatenate([data.test_idx, new_idx["test"]]),
        data_id=data_fingerprint(df, train_idx),
    )
    # Background-removed photos (filled in here, so that the tomato part reuses the cache of
    # notebooks 01-05 instead of writing a second one)
    tomato, tomato_ok = data.segmented
    new, new_ok = load_segmented(new_df["path"], "open_set_segmented")
    data6.segmented = torch.cat([tomato, new]), np.concatenate([tomato_ok, new_ok])
    return data6


# ----------------------------------------------------------------------------
# Out-of-scope test photos
# ----------------------------------------------------------------------------
def held_out_records() -> pd.DataFrame:
    # HELD_OUT_PER_FOLDER photos of every held-out folder (leaf ids are not needed: test only).
    rng = np.random.default_rng(DATA_CONFIG["split_seed"])
    rows = []
    for label, folders in HELD_OUT_FOLDERS.items():
        for folder in folders:
            files = folder_photos(folder)
            chosen = rng.choice(
                len(files), min(HELD_OUT_PER_FOLDER, len(files)), replace=False
            )
            rows += [
                {
                    "path": os.path.join(DATA_ROOT, "raw", "color", folder, files[i]),
                    "label": label,
                    "folder": folder,
                }
                for i in sorted(chosen)
            ]
    return pd.DataFrame(rows)


def out_of_scope_field_records() -> pd.DataFrame:
    # PlantDoc photos of other tomato diseases and other crops, with whether their kind is seen in training.
    field_df = field_records({f: label for f, (label, _) in FIELD_FOLDERS.items()})
    return field_df.assign(
        **{"seen kind": field_df["folder"].map(lambda f: FIELD_FOLDERS[f][1])}
    )


# ----------------------------------------------------------------------------
# Rejection: does a model refuse a photo outside the four tomato classes?
# ----------------------------------------------------------------------------
def tomato_confidence(probs: np.ndarray) -> np.ndarray:
    # Probability of the most likely tomato class (four-class models: the usual confidence).
    return probs[:, TOMATO_LABELS].max(1)


def rejection_threshold(val_probs: np.ndarray, keep: float = 0.95) -> float:
    # Confidence threshold that keeps `keep` of the tomato validation photos.
    return np.quantile(tomato_confidence(val_probs), 1 - keep)


def rejected(probs: np.ndarray, threshold: float) -> np.ndarray:
    # A photo is rejected if the model predicts a class outside the four tomato classes
    # (six-class models only) or if its tomato confidence is below the threshold.
    return (probs.argmax(1) >= len(TOMATO_LABELS)) | (
        tomato_confidence(probs) < threshold
    )


def out_of_scope_auroc(
    probs_in: np.ndarray, probs_out: np.ndarray
) -> float | np.floating:
    # Threshold-free separation of tomato photos (in scope) from out-of-scope photos by
    # 1 - tomato confidence: 0.5 = chance, 1 = perfect.
    score = 1 - np.concatenate(
        [tomato_confidence(probs_in), tomato_confidence(probs_out)]
    )
    is_out = np.r_[np.zeros(len(probs_in)), np.ones(len(probs_out))]
    return roc_auc_score(is_out, score)
