"""Build a SQLite database over the PlantVillage dataset.

Usage: python database/build_db.py [--data-root ./PlantVillage-Dataset] [--db ./database/plantvillage.db]
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

# the leaf-id rules of the notebooks (plantdisease/leafmap.py, no torch import)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plantdisease.leafmap import EXT_PATTERN, image_key, leaf_class, leaf_id

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
# color first: its file names define the photo stems
VARIANTS = ["color", "grayscale", "segmented"]
TABLES = ["crop", "disease", "class", "leaf", "leaf_map_entry", "photo", "image"]
SEG_SUFFIX = "_final_masked"


def display(name):
    # 'Corn_(maize)' -> 'Corn (maize)', 'Common_rust_' -> 'Common rust'
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def file_info(path):
    # (size in bytes, width, height, Pillow mode, sha256) of one image file
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with Image.open(path) as img:  # reads the header only
        width, height = img.size
        mode = img.mode
    return os.path.getsize(path), width, height, mode, digest


def insert_classes(con, class_names):
    crops, diseases = {}, {}
    for label_index, folder_name in enumerate(class_names):
        crop_name, disease_name = folder_name.split("___")
        if crop_name not in crops:
            crops[crop_name] = con.execute(
                "INSERT INTO crop (name, display_name) VALUES (?, ?)",
                (crop_name, display(crop_name)),
            ).lastrowid
        if disease_name not in diseases:
            diseases[disease_name] = con.execute(
                "INSERT INTO disease (name, display_name, is_healthy) VALUES (?, ?, ?)",
                (disease_name, display(disease_name), int(disease_name == "healthy")),
            ).lastrowid
        con.execute(
            "INSERT INTO class (class_id, folder_name, label_index, crop_id, disease_id) VALUES (?, ?, ?, ?, ?)",
            (
                label_index + 1,
                folder_name,
                label_index,
                crops[crop_name],
                diseases[disease_name],
            ),
        )
    return {name: i + 1 for i, name in enumerate(class_names)}


def insert_leaf_map(con, leaf_map, class_ids):
    leaf_ids = {}
    entries = set()
    for key, values in leaf_map.items():
        for leaf_key in values:
            class_name = leaf_class(leaf_key)
            if class_name not in class_ids:
                print(f"Skipping leaf-map entry of unknown class: {leaf_key}")
                continue
            if leaf_key not in leaf_ids:
                leaf_ids[leaf_key] = con.execute(
                    "INSERT INTO leaf (class_id, leaf_key, source) VALUES (?, ?, 'leaf map')",
                    (class_ids[class_name], leaf_key),
                ).lastrowid
            entries.add((key, class_ids[class_name], leaf_ids[leaf_key]))
    con.executemany(
        "INSERT INTO leaf_map_entry (image_key, class_id, leaf_id) VALUES (?, ?, ?)",
        sorted(entries),
    )
    return leaf_ids


def collect_images(data_root, class_names):
    # Group the files of all variants into photos. Returns (photos, images):
    # photos: list of (class_name, stem); images: list of (photo index, variant, rel_path)
    photos, images = [], []
    by_stem = {}  # (class, stem) -> photo index
    by_name = defaultdict(list)  # (class, stem without uuid) -> photo indices
    variants_of = defaultdict(set)  # photo index -> variants found
    for variant in VARIANTS:
        for class_name in class_names:
            class_dir = os.path.join(data_root, "raw", variant, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in sorted(os.listdir(class_dir)):
                if not EXT_PATTERN.search(file_name):
                    continue  # skips .DS_Store etc.
                stem = EXT_PATTERN.sub("", file_name).removesuffix(SEG_SUFFIX)
                name = stem.split("___")[-1]
                idx = by_stem.get((class_name, stem))
                if idx is None:
                    # Some segmented files lack the uuid prefix of their colour photo: match on the rest of the name
                    candidates = [
                        i
                        for i in by_name[(class_name, name)]
                        if variant not in variants_of[i]
                    ]
                    if len(candidates) == 1:
                        idx = candidates[0]
                    else:
                        idx = len(photos)
                        photos.append((class_name, stem))
                        by_stem[(class_name, stem)] = idx
                        by_name[(class_name, name)].append(idx)
                if variant in variants_of[idx]:
                    raise ValueError(
                        f"Two {variant} files for one photo: {class_dir}/{file_name}"
                    )
                variants_of[idx].add(variant)
                images.append((idx, variant, f"raw/{variant}/{class_name}/{file_name}"))
    return photos, images


def build(data_root, db_path, workers):
    start = time.time()
    class_names = sorted(
        {
            d
            for v in VARIANTS
            if os.path.isdir(os.path.join(data_root, "raw", v))
            for d in os.listdir(os.path.join(data_root, "raw", v))
            if "___" in d and os.path.isdir(os.path.join(data_root, "raw", v, d))
        }
    )
    with open(os.path.join(data_root, "leaf-map.json")) as f:
        leaf_map = json.load(f)  # image identifier -> ['<class>:::<leaf group>', ...]

    photos, images = collect_images(data_root, class_names)
    print(
        f"Found {len(class_names)} classes, {len(photos)} photos, {len(images)} image files"
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        infos = list(
            pool.map(
                file_info,
                (os.path.join(data_root, p) for _, _, p in images),
                chunksize=64,
            )
        )
    print(f"Read image metadata in {time.time() - start:.0f} s")

    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = sqlite3.connect(tmp_path)
    try:
        with open(SCHEMA_PATH) as f:
            con.executescript(f.read())
        with con:
            class_ids = insert_classes(con, class_names)
            leaf_ids = insert_leaf_map(con, leaf_map, class_ids)

            photo_rows = []
            for class_name, stem in photos:
                key = image_key(stem)
                leaf_key, source = leaf_id(leaf_map, class_name, stem)
                if leaf_key is not None and leaf_key not in leaf_ids:
                    leaf_ids[leaf_key] = con.execute(
                        "INSERT INTO leaf (class_id, leaf_key, source) VALUES (?, ?, ?)",
                        (class_ids[class_name], leaf_key, source),
                    ).lastrowid
                uuid = stem.split("___")[0] if "___" in stem else None
                photo_rows.append(
                    (class_ids[class_name], uuid, stem, key, leaf_ids.get(leaf_key))
                )
            con.executemany(
                "INSERT INTO photo (photo_id, class_id, uuid, stem, image_key, leaf_id) VALUES (?, ?, ?, ?, ?, ?)",
                ((i + 1, *row) for i, row in enumerate(photo_rows)),
            )
            con.executemany(
                "INSERT INTO image (photo_id, variant, rel_path, file_size, width, height, mode, sha256)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (idx + 1, variant, rel_path, *info)
                    for (idx, variant, rel_path), info in zip(
                        images, infos, strict=True
                    )
                ),
            )
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign key violations: {violations[:5]}")
        con.execute("ANALYZE")
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    os.replace(tmp_path, db_path)

    con = sqlite3.connect(db_path)
    for table in TABLES:
        query = f"SELECT COUNT(*) FROM {table}"  # noqa: S608 (fixed table names)
        print(f"{table:>15}: {con.execute(query).fetchone()[0]}")
    for variant, n in con.execute(
        "SELECT variant, COUNT(*) FROM image GROUP BY variant"
    ):
        print(f"{variant:>15}: {n} images")
    con.close()
    print(f"Wrote {db_path} in {time.time() - start:.0f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", default=os.environ.get("DATA_ROOT", "./PlantVillage-Dataset")
    )
    parser.add_argument("--db", default="./database/plantvillage.db")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()
    build(args.data_root, args.db, args.workers)


if __name__ == "__main__":
    main()
