"""Downloads: PlantVillage (git sparse checkout) and PlantDoc (file by file over HTTPS)."""

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

import pandas as pd

from ..config import DATA_CONFIG, PATHS_CONFIG, PLANTDOC_CONFIG
from .leafmap import IMAGE_SUFFIXES


# ----------------------------------------------------------------------------
# Section 2: PlantVillage
# ----------------------------------------------------------------------------
def sparse_checkout(url: str, root: str, folders: list[str]) -> list[str]:
    # Clone only `folders` of a git repository (sparse checkout); returns the folders that exist afterwards.
    git = shutil.which("git")
    if not os.path.exists(root):
        subprocess.run(  # noqa: S603
            [git, "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, root],
            check=False,
        )
    if not all(os.path.isdir(os.path.join(root, f)) for f in folders):
        subprocess.run([git, "sparse-checkout", "set", *folders], cwd=root, check=False)  # noqa: S603
    return [f for f in folders if os.path.isdir(os.path.join(root, f))]


def download_plantvillage() -> list[str]:
    # The four tomato folders of raw/color (training images) and raw/segmented (background removed).
    folders = [
        f"raw/{kind}/{c}"
        for kind in ["color", "segmented"]
        for c in DATA_CONFIG["classes"]
    ]
    sparse_checkout(
        "https://github.com/spMohanty/PlantVillage-Dataset",
        PATHS_CONFIG["data_root"],
        folders,
    )
    return folders


# ----------------------------------------------------------------------------
# Section 19.2: PlantDoc field photos
# ----------------------------------------------------------------------------
def download_github_folders(url: str, root: str, folders: list[str]) -> list[str]:
    # Download `folders` of a GitHub repository file by file over HTTPS (no git needed); returns the
    # folders that were downloaded completely. Characters that Windows does not allow in file names
    # (PlantDoc has names like '...picture-id81781575?s=612x612.jpg') are replaced by '_'.
    repo = url.removeprefix("https://github.com/").strip("/")
    done = [os.path.join(root, f, ".complete") for f in folders]
    missing = [f for f, d in zip(folders, done, strict=True) if not os.path.exists(d)]
    if missing:
        try:
            api = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
            with urllib.request.urlopen(api, timeout=60) as response:  # noqa: S310
                tree = json.load(response)["tree"]
            for folder in missing:
                files = [
                    e["path"]
                    for e in tree
                    if e["type"] == "blob" and os.path.dirname(e["path"]) == folder
                ]
                os.makedirs(os.path.join(root, folder), exist_ok=True)
                print(f"Downloading {folder} ({len(files)} files)")
                for path in files:
                    name = re.sub(r'[<>:"/\\|?*]', "_", os.path.basename(path))
                    target = os.path.join(root, folder, name)
                    if os.path.exists(target):
                        continue
                    raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(path)}"
                    for attempt in range(3):
                        try:
                            with urllib.request.urlopen(raw, timeout=60) as response:  # noqa: S310
                                with open(target + ".part", "wb") as f:
                                    shutil.copyfileobj(response, f)
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                    os.replace(target + ".part", target)
                open(os.path.join(root, folder, ".complete"), "w").close()
        except OSError as e:
            print(f"Download from {url} stopped: {e}. Run this again to resume.")
    return [f for f, d in zip(folders, done, strict=True) if os.path.exists(d)]


def field_records(folders: dict[str, int]) -> pd.DataFrame:
    # PlantDoc photos (Section 19.2) of `folders` ({PlantDoc folder: label}), from the
    # authors' train and test folders: all are test data here. Missing folders are downloaded;
    # folders that are not complete (download failed) are skipped.
    root = PLANTDOC_CONFIG["root"]
    wanted = [f"{split}/{folder}" for folder in folders for split in ("train", "test")]
    available = download_github_folders(PLANTDOC_CONFIG["url"], root, wanted)
    rows = []
    for folder, label in folders.items():
        for split in ("train", "test"):
            if f"{split}/{folder}" not in available:
                continue
            folder_dir = os.path.join(root, split, folder)
            rows += [
                {"path": os.path.join(folder_dir, f), "label": label, "folder": folder}
                for f in sorted(os.listdir(folder_dir))
                if f.lower().endswith(IMAGE_SUFFIXES)
            ]
    return pd.DataFrame(rows, columns=["path", "label", "folder"])
