import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

import pandas as pd

from ..config import PATHS_CONFIG, PLANTDOC_CONFIG
from .leafmap import IMAGE_SUFFIXES

PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset"


# ----------------------------------------------------------------------------
# PlantVillage
# ----------------------------------------------------------------------------
def download_plantvillage() -> None:
    # Download the whole PlantVillage repository (about 2 GB) with `git clone --depth 1`: only the
    # latest files, not the history. Skipped if the folder already contains the photos.
    root = PATHS_CONFIG["data_root"]
    if os.path.isdir(os.path.join(root, "raw", "color")):
        return
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            f"git is needed to download the dataset from {PLANTVILLAGE_URL}"
        )
    subprocess.run([git, "clone", "--depth", "1", PLANTVILLAGE_URL, root], check=True)  # noqa: S603


# ----------------------------------------------------------------------------
# PlantDoc field photos
# ----------------------------------------------------------------------------
def _download_file(url: str, target: str, retries: int = 3) -> None:
    # Download one file; it only appears under `target` once it is complete.
    for attempt in range(retries):
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310
                open(target + ".part", "wb") as f,
            ):
                shutil.copyfileobj(response, f)
            break
        except OSError:
            if attempt == retries - 1:
                raise
    os.replace(target + ".part", target)


def _download_folder(repo: str, tree: list[dict], root: str, folder: str) -> None:
    # Download every file of one repository folder and mark the folder as complete.
    files = [
        e["path"]
        for e in tree
        if e["type"] == "blob" and os.path.dirname(e["path"]) == folder
    ]
    os.makedirs(os.path.join(root, folder), exist_ok=True)
    print(f"Downloading {folder} ({len(files)} files)")
    for path in files:
        # characters that Windows does not allow in file names become '_'
        # (PlantDoc has names like '...picture-id81781575?s=612x612.jpg')
        name = re.sub(r'[<>:"/\\|?*]', "_", os.path.basename(path))
        target = os.path.join(root, folder, name)
        if not os.path.exists(target):
            raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(path)}"
            _download_file(raw, target)
    open(os.path.join(root, folder, ".complete"), "w").close()


def download_github_folders(url: str, root: str, folders: list[str]) -> list[str]:
    # Download `folders` of a GitHub repository file by file over HTTPS (no git needed); returns the
    # folders that were downloaded completely. Finished folders are skipped, so a failed download
    # resumes when this runs again.
    repo = url.removeprefix("https://github.com/").strip("/")
    is_complete = {
        f: os.path.exists(os.path.join(root, f, ".complete")) for f in folders
    }
    missing = [f for f in folders if not is_complete[f]]
    if missing:
        try:
            api = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
            with urllib.request.urlopen(api, timeout=60) as response:  # noqa: S310
                tree = json.load(response)["tree"]
            for folder in missing:
                _download_folder(repo, tree, root, folder)
                is_complete[folder] = True
        except OSError as e:
            print(f"Download from {url} stopped: {e}. Run this again to resume.")
    return [f for f in folders if is_complete[f]]


def field_records(folders: dict[str, int]) -> pd.DataFrame:
    # PlantDoc photos of `folders` ({PlantDoc folder: class index}), from the
    # authors' train and test folders: all are test data here. Missing folders are downloaded;
    # folders that are not complete (download failed) are skipped.
    root = PLANTDOC_CONFIG["root"]
    wanted = [f"{split}/{folder}" for folder in folders for split in ("train", "test")]
    available = download_github_folders(PLANTDOC_CONFIG["url"], root, wanted)
    rows = []
    for folder, class_index in folders.items():
        for split in ("train", "test"):
            if f"{split}/{folder}" not in available:
                continue
            folder_dir = os.path.join(root, split, folder)
            rows += [
                {
                    "path": os.path.join(folder_dir, f),
                    "class_index": class_index,
                    "folder": folder,
                }
                for f in sorted(os.listdir(folder_dir))
                if f.lower().endswith(IMAGE_SUFFIXES)
            ]
    return pd.DataFrame(rows, columns=["path", "class_index", "folder"])
