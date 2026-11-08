"""Leaf ids of PlantVillage photos from the dataset's leaf-map.json (Section 3).

No torch import, so that database/build_db.py can use it without loading the notebooks' setup.
"""

import re

EXT_PATTERN = re.compile(r"\.(jpe?g|png)$", flags=re.IGNORECASE)
SERIES_PATTERN = re.compile(
    r"^(gh_hl|ghlb_ps|ghlb2es|ghlb2|ghlb) leaf (\d+)"
)  # greenhouse series with leaf numbers in the file name
# leaf-map.json names Apple black rot after its leaf symptom (frogeye leaf spot, same fungus)
LEAF_MAP_CLASS_ALIASES = {"Apple_Frogeye Spot": "Apple___Black_rot"}
# Target Spot files were renamed after the leaf map was made: 'Com.G_TgS_FL 0004' -> 'com.g_fl 0004'
LEAF_MAP_KEY_PREFIXES = {"com.g_tgs_fl ": "com.g_fl "}


def image_key(file_name):
    # '0031da2a-...___RS_HL 0105 copy.JPG' -> 'rs_hl 0105' (same logic as the repository's plant_village.py)
    key = file_name.split("___")[-1].split("copy")[0].strip()
    return EXT_PATTERN.sub("", key).strip().lower()


def leaf_class(entry):
    # '<class>:::<leaf group>' -> class folder name
    class_name = entry.split(":::")[0]
    return LEAF_MAP_CLASS_ALIASES.get(class_name, class_name)


def leaf_map_key(key):
    # image key -> key used in leaf-map.json
    for new, old in LEAF_MAP_KEY_PREFIXES.items():
        if key.startswith(new):
            return old + key.removeprefix(new)
    return key


def leaf_id(leaf_map, class_name, file_name):
    # (leaf id, source) of a photo of the folder `class_name`; the leaf id is None if the photo has
    # no usable leaf information. The exact name is tried before the key without 'copy':
    # leaf-map.json has own entries for some '... copy' files, which are different photos.
    key = image_key(file_name)
    exact = EXT_PATTERN.sub("", file_name).split("___")[-1].strip().lower()
    for lookup in dict.fromkeys([exact, key]):
        for entry in leaf_map.get(leaf_map_key(lookup), []):
            if leaf_class(entry) == class_name:
                return entry, "leaf map"
    match = SERIES_PATTERN.match(key)
    # numbers >= 1000 are photo counters, not leaf numbers
    if match and int(match.group(2)) < 1000:
        return f"{class_name}:::{match.group(1)} leaf {match.group(2)}", "file name"
    return None, "excluded: no leaf information"
