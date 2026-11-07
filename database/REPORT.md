# PlantVillage SQLite Database – Exploration Report

Database: `database/plantvillage.db` (80 MB), built by `database/build_db.py` from the full
PlantVillage clone in `./PlantVillage-Dataset`. The schema is in `database/schema.sql`.

```bash
uv run python database/build_db.py      # ~20 s; rebuilds the database from scratch
sqlite3 database/plantvillage.db
```

## 1. Key findings

1. **The dataset is complete and consistent.**
   - 54,306 original photos across 38 classes, 14 crops and 20 distinct disease names (+ `healthy`).
   - Each photo has three versions: color, grayscale and segmented. In total there are 162,916 image files.
   - Every photo has all three versions except one, which only has a segmented file.
   - The integrity and foreign-key checks are clean.
2. **Classes are heavily imbalanced.** The largest class (Orange HLB, 5,507) is 36× the smallest (Potato healthy, 152). Tomato alone accounts for a third of the data (18,160 photos), and 72% of photos are diseased.
3. **80.2% of photos can be assigned to a physical leaf** (8,480 leaves). Leaf information is completely missing for 7 classes: all 4 Corn classes, Squash, Grape healthy and Tomato mosaic virus. It is largely missing for 4 more.
   - Leaf-group splits (the project's leakage-safe method) are only possible on the leaf-mapped part.
   - Most leaves were photographed 4 times (5,814 leaves). Some were photographed over 20 times.
4. **File names leak the label.**
   - The capture-series code in a file name (e.g. `GCREC_Bact.Sp`, `FAM_B.Msls`) predicts the class with **89.7%** accuracy.
   - 46 of the 49 series codes belong to a single class.
   - File names must never be used as model inputs.
5. **There are almost no duplicates.**
   - 21 pairs of byte-identical files, all within the same class. 18 pairs are within the same leaf. The other 3 pairs have no leaf information, so the notebooks already exclude them.
   - A dHash check (≤10 of 256 bits) adds only 1 near-duplicate pair.
   - No duplicate crosses classes.
6. **I found and fixed three mismatches between `leaf-map.json` and the image files.** These affected 2,046 photos (§6). The project's 4 tomato classes are unaffected, and the database still reproduces the notebooks' 544 excluded images exactly.

## 2. Schema

```mermaid
erDiagram
    crop    ||--o{ class : has
    disease ||--o{ class : has
    class   ||--o{ leaf : contains
    class   ||--o{ photo : labels
    leaf    |o--o{ photo : "shown in"
    leaf    ||--o{ leaf_map_entry : "mapped by"
    photo   ||--|{ image : "versions"
```

| Table | Rows | Content |
|---|---:|---|
| `crop` | 14 | crop species (`name`, `display_name`) |
| `disease` | 21 | 20 diseases + `healthy`; one disease can occur on several crops |
| `class` | 38 | one class per (crop, disease) pair; `label_index` = sorted folder order, same as the HF loader |
| `leaf` | 8,480 | physical leaf; `source` = `leaf map` (7,946) or `file name` (534) |
| `leaf_map_entry` | 42,520 | `leaf-map.json` stored as rows (40,328 keys) |
| `photo` | 54,306 | one original photo: `uuid`, `stem`, `image_key`, `leaf_id` (NULL = unknown) |
| `image` | 162,916 | one file: `variant`, `rel_path`, `file_size`, `width`, `height`, `mode`, `sha256` |

Two views:
- `v_image`: one flat row per file with all its labels.
- `v_class_counts`: images, leaves and images without a leaf, per class and variant.

## 3. Dataset composition

### 3.1 Crops

| Crop | Classes | Photos | | Crop | Classes | Photos |
|---|---:|---:|---|---|---:|---:|
| Tomato | 10 | 18,160 | | Peach | 2 | 2,657 |
| Orange | 1 | 5,507 | | Pepper, bell | 2 | 2,475 |
| Soybean | 1 | 5,090 | | Potato | 3 | 2,152 |
| Grape | 4 | 4,063 | | Cherry | 2 | 1,906 |
| Corn (maize) | 4 | 3,852 | | Squash | 1 | 1,835 |
| Apple | 4 | 3,171 | | Strawberry | 2 | 1,565 |
| | | | | Blueberry | 1 | 1,502 |
| | | | | Raspberry | 1 | 371 |

- 12 of the 14 crops have a `healthy` class.
- Orange and Squash have only one disease class each and no healthy class.
- Soybean, Blueberry and Raspberry have **only** a healthy class.
- For those 5 crops, a model can tell the label from the crop alone.

### 3.2 Diseases shared across crops

| Disease | Crops |
|---|---|
| Bacterial_spot | Peach, Pepper bell, Tomato |
| Black_rot | Apple, Grape |
| Early_blight | Potato, Tomato |
| Late_blight | Potato, Tomato |
| Powdery_mildew | Cherry, Squash |

This is why the database has 21 disease rows, not the "26 diseases" the dataset README advertises. The README counts crop–disease pairs.

### 3.3 Classes

| idx | Class | Photos | Leaves | No leaf | Photos/leaf |
|---:|---|---:|---:|---:|---:|
| 0 | Apple – Apple scab | 630 | 108 | 0 | 5.8 |
| 1 | Apple – Black rot | 621 | 71 | 0 | 8.7 |
| 2 | Apple – Cedar apple rust | 275 | 69 | 0 | 4.0 |
| 3 | Apple – healthy | 1,645 | 257 | 0 | 6.4 |
| 4 | Blueberry – healthy | 1,502 | 160 | 0 | 9.4 |
| 5 | Cherry – Powdery mildew | 1,052 | 65 | 0 | 16.2 |
| 6 | Cherry – healthy | 854 | 107 | 0 | 8.0 |
| 7 | Corn – Cercospora / Gray leaf spot | 513 | 0 | 513 | – |
| 8 | Corn – Common rust | 1,192 | 0 | 1,192 | – |
| 9 | Corn – Northern Leaf Blight | 985 | 0 | 985 | – |
| 10 | Corn – healthy | 1,162 | 0 | 1,162 | – |
| 11 | Grape – Black rot | 1,180 | 295 | 0 | 4.0 |
| 12 | Grape – Esca (Black Measles) | 1,384 | 346 | 0 | 4.0 |
| 13 | Grape – Leaf blight | 1,076 | 269 | 0 | 4.0 |
| 14 | Grape – healthy | 423 | 0 | 423 | – |
| 15 | Orange – Huanglongbing | 5,507 | 991 | 0 | 5.6 |
| 16 | Peach – Bacterial spot | 2,297 | 332 | 0 | 6.9 |
| 17 | Peach – healthy | 360 | 57 | 0 | 6.3 |
| 18 | Pepper bell – Bacterial spot | 997 | 175 | 0 | 5.7 |
| 19 | Pepper bell – healthy | 1,478 | 165 | 2 | 8.9 |
| 20 | Potato – Early blight | 1,000 | 250 | 0 | 4.0 |
| 21 | Potato – Late blight | 1,000 | 250 | 0 | 4.0 |
| 22 | Potato – healthy | 152 | 38 | 0 | 4.0 |
| 23 | Raspberry – healthy | 371 | 37 | 0 | 10.0 |
| 24 | Soybean – healthy | 5,090 | 509 | 0 | 10.0 |
| 25 | Squash – Powdery mildew | 1,835 | 0 | 1,835 | – |
| 26 | Strawberry – Leaf scorch | 1,109 | 75 | 333 | 10.3 |
| 27 | Strawberry – healthy | 456 | 115 | 0 | 4.0 |
| 28 | **Tomato – Bacterial spot** | 2,127 | 532 | 0 | 4.0 |
| 29 | **Tomato – Early blight** | 1,000 | 250 | 1 | 4.0 |
| 30 | **Tomato – Late blight** | 1,909 | 404 | 541 | 3.4 |
| 31 | Tomato – Leaf Mold | 952 | 238 | 0 | 4.0 |
| 32 | Tomato – Septoria leaf spot | 1,771 | 250 | 771 | 4.0 |
| 33 | Tomato – Spider mites | 1,676 | 419 | 0 | 4.0 |
| 34 | Tomato – Target Spot | 1,404 | 351 | 0 | 4.0 |
| 35 | Tomato – Yellow Leaf Curl Virus | 5,357 | 685 | 2,619 | 4.0 |
| 36 | Tomato – mosaic virus | 373 | 0 | 373 | – |
| 37 | **Tomato – healthy** | 1,591 | 610 | 2 | 2.6 |

Bold rows are the project's 4 classes.

- **Size:** classes range from 152 to 5,507 photos (mean 1,429). The imbalance ratio is 36:1.
- **Healthy vs. diseased:** 15,084 healthy (27.8%) and 39,222 diseased (72.2%).

## 4. Image files

| Variant | Files | Size on disk | Avg file | Format |
|---|---:|---:|---:|---|
| color | 54,305 | 811 MB | 15.3 KB | JPEG (1 `.png`, 1 `.jpeg`) |
| grayscale | 54,305 | 778 MB | 14.7 KB | JPEG (1 `.png`, 1 `.jpeg`) |
| segmented | 54,306 | 490 MB | 9.2 KB | JPEG |

- **Resolution:** 162,910 of 162,916 files are **256×256 RGB**. The exceptions are:
  - `Pepper,_bell___healthy/…Screen Shot 2015-05-06 at 4.01.13 PM.png` (color): RGBA. It is a screenshot, not a camera photo.
  - 4 segmented files at 324–470 × 512 px (Peach Bacterial spot ×2, Strawberry Leaf scorch, Tomato Spider mites). These are the segmented versions of the photos whose file names don't match (§6.1).
  - 1 segmented file in mode `L` (single-channel greyscale; Soybean healthy, `RS_HL 3512`).
- **Grayscale files** are stored as 3-channel RGB, but all 500 sampled files have R = G = B. They are real greyscale images, just stored with three identical channels.
- **Segmented files** put the leaf on a black background. A median of 54% of pixels are black (10th–90th percentile: 37–71%). This is why the segmented files are about 40% smaller.
- **The segmented-only photo:** `Grape___Esca_(Black_Measles)/…FAM_B.Msls 4430` has no color or grayscale file.

## 5. Leaf grouping

Several photos of the same physical leaf must stay in the same split. Otherwise test accuracy is inflated (the reason for the project's leaf-group split).

| Leaf source | Photos | Leaves |
|---|---:|---:|
| `leaf-map.json` | 42,516 | 7,946 |
| parsed from the file name (greenhouse series, as in the notebooks) | 1,038 | 534 |
| **none** | **10,752** | – |

- **Photos per leaf:**

  | Photos per leaf | Leaves | Photos |
  |---|---:|---:|
  | 1 | 218 | 218 |
  | 2–3 | 361 | 818 |
  | **4** | **5,814** | **23,256** |
  | 5–10 | 1,613 | 12,253 |
  | 11–20 | 427 | 5,946 |
  | > 20 | 47 | 1,063 |

  - The typical protocol was 4 photos per leaf.
  - The largest groups are Soybean healthy leaf 367 (33 photos), Blueberry healthy leaf 144 (29) and Orange HLB leaf 47 (25).
- **Classes without leaf information:**
  - None at all: all 4 Corn classes, Squash Powdery mildew, Grape healthy and Tomato mosaic virus.
  - Partial: Tomato YLCV (2,619 photos), Septoria (771), Late blight (541) and Strawberry Leaf scorch (333).
  - A leakage-safe split over all 38 classes would have to drop those photos (19.8%) or accept the leakage risk.
- **Leaf-map keys are not unique across crops.**
  - 2,172 keys appear in 2–3 classes. All of these are healthy classes from the `RS_HL` and `JR_HL` series, e.g. Apple/Soybean healthy (977 keys) and Blueberry/Soybean healthy (608 keys).
  - So a key only identifies a leaf **together with its class**. The database stores `leaf_map_entry.class_id` for this reason.
- **The project's tomato classes match the notebooks.**
  - Leaves per class: healthy 610, Bacterial spot 532, Early blight 250, Late blight 404. These are equal to the `groups` column of `outputs3_yolo/class_counts.csv`.
  - Images without a leaf: 2 + 0 + 1 + 541 = 544. That equals the row count of `excluded_images.csv` in `outputs2/`, `outputs3/` and `outputs3_yolo/`.

## 6. Data-quality issues (found while exploring)

### 6.1 File names that differ between versions
- 1,192 Corn Common rust color/grayscale files have no `uuid___` prefix, while their segmented versions do.
- 4 segmented files lack the prefix that their color versions have.
- The build matches these by the rest of the file name. 54,305 photos then link cleanly to all three versions.

### 6.2 `leaf-map.json` mismatches (fixed in `build_db.py`)
The first build used the dataset's own lookup rule. This review found three mismatches between the leaf map and the image files, now handled in `build_db.py`:

| Problem | Affected | Fix |
|---|---:|---|
| The leaf map calls Apple Black rot **`Apple_Frogeye Spot`**. Frogeye leaf spot is the leaf symptom of the same fungus (*Botryosphaeria obtusa*). | 621 photos, 71 leaves | class alias `LEAF_MAP_CLASS_ALIASES` |
| Target Spot files were **renamed** after the leaf map was made: `Com.G_TgS_FL 0004` ↔ key `com.g_fl 0004`. All 1,404 numbers match one-to-one. | 1,404 photos, 351 leaves | key rewrite `LEAF_MAP_KEY_PREFIXES` |
| The lookup rule strips `copy` from file names. But 21 Cherry healthy `… copy` files have **their own** leaf-map entries, pointing to a different leaf than the original (e.g. `JR_HL 9874 copy` → leaf 101, not 102). | 21 photos | try the exact name before the stripped key |

- **Before:** leaf coverage was 77.6%, with 353 leaves that had no photo and 1,435 leaf-map keys that matched no file.
- **After:** coverage is 80.2%, and every leaf and every leaf-map key is matched.
- These fixes affect none of the project's 4 tomato classes, so the results above don't change.

### 6.3 Duplicates
- **Byte-identical files (SHA-256):** 21 pairs.
  - Apple healthy (7), Tomato Late blight (8) and Tomato healthy (6).
  - The two files of a pair always differ only in the uuid prefix (e.g. `…___RS_HL 6271.JPG` twice).
  - All are within one class: 18 pairs share a leaf, and 3 Late blight `GHLB2 Leaf 8xxx/9xxx` pairs have no leaf information.
- **Near-duplicates (256-bit dHash, the same check as the notebooks):**
  - ≤ 10 bits: 22 pairs over all 54,305 color photos, so only 1 beyond the exact duplicates.
  - **0 pairs across classes.**
- **"copy" files:** 1,367 photos have `copy` in the name (1,141 of them in Corn healthy). None is a near-duplicate of its "original". They are different photos with a misleading name.

### 6.4 File names leak the label
The series code in a file name (the part before the number) says where and how a photo was taken, and often names the disease:

| Series code | Photos | Class |
|---|---:|---|
| `crec_hlb` | 3,371 | Orange HLB only |
| `uf.grc_ylcv_lab` | 2,619 | Tomato YLCV only |
| `gcrec_bact.sp` | 1,756 | Tomato Bacterial spot only |
| `com.g_spm_fl` | 1,676 | Tomato Spider mites only |
| `fam_b.msls` | 1,384 | Grape Esca only |
| `rs_hl` | 9,844 | 6 healthy classes |

- 46 of the 49 codes belong to a single class. Predicting each photo's class from its code alone is 89.7% accurate.
- The three mixed codes (`rs_hl`, `jr_hl`, `gh_hl leaf`) all mean "healthy".
- Each class was photographed in its own sessions, so lighting and background may also correlate with the label (cf. the background-bias tests in v2).

## 7. Example queries

```sql
-- Image counts per class and version
SELECT * FROM v_class_counts WHERE class_name LIKE 'Tomato%';

-- All color photos of the same physical leaf as a given file
SELECT rel_path FROM v_image
WHERE variant = 'color'
  AND leaf_key = (SELECT leaf_key FROM v_image WHERE rel_path LIKE '%RS_HL 0105%' AND variant = 'color');

-- Diseases that occur on more than one crop
SELECT d.name AS disease, GROUP_CONCAT(cr.name, ', ') AS crops
FROM class c JOIN crop cr USING (crop_id) JOIN disease d USING (disease_id)
WHERE NOT d.is_healthy GROUP BY d.disease_id HAVING COUNT(*) > 1;

-- Byte-identical files
SELECT a.rel_path, b.rel_path FROM image a
JOIN image b ON a.sha256 = b.sha256 AND a.image_id < b.image_id
WHERE a.variant = 'color';

-- Photos usable for a leaf-group split (and those that would be excluded)
SELECT class_name, SUM(leaf_key IS NOT NULL) AS usable, SUM(leaf_key IS NULL) AS excluded
FROM v_image WHERE variant = 'segmented' GROUP BY class_name;
```

Loading into pandas:

```python
import sqlite3
import pandas as pd

con = sqlite3.connect("database/plantvillage.db")
df = pd.read_sql(
    "SELECT * FROM v_image WHERE variant = 'segmented' AND crop = 'Tomato'", con
)
```
