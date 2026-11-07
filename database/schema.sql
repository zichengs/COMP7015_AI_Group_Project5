-- Relational schema for the PlantVillage dataset (built by database/build_db.py)

PRAGMA foreign_keys = ON;

-- Crop species, e.g. 'Tomato', 'Corn_(maize)'
CREATE TABLE crop (
    crop_id      INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

-- Disease (or 'healthy'); shared across crops, e.g. 'Bacterial_spot' for Peach, Pepper and Tomato
CREATE TABLE disease (
    disease_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    is_healthy   INTEGER NOT NULL CHECK (is_healthy IN (0, 1))
);

-- One class folder = one (crop, disease) pair; label_index follows the sorted folder names
CREATE TABLE class (
    class_id    INTEGER PRIMARY KEY,
    folder_name TEXT NOT NULL UNIQUE,
    label_index INTEGER NOT NULL UNIQUE,
    crop_id     INTEGER NOT NULL REFERENCES crop (crop_id),
    disease_id  INTEGER NOT NULL REFERENCES disease (disease_id),
    UNIQUE (crop_id, disease_id)
);

-- A physical leaf; several photos can show the same leaf
CREATE TABLE leaf (
    leaf_id  INTEGER PRIMARY KEY,
    class_id INTEGER NOT NULL REFERENCES class (class_id),
    leaf_key TEXT NOT NULL UNIQUE,  -- '<class>:::<leaf group>', e.g. 'Tomato___healthy:::207.0'
    source   TEXT NOT NULL CHECK (source IN ('leaf map', 'file name'))
);

-- leaf-map.json as rows: image identifier -> leaf (a key can map to leaves of several classes)
CREATE TABLE leaf_map_entry (
    image_key TEXT NOT NULL,
    class_id  INTEGER NOT NULL REFERENCES class (class_id),
    leaf_id   INTEGER NOT NULL REFERENCES leaf (leaf_id),
    PRIMARY KEY (image_key, leaf_id)
);

-- One original photo; its color, grayscale and segmented versions are rows of `image`
CREATE TABLE photo (
    photo_id  INTEGER PRIMARY KEY,
    class_id  INTEGER NOT NULL REFERENCES class (class_id),
    uuid      TEXT,                             -- prefix before '___' (NULL for files without one)
    stem      TEXT NOT NULL,                    -- file name without extension and '_final_masked'
    image_key TEXT NOT NULL,                    -- lookup key of leaf-map.json, e.g. 'rs_hl 0105'
    leaf_id   INTEGER REFERENCES leaf (leaf_id),  -- NULL = no usable leaf information
    UNIQUE (class_id, stem)
);

-- One image file on disk
CREATE TABLE image (
    image_id  INTEGER PRIMARY KEY,
    photo_id  INTEGER NOT NULL REFERENCES photo (photo_id),
    variant   TEXT NOT NULL CHECK (variant IN ('color', 'grayscale', 'segmented')),
    rel_path  TEXT NOT NULL UNIQUE,             -- relative to the dataset root, e.g. 'raw/color/...'
    file_size INTEGER NOT NULL,                 -- bytes
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    mode      TEXT NOT NULL,                    -- Pillow mode, e.g. 'RGB', 'L'
    sha256    TEXT NOT NULL,
    UNIQUE (photo_id, variant)
);

CREATE INDEX idx_class_crop ON class (crop_id);
CREATE INDEX idx_class_disease ON class (disease_id);
CREATE INDEX idx_leaf_class ON leaf (class_id);
CREATE INDEX idx_leaf_map_entry_class ON leaf_map_entry (class_id);
CREATE INDEX idx_leaf_map_entry_leaf ON leaf_map_entry (leaf_id);
CREATE INDEX idx_photo_leaf ON photo (leaf_id);
CREATE INDEX idx_photo_image_key ON photo (image_key);
CREATE INDEX idx_image_sha256 ON image (sha256);

-- Flat view: one row per image file with all labels
CREATE VIEW v_image AS
SELECT
    i.image_id,
    i.rel_path,
    i.variant,
    i.width,
    i.height,
    i.mode,
    i.file_size,
    i.sha256,
    p.photo_id,
    p.uuid,
    p.image_key,
    c.folder_name AS class_name,
    c.label_index,
    cr.name AS crop,
    d.name AS disease,
    d.is_healthy,
    l.leaf_key,
    l.source AS leaf_source
FROM image AS i
JOIN photo AS p ON p.photo_id = i.photo_id
JOIN class AS c ON c.class_id = p.class_id
JOIN crop AS cr ON cr.crop_id = c.crop_id
JOIN disease AS d ON d.disease_id = c.disease_id
LEFT JOIN leaf AS l ON l.leaf_id = p.leaf_id;

-- Images per class and variant, with the number of distinct leaves
CREATE VIEW v_class_counts AS
SELECT
    c.label_index,
    c.folder_name AS class_name,
    i.variant,
    COUNT(*) AS images,
    COUNT(DISTINCT p.leaf_id) AS leaves,
    SUM(p.leaf_id IS NULL) AS images_without_leaf
FROM image AS i
JOIN photo AS p ON p.photo_id = i.photo_id
JOIN class AS c ON c.class_id = p.class_id
GROUP BY c.class_id, i.variant
ORDER BY c.label_index, i.variant;
