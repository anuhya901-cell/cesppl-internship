from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = REPO_ROOT / "Week-09"
IMAGE_DIR = REPO_ROOT / "data" / "cesppl_processed"
CLASSES_FILE = REPO_ROOT / "CLASSES.md"

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def get_class_names():
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"CLASSES.md not found:\n{CLASSES_FILE}"
        )

    class_names = []

    with open(CLASSES_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            line = re.sub(r"^[-*+]\s+", "", line)

            if line:
                class_names.append(
                    line.replace("_", " ").strip()
                )

    if len(class_names) != 10:
        raise ValueError(
            f"Expected 10 classes, but found {len(class_names)}."
        )

    return class_names


def _read_split_csv(split_name):
    split_name = split_name.lower().strip()

    if split_name not in {"train", "val", "test"}:
        raise ValueError(
            "split_name must be train, val or test."
        )

    csv_path = SPLIT_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {"filename", "class"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"{csv_path.name} must contain filename and class columns."
        )

    dataframe = dataframe[["filename", "class"]].copy()

    if dataframe.isna().any().any():
        raise ValueError(
            f"Missing values found in {csv_path.name}."
        )

    dataframe["filename"] = (
        dataframe["filename"]
        .astype(str)
        .str.strip()
    )

    dataframe["class"] = (
        dataframe["class"]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
    )

    return dataframe


def _find_image_path(filename):
    filename_path = Path(str(filename))

    possible_paths = []

    if filename_path.is_absolute():
        possible_paths.append(filename_path)

    possible_paths.extend(
        [
            IMAGE_DIR / filename_path,
            REPO_ROOT / filename_path,
            SPLIT_DIR / filename_path,
        ]
    )

    for path in possible_paths:
        if path.exists():
            return path.resolve().as_posix()

    attempted_paths = "\n".join(
        f"- {path}"
        for path in possible_paths
    )

    raise FileNotFoundError(
        f"Image not found for CSV entry:\n{filename}\n\n"
        f"Attempted paths:\n{attempted_paths}"
    )


def _load_image(image_path, label, img_size):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [img_size, img_size]
    )

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)

    return image, label


def load_split(
    split_name,
    img_size=224,
    batch_size=32
):
    split_name = split_name.lower().strip()

    dataframe = _read_split_csv(split_name)

    class_names = get_class_names()

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    unknown_classes = sorted(
        set(dataframe["class"]) - set(class_names)
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown classes found: {unknown_classes}"
        )

    image_paths = np.array(
        [
            _find_image_path(filename)
            for filename in dataframe["filename"]
        ],
        dtype=str
    )

    labels = (
        dataframe["class"]
        .map(class_to_index)
        .astype(np.int32)
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if split_name == "train":
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: _load_image(
            path,
            label,
            img_size
        ),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_class_weights():
    train_df = _read_split_csv("train")

    class_names = get_class_names()

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    unknown_classes = sorted(
        set(train_df["class"]) - set(class_names)
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown classes found in train.csv: {unknown_classes}"
        )

    labels = (
        train_df["class"]
        .map(class_to_index)
        .astype(np.int32)
        .to_numpy()
    )

    class_indices = np.arange(len(class_names))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=labels
    )

    class_weight_dict = {
        int(index): float(weight)
        for index, weight in zip(
            class_indices,
            weights
        )
    }

    return class_weight_dict