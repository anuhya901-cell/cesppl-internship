from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image


keras = tf.keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

_MODEL: keras.Model | None = None
_CLASS_NAMES: list[str] | None = None


def _read_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"class_names.json was not found:\n{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_data = json.load(file)

    if isinstance(class_data, dict):
        class_names = [
            class_data[str(index)]
            for index in range(len(class_data))
        ]

    elif isinstance(class_data, list):
        class_names = [
            str(class_name)
            for class_name in class_data
        ]

    else:
        raise ValueError(
            "class_names.json must contain either a list "
            "or an index-to-class mapping."
        )

    if not class_names:
        raise ValueError(
            "class_names.json does not contain any classes."
        )

    return class_names


def load_model() -> tuple[keras.Model, list[str]]:
    """
    Load the final model and class names only once.
    """

    global _MODEL
    global _CLASS_NAMES

    if _MODEL is not None and _CLASS_NAMES is not None:
        return _MODEL, _CLASS_NAMES

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model was not found:\n{MODEL_PATH}\n"
            "Run export_final_model.py first."
        )

    print(f"Loading model from:\n{MODEL_PATH}")

    preprocess_function = (
        tf.keras.applications.efficientnet.preprocess_input
    )

    _MODEL = keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        safe_mode=False,
        custom_objects={
            "preprocess_input": preprocess_function,
            "function": preprocess_function,
        },
    )
    _CLASS_NAMES = _read_class_names()

    number_of_outputs = int(
        _MODEL.output_shape[-1]
    )

    if number_of_outputs != len(_CLASS_NAMES):
        raise ValueError(
            "The model outputs and class-name count do not match. "
            f"Model outputs: {number_of_outputs}, "
            f"class names: {len(_CLASS_NAMES)}"
        )

    print("Model loaded successfully.")

    return _MODEL, _CLASS_NAMES


def get_class_names() -> list[str]:
    """
    Return the ordered class names.
    """

    _, class_names = load_model()

    return list(class_names)


def prepare_image(
    pil_image: Image.Image,
) -> tf.Tensor:
    """
    Convert one PIL image into the exact model input format.

    EfficientNet preprocessing is already stored inside
    final_model.keras, so pixel values remain in the 0-255 range.
    """

    model, _ = load_model()

    if not isinstance(pil_image, Image.Image):
        raise TypeError(
            "predict_image expects a PIL.Image.Image object."
        )

    image_height = int(model.input_shape[1])
    image_width = int(model.input_shape[2])

    rgb_image = pil_image.convert("RGB")

    image_array = np.asarray(
        rgb_image,
        dtype=np.float32,
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    image_tensor = tf.image.resize(
        image_tensor,
        size=(image_height, image_width),
        method="bilinear",
        antialias=False,
    )

    image_batch = tf.expand_dims(
        image_tensor,
        axis=0,
    )

    return image_batch


def predict_image(
    pil_image: Image.Image,
) -> tuple[str, float, list[dict[str, Any]]]:
    """
    Return top-one class, top-one confidence and top-three predictions.
    """

    model, class_names = load_model()

    image_batch = prepare_image(pil_image)

    probabilities = model(
        image_batch,
        training=False,
    ).numpy()[0]

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top_three: list[dict[str, Any]] = []

    for class_index in top_indices:
        index = int(class_index)

        top_three.append(
            {
                "class_index": index,
                "class_name": class_names[index],
                "confidence": float(
                    probabilities[index]
                ),
            }
        )

    predicted_class = top_three[0]["class_name"]
    confidence = top_three[0]["confidence"]

    return predicted_class, confidence, top_three