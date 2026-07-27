"""
Classification and storage pipeline for the CESPPL dashboard.

This module connects the trained image classifier with the SQLite
storage layer.

Main public function:
    classify_and_store(image_bytes)
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from src import classifier
from src import db


# ---------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------

MAX_LONG_EDGE = 1600
JPEG_QUALITY = 90


# ---------------------------------------------------------
# Custom exception
# ---------------------------------------------------------

class InvalidImageError(ValueError):
    """
    Raised when uploaded bytes cannot be opened as an image.
    """


# ---------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------

def _validate_image_bytes(
    image_bytes: bytes | bytearray,
) -> bytes:
    """
    Validate the incoming image bytes and return immutable bytes.
    """

    if not isinstance(
        image_bytes,
        (bytes, bytearray),
    ):
        raise TypeError(
            "image_bytes must be bytes or bytearray."
        )

    if len(image_bytes) == 0:
        raise InvalidImageError(
            "The uploaded image is empty."
        )

    return bytes(image_bytes)


def _prepare_image(
    image_bytes: bytes,
) -> tuple[Image.Image, bytes]:
    """
    Open and standardise an uploaded image.

    Processing steps:
    1. Validate the image.
    2. Correct EXIF orientation.
    3. Convert to RGB.
    4. Resize only when the longest edge exceeds 1600 pixels.
    5. Encode as JPEG with quality 90.

    Storage rationale:
    The standardised JPEG bytes are stored instead of the original upload
    so database image sizes remain predictable and large phone photographs
    do not unnecessarily increase database storage.
    """

    try:
        with Image.open(
            BytesIO(image_bytes)
        ) as opened_image:

            # Force Pillow to decode the full image while the
            # underlying BytesIO object is still available.
            opened_image.load()

            prepared_image = ImageOps.exif_transpose(
                opened_image
            )

            prepared_image = prepared_image.convert(
                "RGB"
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise InvalidImageError(
            "The uploaded file is not a valid supported image."
        ) from error

    width, height = prepared_image.size

    if width <= 0 or height <= 0:
        raise InvalidImageError(
            "The uploaded image has invalid dimensions."
        )

    if max(width, height) > MAX_LONG_EDGE:
        prepared_image.thumbnail(
            (
                MAX_LONG_EDGE,
                MAX_LONG_EDGE,
            ),
            Image.Resampling.LANCZOS,
        )

    output_buffer = BytesIO()

    prepared_image.save(
        output_buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )

    # This is the important correction:
    # getvalue() returns actual bytes rather than a BytesIO object.
    processed_bytes = output_buffer.getvalue()

    if not processed_bytes:
        raise InvalidImageError(
            "The image could not be encoded."
        )

    return prepared_image, processed_bytes


def _normalise_class_name(
    class_name: Any,
) -> str:
    """
    Convert classifier output into the standard class-name format.
    """

    class_value = str(class_name).strip()

    if not class_value:
        raise ValueError(
            "The classifier returned an empty class name."
        )

    return (
        class_value
        .replace("_", " ")
        .upper()
    )


def _normalise_confidence(
    confidence: Any,
) -> float:
    """
    Convert confidence to a float between 0 and 1.

    Both formats are supported:
    0.97 and 97.0.
    """

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The classifier returned an invalid confidence value."
        ) from error

    # Support classifiers that return percentages.
    if 1.0 < confidence_value <= 100.0:
        confidence_value /= 100.0

    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError(
            "Classifier confidence must be between 0 and 1."
        )

    return confidence_value


def _normalise_prediction(
    prediction: Any,
) -> tuple[str, float]:
    """
    Support common prediction response formats.

    Supported dictionaries:
    {
        "class_name": "BIN WASHING",
        "confidence": 0.98
    }

    {
        "class": "BIN WASHING",
        "confidence": 98.0
    }

    Supported tuple:
    ("BIN WASHING", 0.98)
    """

    class_name: Any = None
    confidence: Any = None

    if isinstance(prediction, dict):
        class_name = (
            prediction.get("class_name")
            or prediction.get("class")
            or prediction.get("label")
            or prediction.get("prediction")
        )

        confidence = (
            prediction.get("confidence")
            if "confidence" in prediction
            else prediction.get("score")
        )

    elif isinstance(
        prediction,
        (tuple, list),
    ) and len(prediction) >= 2:
        class_name = prediction[0]
        confidence = prediction[1]

    else:
        class_name = getattr(
            prediction,
            "class_name",
            None,
        )

        confidence = getattr(
            prediction,
            "confidence",
            None,
        )

    if class_name is None:
        raise ValueError(
            "The classifier result does not contain a class name."
        )

    if confidence is None:
        raise ValueError(
            "The classifier result does not contain confidence."
        )

    return (
        _normalise_class_name(class_name),
        _normalise_confidence(confidence),
    )


def _get_uploaded_at(
    filename: str,
) -> str:
    """
    Retrieve the exact database timestamp for the inserted upload.
    """

    try:
        recent_rows = db.recent_uploads(n=10)

        for row in recent_rows:
            if row.get("filename") == filename:
                return str(
                    row.get("uploaded_at")
                )

    except Exception:
        # The upload itself has already succeeded.
        # Timestamp fallback prevents a metadata lookup problem
        # from breaking the whole pipeline.
        pass

    return datetime.now().isoformat(
        timespec="seconds"
    )


# ---------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------

def classify_and_store(
    image_bytes: bytes | bytearray,
) -> dict[str, Any]:
    """
    Validate, standardise, classify and store one uploaded image.

    Returns:
    {
        "filename": str,
        "class_name": str,
        "confidence": float,
        "uploaded_at": str
    }
    """

    validated_bytes = _validate_image_bytes(
        image_bytes
    )

    prepared_image, processed_bytes = _prepare_image(
        validated_bytes
    )

    # The classifier receives the prepared PIL RGB image.
    prediction = classifier.predict_image(
        prepared_image
    )

    class_name, confidence = _normalise_prediction(
        prediction
    )

    # processed_bytes is a real bytes object.
    # Do not pass output_buffer or prepared_image here.
    filename = db.insert_upload(
        class_name=class_name,
        image_bytes=processed_bytes,
        confidence=confidence,
    )

    uploaded_at = _get_uploaded_at(
        filename
    )

    return {
        "filename": filename,
        "class_name": class_name,
        "confidence": confidence,
        "uploaded_at": uploaded_at,
    }