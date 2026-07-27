from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src import classifier
from src import db
from src import pipeline
from src.pipeline import InvalidImageError


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CESPPL Field Image Classifier",
    page_icon="♻️",
    layout="centered",
)


# =========================================================
# LOAD MODEL ONLY ONCE
# =========================================================

@st.cache_resource
def load_cached_model() -> tuple[Any, list[str]]:
    """
    Load the trained TensorFlow model and class names once.

    Streamlit reruns the script after every interaction.
    @st.cache_resource prevents the model from reloading
    on every upload or button click.
    """
    return classifier.load_model()


# =========================================================
# CREATE CLASS-COUNT TABLE
# =========================================================

def create_class_counts_table(
    counts: dict[str, int],
) -> pd.DataFrame:
    """
    Convert the dictionary returned by db.class_counts()
    into a Streamlit-friendly table.
    """

    rows = [
        {
            "Class": class_name,
            "Stored Images": count,
        }
        for class_name, count in counts.items()
    ]

    return pd.DataFrame(rows)


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("♻️ CESPPL Field Image Classifier")

st.write(
    "Upload a field image, classify it using the trained model, "
    "and save it automatically into the SQLite database."
)


# =========================================================
# INITIALISE DATABASE
# =========================================================

try:
    database_path = db.init_db()
except Exception as error:
    st.error(
        "The database could not be initialized.\n\n"
        f"Details: {error}"
    )
    st.stop()


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:
    model, class_names = load_cached_model()

except FileNotFoundError as error:
    st.error(
        "The trained model or class-names file was not found.\n\n"
        f"Details: {error}"
    )
    st.stop()

except Exception as error:
    st.error(
        "The trained model could not be loaded.\n\n"
        f"Details: {error}"
    )
    st.stop()


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a field image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG",
)


# =========================================================
# IMAGE PREVIEW, CLASSIFICATION AND STORAGE
# =========================================================

if uploaded_file is not None:

    try:
        # getvalue() returns the uploaded file as real bytes.
        image_bytes = uploaded_file.getvalue()

        if not image_bytes:
            raise InvalidImageError(
                "The uploaded file is empty."
            )

        # Open the uploaded bytes for browser preview.
        preview_image = Image.open(uploaded_file)
        preview_image.load()
        preview_image = preview_image.convert("RGB")

        st.image(
            preview_image,
            caption=f"Uploaded image: {uploaded_file.name}",
            width="stretch",
        )

        classify_button = st.button(
            "Classify and save",
            type="primary",
            width="stretch",
        )

        if classify_button:

            try:
                with st.spinner(
                    "Classifying the image and saving it..."
                ):
                    result = pipeline.classify_and_store(
                        image_bytes
                    )

                predicted_class = result["class_name"]
                confidence = float(result["confidence"])
                stored_filename = result["filename"]
                uploaded_at = result["uploaded_at"]

                confidence_percentage = confidence * 100

                st.success(
                    f"Predicted class: {predicted_class}\n\n"
                    f"Confidence: {confidence_percentage:.2f}%\n\n"
                    f"Stored filename: {stored_filename}\n\n"
                    f"Uploaded at: {uploaded_at}"
                )

            except InvalidImageError as error:
                st.error(
                    "The uploaded file is not a valid image. "
                    "Please upload a proper JPG, JPEG or PNG file.\n\n"
                    f"Details: {error}"
                )

            except TypeError as error:
                st.error(
                    "The uploaded image could not be processed because "
                    "its data format was invalid.\n\n"
                    f"Details: {error}"
                )

            except Exception as error:
                st.error(
                    "The image could not be classified or saved.\n\n"
                    f"Details: {error}"
                )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        InvalidImageError,
    ):
        st.error(
            "This file has an image extension, but its contents are "
            "not a valid image. Please upload a proper JPG, JPEG or "
            "PNG image."
        )


# =========================================================
# STORED IMAGE TRACKER
# =========================================================

st.divider()
st.subheader("Stored Image Tracker")

try:
    counts = db.class_counts()

    counts_dataframe = create_class_counts_table(
        counts
    )

    total_stored_images = int(
        counts_dataframe["Stored Images"].sum()
    )

    st.metric(
        label="Total stored images",
        value=total_stored_images,
    )

    st.dataframe(
        counts_dataframe,
        width="stretch",
        hide_index=True,
    )

except Exception as error:
    st.error(
        "The stored-image counts could not be loaded.\n\n"
        f"Details: {error}"
    )


# =========================================================
# APPLICATION INFORMATION
# =========================================================

with st.expander("Model information"):
    st.write(f"Number of classes: **{len(class_names)}**")

    st.write("Supported classes:")

    for index, class_name in enumerate(
        class_names,
        start=1,
    ):
        st.write(f"{index}. {class_name}")