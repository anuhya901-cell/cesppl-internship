from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src import classifier
from src import db
from src import pipeline
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CESPPL Field Image Tracker",
    page_icon="♻️",
    layout="wide",
)


# ---------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------

@st.cache_resource
def load_classifier_resources():
    """
    Load the trained model and class names only once.
    """
    return classifier.load_model()


# ---------------------------------------------------------
# Shared database initialization
# ---------------------------------------------------------

db.init_db()


# ---------------------------------------------------------
# Upload page
# ---------------------------------------------------------

def show_upload_page():
    st.title("♻️ CESPPL Field Image Classifier")

    st.write(
        "Upload a field image, classify it using the trained "
        "model, and save it automatically into the SQLite database."
    )

    try:
        load_classifier_resources()
    except Exception as error:
        st.error(
            "The trained model could not be loaded.\n\n"
            f"{error}"
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a field image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="field_image_uploader",
    )

    if uploaded_file is None:
        st.info(
            "Choose a JPG, JPEG, or PNG field image to begin."
        )
        return

    image_bytes = uploaded_file.getvalue()

    try:
        preview_image = Image.open(
            BytesIO(image_bytes)
        )

        preview_image.load()

        st.image(
            preview_image,
            caption=(
                f"Uploaded image: "
                f"{uploaded_file.name}"
            ),
            use_container_width=True,
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        st.error(
            "This file has an image extension, but its contents "
            "are not a valid image. Please upload a proper JPG, "
            "JPEG, or PNG image."
        )
        return

    classify_button = st.button(
        "Classify and save",
        type="primary",
        use_container_width=True,
    )

    if classify_button:
        try:
            with st.spinner(
                "Classifying and saving the image..."
            ):
                result = (
                    pipeline.classify_and_store(
                        image_bytes
                    )
                )

            st.success(
                "Image classified and saved successfully."
            )

            result_column_1, result_column_2 = (
                st.columns(2)
            )

            with result_column_1:
                st.metric(
                    "Predicted class",
                    result["class_name"],
                )

            with result_column_2:
                st.metric(
                    "Confidence",
                    (
                        f"{result['confidence']:.2%}"
                    ),
                )

            st.write(
                "**Stored filename:** "
                f"`{result['filename']}`"
            )

            st.write(
                "**Uploaded at:** "
                f"`{result['uploaded_at']}`"
            )

        except pipeline.InvalidImageError as error:
            st.error(
                "The uploaded file is not a valid image.\n\n"
                f"{error}"
            )

        except Exception as error:
            st.error(
                "The image could not be classified or saved.\n\n"
                f"{type(error).__name__}: {error}"
            )


# ---------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------

def show_dashboard_page():
    st.title("📊 Tracker Dashboard")

    st.write(
        "Live upload totals and per-class counts from the "
        "CESPPL SQLite database."
    )

    try:
        summary = db.dashboard_summary()
        counts = db.class_counts()
        recent_rows = db.recent_uploads(n=20)

    except Exception as error:
        st.error(
            "Dashboard data could not be loaded.\n\n"
            f"{type(error).__name__}: {error}"
        )
        return

    # -----------------------------------------------------
    # Top metrics
    # -----------------------------------------------------

    metric_column_1, metric_column_2, metric_column_3 = (
        st.columns(3)
    )

    with metric_column_1:
        st.metric(
            "Total uploads",
            summary["total_uploads"],
        )

    with metric_column_2:
        st.metric(
            "Classes with uploads",
            summary["active_classes"],
        )

    with metric_column_3:
        st.metric(
            "Uploads today",
            summary["uploads_today"],
        )

    st.divider()

    # -----------------------------------------------------
    # Per-class bar chart
    # -----------------------------------------------------

    st.subheader("Uploads by activity class")

    counts_dataframe = pd.DataFrame(
        {
            "Activity class": list(
                counts.keys()
            ),
            "Upload count": list(
                counts.values()
            ),
        }
    )

    counts_dataframe = (
        counts_dataframe
        .set_index("Activity class")
    )

    # Static per-class bar chart
    chart_dataframe = counts_dataframe.reset_index()

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.barh(
        chart_dataframe["Activity class"],
        chart_dataframe["Upload count"],
    )

    axis.set_title(
        "Uploads by activity class",
        fontsize=16,
    )

    axis.set_xlabel(
        "Number of uploads",
    )

    axis.set_ylabel(
        "Activity class",
    )

    axis.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    # Display the upload count at the end of every bar
    for index, upload_count in enumerate(
        chart_dataframe["Upload count"]
    ):
        axis.text(
            upload_count + 0.2,
            index,
            str(int(upload_count)),
            va="center",
        )

    figure.tight_layout()

    st.pyplot(
        figure,
        use_container_width=True,
    )

    plt.close(figure)

    with st.expander(
        "View exact class counts"
    ):
        st.dataframe(
            counts_dataframe,
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------
    # Recent uploads table
    # -----------------------------------------------------

    st.subheader("Recent uploads")

    if not recent_rows:
        st.info(
            "No uploaded images are currently stored."
        )
        return

    recent_dataframe = pd.DataFrame(
        recent_rows
    )

    recent_dataframe = recent_dataframe[
        [
            "filename",
            "class_name",
            "confidence",
            "uploaded_at",
        ]
    ].copy()

    # Convert decimal confidence (0.95) to percentage (95.00)
    recent_dataframe.loc[:, "confidence"] = (
        recent_dataframe["confidence"] * 100
    )

    recent_dataframe = recent_dataframe.rename(
        columns={
            "filename": "Filename",
            "class_name": "Class",
            "confidence": "Confidence",
            "uploaded_at": "Uploaded at",
        }
    )

    st.dataframe(
        recent_dataframe,
        column_config={
            "Filename": st.column_config.TextColumn(
                "Filename"
            ),
            "Class": st.column_config.TextColumn(
                "Class"
            ),
            "Confidence": (
                st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f%%",
                )
            ),
            "Uploaded at": (
                st.column_config.TextColumn(
                    "Uploaded at"
                )
            ),
        },
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Browse page
# ---------------------------------------------------------

def show_browse_page():
    st.title("🖼️ Browse Stored Images")

    st.write(
        "Choose an activity class to view its saved uploads."
    )

    try:
        counts = db.class_counts()

    except Exception as error:
        st.error(
            "The class list could not be loaded.\n\n"
            f"{type(error).__name__}: {error}"
        )
        return

    class_names = list(
        counts.keys()
    )

    selected_class = st.selectbox(
        "Select an activity class",
        options=class_names,
    )

    try:
        uploads = db.list_uploads(
            selected_class
        )

    except Exception as error:
        st.error(
            "Stored uploads could not be loaded.\n\n"
            f"{type(error).__name__}: {error}"
        )
        return

    st.metric(
        "Images in selected class",
        len(uploads),
    )

    if not uploads:
        st.info(
            f"No images are stored for "
            f"{selected_class}."
        )
        return

    uploads_dataframe = pd.DataFrame(
        uploads
    )

    uploads_dataframe = uploads_dataframe[
        [
            "id",
            "filename",
            "confidence",
            "uploaded_at",
        ]
    ]

    upload_options = {
        (
            f"{row['filename']} | "
            f"{row['confidence']:.2%} | "
            f"{row['uploaded_at']}"
        ): int(row["id"])
        for row in uploads
    }

    selected_upload_label = st.selectbox(
        "Select an uploaded image",
        options=list(
            upload_options.keys()
        ),
    )

    selected_upload_id = upload_options[
        selected_upload_label
    ]

    try:
        stored_image_bytes = db.get_image(
            selected_upload_id
        )

        if stored_image_bytes is None:
            st.warning(
                "The selected image was not found "
                "in the database."
            )

        else:
            stored_image = Image.open(
                BytesIO(stored_image_bytes)
            )

            st.image(
                stored_image,
                caption=selected_upload_label,
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            "The selected image could not be displayed.\n\n"
            f"{type(error).__name__}: {error}"
        )

    st.subheader(
        f"All uploads: {selected_class}"
    )

    uploads_dataframe = uploads_dataframe.copy()

    # Convert decimal confidence (0.95) to percentage (95.00)
    uploads_dataframe.loc[:, "confidence"] = (
        uploads_dataframe["confidence"] * 100
    )

    uploads_dataframe = uploads_dataframe.rename(
        columns={
            "id": "ID",
            "filename": "Filename",
            "confidence": "Confidence",
            "uploaded_at": "Uploaded at",
        }
    )

    st.dataframe(
        uploads_dataframe,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                format="%d",
            ),
            "Filename": st.column_config.TextColumn(
                "Filename"
            ),
            "Confidence": (
                st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f%%",
                )
            ),
            "Uploaded at": (
                st.column_config.TextColumn(
                    "Uploaded at"
                )
            ),
        },
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Application navigation
# ---------------------------------------------------------

st.sidebar.title("CESPPL Tracker")

selected_page = st.sidebar.radio(
    "Navigation",
    options=[
        "Upload",
        "Dashboard",
        "Browse",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "CESPPL Field Image Classification "
    "and Tracking System"
)


if selected_page == "Upload":
    show_upload_page()

elif selected_page == "Dashboard":
    show_dashboard_page()

elif selected_page == "Browse":
    show_browse_page()