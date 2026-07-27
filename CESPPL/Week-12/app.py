from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime
from io import BytesIO
from math import ceil

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
# Prevent duplicate processing
# ---------------------------------------------------------

if "processing_upload" not in st.session_state:
    st.session_state.processing_upload = False

if "last_saved_image_hash" not in st.session_state:
    st.session_state.last_saved_image_hash = None


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

    current_image_hash = hashlib.sha256(
        image_bytes
    ).hexdigest()

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
        disabled=st.session_state.processing_upload,
    )

    if classify_button:
        if (
            st.session_state.last_saved_image_hash
            == current_image_hash
        ):
            st.warning(
                "This image was already saved. "
                "It was not stored again."
            )
            return

        if st.session_state.processing_upload:
            st.stop()

        st.session_state.processing_upload = True
        try:
            with st.spinner(
                "Classifying and saving the image..."
            ):
                result = (
                    pipeline.classify_and_store(
                        image_bytes
                    )
                )

            st.session_state.last_saved_image_hash = (
                current_image_hash
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
        except pipeline.DuplicateImageError as error:
            st.warning(
                "This image was already saved, so it was not "
                "stored again.\n\n"
                f"{error}"
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
        finally:
            st.session_state.processing_upload = False


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
# Browse-page configuration
# ---------------------------------------------------------

IMAGES_PER_PAGE = 24
THUMBNAIL_SIZE = (300, 300)


# ---------------------------------------------------------
# Whole-class ZIP helper
# ---------------------------------------------------------

def build_class_zip(
    uploads: list[dict],
) -> bytes:
    """
    Build an in-memory ZIP file containing all images
    represented by the supplied upload metadata.

    Image BLOBs are fetched only when this function is called.
    """

    zip_buffer = BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for upload in uploads:
            upload_id = int(upload["id"])
            filename = str(upload["filename"])

            image_bytes = db.get_image(
                upload_id
            )

            if image_bytes is None:
                continue

            zip_file.writestr(
                filename,
                image_bytes,
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ---------------------------------------------------------
# Browse page
# ---------------------------------------------------------

def show_browse_page():
    st.title("🖼️ Browse Stored Images")

    st.write(
        "Choose an activity class to browse its saved images, "
        "view image details, and download individual images "
        "or the complete class."
    )

    # -----------------------------------------------------
    # Load class names and counts
    # -----------------------------------------------------

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
        key="browse_class_selector",
    )

    selected_class_count = int(
        counts.get(
            selected_class,
            0,
        )
    )

    st.caption(
        f"{selected_class_count} uploaded image"
        f"{'' if selected_class_count == 1 else 's'} "
        f"in {selected_class}."
    )

    # -----------------------------------------------------
    # Load metadata only — no image BLOBs here
    # -----------------------------------------------------

    try:
        uploads = db.list_uploads(
            selected_class
        )

    except Exception as error:
        st.error(
            "The stored upload list could not be loaded.\n\n"
            f"{type(error).__name__}: {error}"
        )
        return

    # -----------------------------------------------------
    # Empty-class handling
    # -----------------------------------------------------

    if not uploads:
        st.info(
            f"No uploads yet for {selected_class}. "
            "Images classified into this class will appear here."
        )
        return

    # -----------------------------------------------------
    # Whole-class download
    # -----------------------------------------------------

    class_filename = (
        selected_class
        .replace(" ", "_")
        .replace("/", "_")
    )

    current_date = datetime.now().strftime(
        "%Y-%m-%d"
    )

    zip_filename = (
        f"{class_filename}_{current_date}.zip"
    )

    st.subheader("Download complete class")

    st.caption(
        "Preparing the complete ZIP fetches all images in "
        "this class. Gallery browsing fetches only the visible page."
    )

    prepare_zip_button = st.button(
        (
            f"Prepare {selected_class} ZIP "
            f"({len(uploads)} images)"
        ),
        key=(
            f"prepare_zip_{class_filename}"
        ),
    )

    zip_state_key = (
        f"prepared_zip_{class_filename}"
    )

    if prepare_zip_button:
        try:
            with st.spinner(
                f"Preparing {len(uploads)} images..."
            ):
                st.session_state[
                    zip_state_key
                ] = build_class_zip(
                    uploads
                )

            st.success(
                "The class ZIP is ready."
            )

        except Exception as error:
            st.error(
                "The class ZIP could not be created.\n\n"
                f"{type(error).__name__}: {error}"
            )

    if zip_state_key in st.session_state:
        st.download_button(
            label=(
                f"Download {selected_class} "
                f"({len(uploads)} images)"
            ),
            data=st.session_state[
                zip_state_key
            ],
            file_name=zip_filename,
            mime="application/zip",
            key=(
                f"download_zip_{class_filename}"
            ),
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------
    # Pagination
    # -----------------------------------------------------

    total_uploads = len(uploads)

    total_pages = max(
        1,
        ceil(
            total_uploads
            / IMAGES_PER_PAGE
        ),
    )

    page_numbers = list(
        range(
            1,
            total_pages + 1,
        )
    )

    page_column, information_column = (
        st.columns(
            [1, 3]
        )
    )

    with page_column:
        selected_page = st.selectbox(
            "Page",
            options=page_numbers,
            index=0,
            key=(
                f"browse_page_{class_filename}"
            ),
        )

    start_index = (
        selected_page - 1
    ) * IMAGES_PER_PAGE

    end_index = min(
        start_index
        + IMAGES_PER_PAGE,
        total_uploads,
    )

    visible_uploads = uploads[
        start_index:end_index
    ]

    with information_column:
        st.write("")
        st.write(
            f"Showing images "
            f"**{start_index + 1}–{end_index}** "
            f"of **{total_uploads}**"
        )

    st.subheader(
        f"{selected_class} gallery"
    )

    # -----------------------------------------------------
    # Four-column thumbnail gallery
    # -----------------------------------------------------

    gallery_columns = st.columns(
        4
    )

    for position, upload in enumerate(
        visible_uploads
    ):
        upload_id = int(
            upload["id"]
        )

        filename = str(
            upload["filename"]
        )

        confidence = float(
            upload["confidence"]
        )

        uploaded_at = str(
            upload["uploaded_at"]
        )

        current_column = gallery_columns[
            position % 4
        ]

        with current_column:
            try:
                # Only visible-page image BLOBs are fetched.
                image_bytes = db.get_image(
                    upload_id
                )

                if image_bytes is None:
                    st.warning(
                        f"{filename} could not be found."
                    )
                    continue

                with Image.open(
                    BytesIO(image_bytes)
                ) as opened_image:

                    opened_image.load()

                    full_image = (
                        opened_image
                        .convert("RGB")
                        .copy()
                    )

                thumbnail_image = (
                    full_image.copy()
                )

                thumbnail_image.thumbnail(
                    THUMBNAIL_SIZE,
                    Image.Resampling.LANCZOS,
                )

                st.image(
                    thumbnail_image,
                    caption=filename,
                    use_container_width=True,
                )

                with st.expander(
                    "View full image and details"
                ):
                    st.image(
                        full_image,
                        caption=filename,
                        use_container_width=True,
                    )

                    st.write(
                        f"**Filename:** `{filename}`"
                    )

                    st.write(
                        f"**Confidence:** "
                        f"{confidence:.2%}"
                    )

                    st.write(
                        f"**Uploaded at:** "
                        f"`{uploaded_at}`"
                    )
                    st.markdown("---")

                st.write("### Wrong class?")

                st.caption(
                    "Select the correct activity class and reassign "
                    "this stored image."
                )

                available_classes = [
                    class_name
                    for class_name in class_names
                    if class_name != selected_class
                ]

                new_class = st.selectbox(
                    "Correct activity class",
                    options=available_classes,
                    key=f"reassign_class_{upload_id}",
                )

                reassign_button = st.button(
                    "Reassign image",
                    key=f"reassign_button_{upload_id}",
                    use_container_width=True,
                )

                if reassign_button:
                    try:
                        updated = db.reassign_upload(
                            upload_id=upload_id,
                            new_class_name=new_class,
                        )

                        if updated:
                            st.success(
                                f"Image reassigned from "
                                f"{selected_class} to {new_class}."
                            )

                            st.rerun()

                        else:
                            st.error(
                                "The selected image could not be found."
                            )

                    except Exception as error:
                        st.error(
                            "The image could not be reassigned.\n\n"
                            f"{type(error).__name__}: {error}"
                        )

                    st.download_button(
                        label="Download image",
                        data=image_bytes,
                        file_name=filename,
                        mime="image/jpeg",
                        key=(
                            f"download_image_"
                            f"{upload_id}"
                        ),
                        use_container_width=True,
                    )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as error:
                st.error(
                    f"{filename} is not a readable image.\n\n"
                    f"{type(error).__name__}: {error}"
                )

            except Exception as error:
                st.error(
                    f"{filename} could not be displayed.\n\n"
                    f"{type(error).__name__}: {error}"
                )

    st.divider()

    st.caption(
        f"Page {selected_page} of {total_pages}"
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