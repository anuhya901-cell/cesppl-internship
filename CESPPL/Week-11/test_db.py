from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

import src.db as db


def create_test_image_bytes() -> bytes:
    """
    Create a small valid JPEG image entirely in memory.
    """

    image = Image.new(
        mode="RGB",
        size=(32, 32),
        color=(120, 180, 90),
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def test_init_db_creates_database(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    result = db.init_db(database_path)

    assert result.exists()
    assert result == database_path.resolve()


def test_insert_and_fetch_image(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    image_bytes = create_test_image_bytes()

    filename = db.insert_upload(
        class_name="BIN WASHING",
        image_bytes=image_bytes,
        confidence=0.95,
        db_path=database_path,
    )

    uploads = db.list_uploads(
        class_name="BIN WASHING",
        db_path=database_path,
    )

    assert len(uploads) == 1
    assert uploads[0]["filename"] == filename
    assert uploads[0]["confidence"] == 0.95

    fetched_bytes = db.get_image(
        upload_id=uploads[0]["id"],
        db_path=database_path,
    )

    assert fetched_bytes == image_bytes


def test_class_counts_includes_all_classes(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    image_bytes = create_test_image_bytes()

    db.insert_upload(
        class_name="ROAD SWEEPING",
        image_bytes=image_bytes,
        confidence=0.88,
        db_path=database_path,
    )

    counts = db.class_counts(
        db_path=database_path
    )

    assert len(counts) == 10

    assert counts["ROAD SWEEPING"] == 1
    assert counts["BIN LIFTING"] == 0
    assert counts["GATE MEETING"] == 0

    assert sum(counts.values()) == 1


def test_list_uploads_filters_by_class(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    image_bytes = create_test_image_bytes()

    db.insert_upload(
        class_name="BIN LIFTING",
        image_bytes=image_bytes,
        confidence=0.91,
        db_path=database_path,
    )

    db.insert_upload(
        class_name="ROAD SWEEPING",
        image_bytes=image_bytes,
        confidence=0.92,
        db_path=database_path,
    )

    bin_lifting_uploads = db.list_uploads(
        class_name="BIN LIFTING",
        db_path=database_path,
    )

    assert len(bin_lifting_uploads) == 1

    assert (
        bin_lifting_uploads[0]["confidence"]
        == 0.91
    )

    assert "image" not in bin_lifting_uploads[0]


def test_recent_uploads(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    image_bytes = create_test_image_bytes()

    db.insert_upload(
        class_name="BIN WASHING",
        image_bytes=image_bytes,
        confidence=0.90,
        db_path=database_path,
    )

    db.insert_upload(
        class_name="PRIMARY COLLECTION",
        image_bytes=image_bytes,
        confidence=0.93,
        db_path=database_path,
    )

    rows = db.recent_uploads(
        n=1,
        db_path=database_path,
    )

    assert len(rows) == 1

    assert set(rows[0].keys()) == {
        "id",
        "filename",
        "class_name",
        "confidence",
        "uploaded_at",
    }


def test_filename_uniqueness_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    image_bytes = create_test_image_bytes()

    fixed_timestamp = datetime(
        2026,
        8,
        5,
        14,
        35,
        12,
    )

    monkeypatch.setattr(
        db,
        "_current_timestamp",
        lambda: fixed_timestamp,
    )

    first_filename = db.insert_upload(
        class_name="BIN WASHING",
        image_bytes=image_bytes,
        confidence=0.95,
        db_path=database_path,
    )

    second_filename = db.insert_upload(
        class_name="BIN WASHING",
        image_bytes=image_bytes,
        confidence=0.96,
        db_path=database_path,
    )

    third_filename = db.insert_upload(
        class_name="BIN WASHING",
        image_bytes=image_bytes,
        confidence=0.97,
        db_path=database_path,
    )

    assert (
        first_filename
        == "BIN_WASHING_2026-08-05_143512.jpg"
    )

    assert (
        second_filename
        == "BIN_WASHING_2026-08-05_143512_1.jpg"
    )

    assert (
        third_filename
        == "BIN_WASHING_2026-08-05_143512_2.jpg"
    )

    uploads = db.list_uploads(
        class_name="BIN WASHING",
        db_path=database_path,
    )

    assert len(uploads) == 3


def test_missing_image_returns_none(
    tmp_path: Path,
) -> None:
    database_path = (
        tmp_path / "test_dashboard.db"
    )

    db.init_db(database_path)

    result = db.get_image(
        upload_id=999999,
        db_path=database_path,
    )

    assert result is None