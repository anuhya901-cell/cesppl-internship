from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "cesppl_dashboard.db"
CLASSES_FILE = REPO_ROOT / "CLASSES.md"


# Used only if CLASSES.md cannot be parsed correctly.
# The normal behaviour is to read the classes from CLASSES.md.
FALLBACK_CLASSES = [
    "BIN LIFTING",
    "BIN WASHING",
    "GATE MEETING",
    "LFC",
    "MANUAL BEACH CLEANING",
    "MECHANICAL SWEEPING",
    "MECHANIZED BEACH CLEANING",
    "PRIMARY COLLECTION",
    "ROAD SWEEPING",
    "SECONDARY VEHICLES",
]


# ---------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------

def _get_connection(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.

    The default database is:
    data/cesppl_dashboard.db
    """

    resolved_path = Path(
        db_path if db_path is not None else DEFAULT_DB_PATH
    )

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(resolved_path)

    # Allows rows to be accessed using column names.
    connection.row_factory = sqlite3.Row

    return connection


def _current_timestamp() -> datetime:
    """
    Return the current local date and time.

    This separate function also makes timestamp behaviour
    easy to test.
    """

    return datetime.now()


def _normalise_class_name(class_name: str) -> str:
    """
    Convert class names into the standard display format.

    Example:
    BIN_WASHING -> BIN WASHING
    """

    cleaned = str(class_name).strip().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    if not cleaned:
        raise ValueError("class_name cannot be empty.")

    return cleaned.upper()


def _filename_class_name(class_name: str) -> str:
    """
    Convert a class name into a filename-safe form.

    Example:
    BIN WASHING -> BIN_WASHING
    """

    normalised = _normalise_class_name(class_name)

    filename_value = re.sub(
        r"[^A-Z0-9]+",
        "_",
        normalised,
    )

    return filename_value.strip("_")


def _is_valid_class_name(value: str) -> bool:
    """
    Check whether a parsed CLASSES.md entry resembles
    one of the CESPPL activity classes.
    """

    cleaned = _normalise_class_name(value)

    return cleaned in FALLBACK_CLASSES


def load_class_names() -> list[str]:
    """
    Read the ten class names from CLASSES.md.

    Supported formats include Markdown bullet lists,
    numbered lists, and simple table rows.

    If the file cannot be parsed, the known ten classes
    are returned as a safe fallback.
    """

    if not CLASSES_FILE.exists():
        return FALLBACK_CLASSES.copy()

    text = CLASSES_FILE.read_text(
        encoding="utf-8"
    )

    parsed_classes: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Markdown bullet:
        # - BIN LIFTING
        # * BIN LIFTING
        bullet_match = re.match(
            r"^[-*+]\s+(.+?)\s*$",
            line,
        )

        # Numbered list:
        # 1. BIN LIFTING
        numbered_match = re.match(
            r"^\d+[.)]\s+(.+?)\s*$",
            line,
        )

        candidates: list[str] = []

        if bullet_match:
            candidates.append(
                bullet_match.group(1)
            )

        if numbered_match:
            candidates.append(
                numbered_match.group(1)
            )

        # Markdown table:
        # | BIN LIFTING | 0 |
        if line.startswith("|") and line.endswith("|"):
            cells = [
                cell.strip()
                for cell in line.strip("|").split("|")
            ]

            if cells:
                candidates.append(cells[0])

        for candidate in candidates:
            candidate = candidate.replace(
                "`",
                "",
            ).strip()

            if _is_valid_class_name(candidate):
                normalised = _normalise_class_name(
                    candidate
                )

                if normalised not in parsed_classes:
                    parsed_classes.append(normalised)

    if len(parsed_classes) == len(FALLBACK_CLASSES):
        return parsed_classes

    return FALLBACK_CLASSES.copy()


# ---------------------------------------------------------
# Public database functions
# ---------------------------------------------------------

def init_db(
    db_path: str | Path | None = None,
) -> Path:
    """
    Create the uploads table and class-name index
    if they do not already exist.

    Returns the resolved database path.
    """

    resolved_path = Path(
        db_path if db_path is not None else DEFAULT_DB_PATH
    ).resolve()

    with _get_connection(resolved_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                uploaded_at TEXT NOT NULL,
                image BLOB NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_uploads_class_name
            ON uploads(class_name)
            """
        )

        connection.commit()

    return resolved_path


def insert_upload(
    class_name: str,
    image_bytes: bytes,
    confidence: float,
    db_path: str | Path | None = None,
) -> str:
    """
    Insert one uploaded image and return its generated filename.

    Filename example:
    BIN_WASHING_2026-08-05_143512.jpg

    If the filename already exists, retry using:
    BIN_WASHING_2026-08-05_143512_1.jpg
    BIN_WASHING_2026-08-05_143512_2.jpg
    """

    if not isinstance(
        image_bytes,
        (bytes, bytearray),
    ):
        raise TypeError(
            "image_bytes must be bytes or bytearray."
        )

    if len(image_bytes) == 0:
        raise ValueError(
            "image_bytes cannot be empty."
        )

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "confidence must be a numeric value."
        ) from error

    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError(
            "confidence must be between 0.0 and 1.0."
        )

    normalised_class = _normalise_class_name(
        class_name
    )

    filename_class = _filename_class_name(
        normalised_class
    )

    timestamp = _current_timestamp()

    uploaded_at = timestamp.isoformat(
        timespec="seconds"
    )

    filename_timestamp = timestamp.strftime(
        "%Y-%m-%d_%H%M%S"
    )

    base_filename = (
        f"{filename_class}_{filename_timestamp}"
    )

    init_db(db_path)

    suffix_number = 0

    while True:
        if suffix_number == 0:
            filename = f"{base_filename}.jpg"
        else:
            filename = (
                f"{base_filename}_{suffix_number}.jpg"
            )

        try:
            with _get_connection(db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO uploads (
                        filename,
                        class_name,
                        confidence,
                        uploaded_at,
                        image
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        filename,
                        normalised_class,
                        confidence_value,
                        uploaded_at,
                        sqlite3.Binary(image_bytes),
                    ),
                )

                connection.commit()

            return filename

        except sqlite3.IntegrityError as error:
            # Only retry filename UNIQUE collisions.
            if "UNIQUE constraint failed" not in str(error):
                raise

            suffix_number += 1


def class_counts(
    db_path: str | Path | None = None,
) -> dict[str, int]:
    """
    Return all ten classes with their image counts.

    Classes with no uploads are included with a count of zero.
    """

    init_db(db_path)

    class_names = load_class_names()

    counts = {
        class_name: 0
        for class_name in class_names
    }

    with _get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT class_name, COUNT(*) AS image_count
            FROM uploads
            GROUP BY class_name
            ORDER BY class_name
            """
        ).fetchall()

    for row in rows:
        class_name = _normalise_class_name(
            row["class_name"]
        )

        counts[class_name] = int(
            row["image_count"]
        )

    return counts


def list_uploads(
    class_name: str,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return upload metadata for one class.

    The image BLOB is intentionally not returned.
    """

    init_db(db_path)

    normalised_class = _normalise_class_name(
        class_name
    )

    with _get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                confidence,
                uploaded_at
            FROM uploads
            WHERE class_name = ?
            ORDER BY uploaded_at DESC, id DESC
            """,
            (normalised_class,),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "filename": row["filename"],
            "confidence": float(
                row["confidence"]
            ),
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]


def get_image(
    upload_id: int,
    db_path: str | Path | None = None,
) -> bytes | None:
    """
    Fetch the original image bytes using the upload ID.

    Returns None when the ID does not exist.
    """

    init_db(db_path)

    with _get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT image
            FROM uploads
            WHERE id = ?
            """,
            (int(upload_id),),
        ).fetchone()

    if row is None:
        return None

    return bytes(row["image"])


def recent_uploads(
    n: int = 10,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return metadata for the most recent uploads.

    The image BLOB is intentionally not returned.
    """

    init_db(db_path)

    if n <= 0:
        return []

    with _get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                filename,
                class_name,
                confidence,
                uploaded_at
            FROM uploads
            ORDER BY uploaded_at DESC, id DESC
            LIMIT ?
            """,
            (int(n),),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "filename": row["filename"],
            "class_name": row["class_name"],
            "confidence": float(
                row["confidence"]
            ),
            "uploaded_at": row["uploaded_at"],
        }
        for row in rows
    ]