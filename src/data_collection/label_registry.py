"""
label_registry.py
-----------------
Loads and queries csv/labels.csv for the FSL data collection pipeline.

This is the single source of truth for available categories and labels.
It has no knowledge of webcam access, sequence recording, or model training.

Reusable by:
  - Phase 1 CLI (scripts/demo_collection.py)
  - Phase 2 Vue/web interface
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LABELS_CSV = PROJECT_ROOT / "csv" / "labels.csv"

REQUIRED_COLUMNS = {"id", "label", "category"}


class LabelRegistry:
    """
    Loads csv/labels.csv and provides category/label queries.

    All category and label strings are stored and returned in UPPERCASE
    to ensure consistent matching regardless of how they appear in the CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to labels.csv. Defaults to <project_root>/csv/labels.csv.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns (id, label, category) are missing,
        or if the CSV contains no valid rows.
    """

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path else DEFAULT_LABELS_CSV
        self._df = self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        """Load and validate the CSV. Returns a normalised DataFrame."""
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found at: {self._csv_path}"
            )

        df = pd.read_csv(self._csv_path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"labels.csv is missing required columns: {missing}"
            )

        # Normalise to uppercase for consistent matching
        df["label"] = df["label"].str.strip().str.upper()
        df["category"] = df["category"].str.strip().str.upper()

        df = df.dropna(subset=["id", "label", "category"])

        if df.empty:
            raise ValueError("labels.csv contains no valid rows.")

        logger.debug(
            "LabelRegistry: loaded %d labels across %d categories from %s",
            len(df),
            df["category"].nunique(),
            self._csv_path,
        )
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_categories(self) -> list[str]:
        """
        Return a sorted list of all unique category names (UPPERCASE).

        Returns
        -------
        list[str]
            e.g. ['CALENDAR', 'COLOR', 'DAYS', 'DRINK', ...]
        """
        return sorted(self._df["category"].unique().tolist())

    def get_labels(self, category: str) -> list[str]:
        """
        Return labels belonging to *category* in their original CSV order.

        Parameters
        ----------
        category : str
            Category name (case-insensitive).

        Returns
        -------
        list[str]
            Ordered list of UPPERCASE label strings.
            Empty list if the category has no labels.
        """
        key = category.strip().upper()
        subset = self._df[self._df["category"] == key]
        return subset["label"].tolist()

    def get_label_id(self, label: str, category: str) -> int | None:
        """
        Return the integer ID for a given label within a category.

        Parameters
        ----------
        label : str
            Label name (case-insensitive).
        category : str
            Category name (case-insensitive).

        Returns
        -------
        int or None
            The label ID if found, else None.
        """
        lkey = label.strip().upper()
        ckey = category.strip().upper()
        row = self._df[
            (self._df["label"] == lkey) & (self._df["category"] == ckey)
        ]
        if row.empty:
            return None
        return int(row.iloc[0]["id"])

    def get_category_for_label(self, label: str) -> str | None:
        """
        Return the category that owns *label*, or None if not found.

        Parameters
        ----------
        label : str
            Label name (case-insensitive).

        Returns
        -------
        str or None
        """
        key = label.strip().upper()
        row = self._df[self._df["label"] == key]
        if row.empty:
            return None
        return str(row.iloc[0]["category"])
