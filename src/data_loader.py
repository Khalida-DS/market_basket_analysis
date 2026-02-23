"""
data_loader.py — Data Ingestion and Validation
===============================================

WHY THIS FILE EXISTS:
In the original notebook, data loading was 2 lines with no safety.
This module treats data loading as a GATE — only clean, validated
data passes through to the analysis.

HOW TO USE:
    from src.data_loader import DataLoader

    loader = DataLoader()
    baskets_df, categories_df = loader.load_all()
"""

import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Tuple

from src.config import (
    BASKETS_FILE,
    CATEGORIES_FILE,
    BASKETS_SCHEMA,
    CATEGORIES_SCHEMA,
    ITEM_ID_MIN,
    ITEM_ID_MAX,
    LOG_LEVEL,
    LOG_FORMAT,
)

# ===========================================================
# LOGGER SETUP
# ===========================================================
# remove() clears the default handler first so we don't get
# duplicate output in the terminal.

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    colorize=True,
)


# ===========================================================
# DATA LOADER CLASS
# ===========================================================

class DataLoader:
    """
    Handles all data ingestion and validation for the pipeline.

    WHY A CLASS instead of just functions?
    - Stores shared state (file paths)
    - Easy to test — pass in test CSV paths
    - Easy to extend — subclass for Azure, S3, SQL later
    - Groups related methods together logically

    Usage:
        loader = DataLoader()
        baskets_df, categories_df = loader.load_all()
    """

    def __init__(
        self,
        baskets_path: Path = BASKETS_FILE,
        categories_path: Path = CATEGORIES_FILE,
    ):
        """
        Initialize DataLoader with file paths.

        WHY accept paths as arguments with defaults from config:
        - Default: uses paths from config.py (no arguments needed)
        - Testing: pass in a small test CSV file path
        - Flexible: override if data is in a different location

        Args:
            baskets_path:    Path to customer_baskets.csv
            categories_path: Path to clothing_categories.csv
        """
        self.baskets_path    = Path(baskets_path)
        self.categories_path = Path(categories_path)
        logger.info("DataLoader initialized")
        logger.debug(f"Baskets path:    {self.baskets_path}")
        logger.debug(f"Categories path: {self.categories_path}")

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and validate both datasets in one call.
        This is the main entry point used by main.py.

        Returns:
            Tuple of (baskets_df, categories_df)

        Example:
            loader = DataLoader()
            baskets_df, categories_df = loader.load_all()
        """
        logger.info("=" * 55)
        logger.info("Starting data loading pipeline")
        logger.info("=" * 55)

        baskets_df    = self.load_baskets()
        categories_df = self.load_categories()

        logger.info("=" * 55)
        logger.info("Data loading complete")
        logger.info(f"  Transactions : {len(baskets_df):,}")
        logger.info(f"  Categories   : {len(categories_df):,}")
        logger.info("=" * 55)

        return baskets_df, categories_df

    def load_baskets(self) -> pd.DataFrame:
        """
        Load, validate, and parse customer_baskets.csv.

        Steps:
        1. Check file exists
        2. Load CSV
        3. Validate required columns
        4. Check for nulls
        5. Parse basket string → list of integers
        6. Add basket_size column
        7. Validate item IDs are in catalog range

        Returns:
            pd.DataFrame with columns:
                customer_id  (int)       : customer identifier
                basket       (str)       : original comma-separated string
                basket_items (list[int]) : parsed list of item IDs
                basket_size  (int)       : number of items in basket
        """
        logger.info("Loading customer baskets...")

        # Step 1 — check file exists
        self._check_file_exists(self.baskets_path)

        # Step 2 — load CSV
        df = pd.read_csv(self.baskets_path)
        logger.info(f"  Rows loaded  : {len(df):,}")
        logger.info(f"  Columns      : {df.columns.tolist()}")

        # Step 3 — validate schema
        self._validate_schema(
            df,
            BASKETS_SCHEMA["required_columns"],
            "customer_baskets"
        )

        # Step 4 — check for nulls
        self._validate_no_nulls(
            df,
            BASKETS_SCHEMA["required_columns"],
            "customer_baskets"
        )

        # Step 5 — parse basket string into list of integers
        df = self._parse_basket_column(df)

        # Step 6 — add basket_size column
        df["basket_size"] = df["basket_items"].apply(len)
        logger.info(f"  Basket size  : {df['basket_size'].min()} – {df['basket_size'].max()}")
        logger.info(f"  Average size : {df['basket_size'].mean():.2f}")

        # Step 7 — validate item IDs
        self._validate_item_ids(df)

        logger.info("  ✓ Baskets validation passed")
        return df

    def load_categories(self) -> pd.DataFrame:
        """
        Load, validate, and clean clothing_categories.csv.

        Steps:
        1. Check file exists
        2. Load CSV
        3. Validate required columns
        4. Check for nulls
        5. Strip whitespace from text columns
        6. Validate category IDs in expected range

        Returns:
            pd.DataFrame with columns:
                category_id  (int) : item identifier
                name         (str) : category name
                description  (str) : category description
        """
        logger.info("Loading clothing categories...")

        # Step 1 — check file exists
        self._check_file_exists(self.categories_path)

        # Step 2 — load CSV
        # Note: original code used index_col=0 which turns category_id
        # into the index, making it hard to query. We load as a regular
        # column instead.
        df = pd.read_csv(self.categories_path)

        # Handle unnamed index column (common in exported CSVs)
        if df.columns[0].startswith("Unnamed"):
            df = df.drop(columns=df.columns[0])
            logger.debug("  Dropped unnamed index column")

        logger.info(f"  Rows loaded  : {len(df):,}")
        logger.info(f"  Columns      : {df.columns.tolist()}")

        # Step 3 — validate schema
        self._validate_schema(
            df,
            CATEGORIES_SCHEMA["required_columns"],
            "clothing_categories"
        )

        # Step 4 — check for nulls
        self._validate_no_nulls(
            df,
            CATEGORIES_SCHEMA["required_columns"],
            "clothing_categories"
        )

        # Step 5 — strip whitespace
        # Original code only stripped 'description'.
        # We strip ALL text columns properly.
        df["name"]        = df["name"].str.strip()
        df["description"] = df["description"].str.strip()
        logger.debug("  Stripped whitespace from text columns")

        # Step 6 — validate category IDs
        invalid_ids = df[
            (df["category_id"] < ITEM_ID_MIN) |
            (df["category_id"] > ITEM_ID_MAX)
        ]
        if not invalid_ids.empty:
            logger.warning(
                f"  {len(invalid_ids)} categories outside expected range "
                f"({ITEM_ID_MIN}–{ITEM_ID_MAX})"
            )

        logger.info("  ✓ Categories validation passed")
        return df

    # -------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------
    # Methods starting with _ are internal helpers.
    # They are only called by methods inside this class.
    # Other modules should not call these directly.

    def _check_file_exists(self, path: Path) -> None:
        """
        Check file exists and raise a clear error if not.

        WHY NOT just let pd.read_csv fail?
        pd.read_csv raises a confusing system error.
        We raise a clear message that tells you exactly what to do.

        Args:
            path: file path to check

        Raises:
            FileNotFoundError: with clear, actionable message
        """
        if not path.exists():
            raise FileNotFoundError(
                f"\n\n  File not found : {path.name}"
                f"\n  Expected at    : {path.resolve()}"
                f"\n  Action needed  : place your CSV files in data/raw/"
                f"\n  See README.md for setup instructions.\n"
            )
        logger.debug(f"  File found: {path.name}")

    def _validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: list,
        file_name: str,
    ) -> None:
        """
        Validate all required columns are present.

        WHY: If someone renames a column in the CSV, we catch it
        here immediately with a message listing the missing column.

        Args:
            df:               DataFrame to validate
            required_columns: list of column names that must exist
            file_name:        name of file for error messages

        Raises:
            ValueError: listing exactly which columns are missing
        """
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"\n\n  Schema error in '{file_name}'"
                f"\n  Missing columns : {missing}"
                f"\n  Columns found   : {df.columns.tolist()}\n"
            )
        logger.debug("  Schema validation passed")

    def _validate_no_nulls(
        self,
        df: pd.DataFrame,
        columns: list,
        file_name: str,
    ) -> None:
        """
        Validate no missing values in critical columns.

        WHY: A null in the basket column causes the parsing step
        to crash with a confusing error. Better to catch it here.

        Args:
            df:        DataFrame to check
            columns:   columns to check for nulls
            file_name: name of file for error messages

        Raises:
            ValueError: if any null values found
        """
        null_counts = df[columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            raise ValueError(
                f"\n\n  Null values found in '{file_name}'"
                f"\n{columns_with_nulls.to_string()}"
                f"\n  Action needed: clean data before running pipeline.\n"
            )
        logger.debug("  Null check passed")

    def _parse_basket_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse basket column from string to list of integers.

        Original code (crashes silently on bad data):
            df['basket_items'] = df["basket"].str.split(",")
                                 .apply(lambda x: [int(i) for i in x])

        Our version:
            - strips whitespace around item IDs
            - catches individual parsing errors
            - reports which rows had problems
            - filters out invalid rows instead of crashing

        Args:
            df: DataFrame with 'basket' column

        Returns:
            DataFrame with new 'basket_items' column (list of ints)
        """
        logger.debug("  Parsing basket strings...")

        def parse_single_basket(basket_str: str, row_index: int) -> list:
            """Parse '1,5,12,23' → [1, 5, 12, 23]"""
            try:
                return [
                    int(item.strip())
                    for item in basket_str.split(",")
                    if item.strip()
                ]
            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"  Could not parse row {row_index}: "
                    f"'{basket_str}' — {e}"
                )
                return []

        df["basket_items"] = [
            parse_single_basket(basket, idx)
            for idx, basket in zip(df.index, df["basket"])
        ]

        # Remove rows where parsing completely failed
        invalid_rows = df[df["basket_items"].apply(len) == 0]
        if not invalid_rows.empty:
            logger.warning(
                f"  Dropping {len(invalid_rows)} unparseable rows"
            )
            df = df[df["basket_items"].apply(len) > 0].copy()

        logger.debug(f"  Parsing complete: {len(df):,} valid rows")
        return df

    def _validate_item_ids(self, df: pd.DataFrame) -> None:
        """
        Validate all item IDs are within the known catalog range.

        WHY: Original code used range(1, 49) as a magic number.
        If a basket had item ID 0 or 55, it was silently ignored.
        Now we know about it immediately.

        Args:
            df: DataFrame with 'basket_items' column
        """
        all_items = pd.Series(
            [item for basket in df["basket_items"] for item in basket],
            dtype="int64"
        )

        out_of_range = all_items[
            (all_items < ITEM_ID_MIN) | (all_items > ITEM_ID_MAX)
        ].unique()

        if len(out_of_range) > 0:
            logger.warning(
                f"  Item IDs outside range "
                f"({ITEM_ID_MIN}–{ITEM_ID_MAX}): "
                f"{sorted(out_of_range.tolist())}"
            )
        else:
            logger.debug(
                f"  All item IDs valid "
                f"(range {ITEM_ID_MIN}–{ITEM_ID_MAX})"
            )

        logger.info(f"  Unique items  : {all_items.nunique()}")
        logger.info(f"  Total purchases: {len(all_items):,}")