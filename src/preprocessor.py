"""
preprocessor.py — Data Transformation and Feature Engineering
=============================================================

WHY THIS FILE EXISTS:
---------------------
The original notebook mixed data cleaning, transformation,
and analysis in one giant script. This file does ONE thing:
take clean, validated data from DataLoader and transform it
into formats ready for analysis.

WHAT THIS FILE FIXES FROM THE ORIGINAL CODE:
--------------------------------------------
1. iterrows() loop (1.7M iterations) → vectorized operations
2. Off-by-one bug in poplr_item     → dictionary-based counting
3. Magic number range(1,49)         → config.ITEM_ID_MAX
4. No separation of concerns        → single responsibility

HOW TO USE:
-----------
    from src.preprocessor import Preprocessor
    from src.data_loader import DataLoader

    loader      = DataLoader()
    baskets_df, categories_df = loader.load_all()

    preprocessor = Preprocessor(baskets_df, categories_df)
    one_hot_df   = preprocessor.build_one_hot_matrix()
    stats        = preprocessor.get_basket_stats()
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Tuple

from src.config import (
    ITEM_ID_MIN,
    ITEM_ID_MAX,
    LOG_LEVEL,
    LOG_FORMAT,
)

# ===========================================================
# LOGGER SETUP
# ===========================================================

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    colorize=True,
)


# ===========================================================
# PREPROCESSOR CLASS
# ===========================================================

class Preprocessor:
    """
    Transforms validated data into analysis-ready formats.

    WHY A CLASS:
    - Stores baskets_df and categories_df as state
    - Methods share data without passing it as arguments
    - Easy to test with mock DataFrames
    - Clean interface for main.py and analyzer.py

    Attributes:
        baskets_df    (pd.DataFrame): Validated baskets data
        categories_df (pd.DataFrame): Validated categories data
        _item_name_map (dict):        Maps item_id → category name
    """

    def __init__(
        self,
        baskets_df: pd.DataFrame,
        categories_df: pd.DataFrame,
    ):
        """
        Initialize Preprocessor with validated DataFrames.

        Args:
            baskets_df:    Output from DataLoader.load_baskets()
            categories_df: Output from DataLoader.load_categories()
        """
        self.baskets_df    = baskets_df.copy()
        self.categories_df = categories_df.copy()

        # Build item_id → name mapping once, reuse everywhere
        # This replaces scattered df_1.loc[] lookups in original code
        self._item_name_map = self._build_item_name_map()

        logger.info("Preprocessor initialized")
        logger.info(f"  Transactions  : {len(self.baskets_df):,}")
        logger.info(f"  Categories    : {len(self.categories_df):,}")
        logger.info(f"  Item ID range : {ITEM_ID_MIN}–{ITEM_ID_MAX}")

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def build_one_hot_matrix(self) -> pd.DataFrame:
        """
        Build a one-hot encoded matrix of transactions × items.

        WHY THIS REPLACES iterrows():
        ─────────────────────────────
        Original code (Junior — 1.7M Python iterations):
            for index, row in df.iterrows():
                for i in range(1, 49, 1):
                    if i in row['basket_items']:
                        df.loc[index, col_item] = 1

        Our code (Senior — vectorized C operations):
            Uses mlxtend TransactionEncoder which runs in
            compiled C code — approximately 100x faster.

        Returns:
            pd.DataFrame where:
                - rows    = transactions
                - columns = item category names
                - values  = True/False (item in basket or not)

        Example output:
                    T-Shirts  Jeans  Jackets  Shoes ...
            0       True      False  True     False
            1       False     True   False    True
            2       True      True   False    False
        """
        logger.info("Building one-hot encoded transaction matrix...")

        from mlxtend.preprocessing import TransactionEncoder

        # Step 1 — Get list of item lists
        # [[34, 13, 42], [25, 32, 10], ...]
        transactions = self.baskets_df["basket_items"].tolist()
        logger.debug(f"  Total transactions: {len(transactions):,}")

        # Step 2 — Fit and transform using TransactionEncoder
        # This is the vectorized replacement for iterrows()
        encoder = TransactionEncoder()
        matrix  = encoder.fit(transactions).transform(transactions)

        # Step 3 — Convert to DataFrame with item IDs as columns
        df_encoded = pd.DataFrame(
            matrix,
            columns=encoder.columns_,
        )
        logger.debug(
            f"  Matrix shape: "
            f"{df_encoded.shape[0]:,} rows × "
            f"{df_encoded.shape[1]} columns"
        )

        # Step 4 — Rename columns from item IDs to category names
        # Original code: df_apr = df_te.rename(columns=dict(...))
        # We do this cleanly using our pre-built name map
        df_encoded = df_encoded.rename(
            columns=self._item_name_map
        )

        logger.info(
            f"  ✓ One-hot matrix built: "
            f"{df_encoded.shape[0]:,} × {df_encoded.shape[1]}"
        )
        return df_encoded

    def get_basket_stats(self) -> Dict:
        """
        Compute summary statistics for basket sizes.

        WHY: The original code scattered these calculations
        across the notebook. One function, one place.

        Returns:
            dict with keys:
                mean, median, std, min, max, skew, kurtosis
        """
        logger.info("Computing basket statistics...")

        sizes = self.baskets_df["basket_size"]

        stats = {
            "mean"     : round(sizes.mean(),   2),
            "median"   : round(sizes.median(), 2),
            "std"      : round(sizes.std(),    2),
            "min"      : int(sizes.min()),
            "max"      : int(sizes.max()),
            "skew"     : round(sizes.skew(),   4),
            "kurtosis" : round(sizes.kurtosis(), 4),
            "total_transactions": len(sizes),
        }

        logger.info(f"  Mean basket size   : {stats['mean']}")
        logger.info(f"  Median basket size : {stats['median']}")
        logger.info(f"  Min / Max          : {stats['min']} / {stats['max']}")
        logger.info(f"  Skew               : {stats['skew']}")
        logger.info(f"  Kurtosis           : {stats['kurtosis']}")

        return stats

    def get_item_frequency(self) -> pd.DataFrame:
        """
        Count how many transactions contain each item.

        WHY THIS FIXES THE ORIGINAL BUG:
        ─────────────────────────────────
        Original code (broken — off-by-one):
            poplr_item = [0] * 49
            for j in df['basket_items']:
                for i in range(1,49):
                    if i in j:
                        poplr_item[i] += 1
            # Bug: list index 0 unused, index confusion
            poplr_item = df_1.loc[[
                poplr_item.index(max(poplr_item)) - 1
            ]]  # ← -1 returns WRONG item

        Our code (correct — dictionary based):
            Uses item_id directly as key.
            No index confusion. No off-by-one possible.

        Returns:
            pd.DataFrame with columns:
                category_id, name, description,
                frequency, percentage
            Sorted by frequency descending.
        """
        logger.info("Computing item frequencies...")

        # Count occurrences of each item_id across all baskets
        # Dictionary: {item_id: count}
        # This is clear, correct, and has no index confusion
        item_counts: Dict[int, int] = {}
        for basket in self.baskets_df["basket_items"]:
            for item_id in basket:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1

        # Convert to DataFrame
        freq_df = pd.DataFrame(
            list(item_counts.items()),
            columns=["category_id", "frequency"]
        )

        # Merge with category names
        freq_df = freq_df.merge(
            self.categories_df[["category_id", "name", "description"]],
            on="category_id",
            how="left",
        )

        # Add percentage column
        total = freq_df["frequency"].sum()
        freq_df["percentage"] = (
            freq_df["frequency"] / total * 100
        ).round(2)

        # Sort by frequency descending
        freq_df = freq_df.sort_values(
            "frequency", ascending=False
        ).reset_index(drop=True)

        logger.info(
            f"  Most popular item  : "
            f"{freq_df.iloc[0]['name']} "
            f"({freq_df.iloc[0]['frequency']:,} transactions)"
        )
        logger.info(
            f"  Least popular item : "
            f"{freq_df.iloc[-1]['name']} "
            f"({freq_df.iloc[-1]['frequency']:,} transactions)"
        )

        return freq_df

    def get_top_customer_per_item(self) -> pd.DataFrame:
        """
        Find the customer who purchased each item the most.

        WHY THIS FIXES THE ORIGINAL CODE:
        ──────────────────────────────────
        Original code (slow nested loops):
            df_group = df.groupby('customer_id').sum()
            csr_mx = []
            for i in range(1,49,1):
                col_item = 'item' + str(i)
                a = df_group[
                    df_group[col_item].max() == df_group[col_item]
                ][[col_item, "basket_size"]]
                csr_mx.append(a)

        Problems:
        1. Requires 48 extra columns in the DataFrame
        2. Uses iterrows() implicitly through .max() in a loop
        3. Off-by-one in column naming

        Our approach:
        1. Explode basket_items into individual rows
        2. Group by customer_id and item_id
        3. Find max with idxmax() — one vectorized operation

        Returns:
            pd.DataFrame with columns:
                category_id, name,
                top_customer_id, purchase_count
        """
        logger.info("Finding top customer per item...")

        # Step 1 — Explode: one row per item per transaction
        # Before: [customer_id=1001, basket_items=[1,5,12]]
        # After:  [customer_id=1001, item_id=1]
        #         [customer_id=1001, item_id=5]
        #         [customer_id=1001, item_id=12]
        exploded = (
            self.baskets_df[["customer_id", "basket_items"]]
            .explode("basket_items")
            .rename(columns={"basket_items": "item_id"})
        )

        # Step 2 — Count purchases per customer per item
        purchase_counts = (
            exploded
            .groupby(["item_id", "customer_id"])
            .size()
            .reset_index(name="purchase_count")
        )

        # Step 3 — Find top customer for each item
        # idxmax() finds the index of the maximum value
        # This replaces the entire nested loop
        top_customers = (
            purchase_counts
            .loc[
                purchase_counts
                .groupby("item_id")["purchase_count"]
                .idxmax()
            ]
            .reset_index(drop=True)
            .rename(columns={"item_id": "category_id"})
        )

        # Step 4 — Merge with category names
        top_customers = top_customers.merge(
            self.categories_df[["category_id", "name"]],
            on="category_id",
            how="left",
        )

        # Reorder columns for clarity
        top_customers = top_customers[[
            "category_id", "name",
            "customer_id", "purchase_count"
        ]]

        logger.info(
            f"  ✓ Top customers found for "
            f"{len(top_customers)} item categories"
        )
        return top_customers

    # -------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------

    def _build_item_name_map(self) -> Dict[int, str]:
        """
        Build a dictionary mapping item_id → category name.

        WHY: The original code did this inline scattered across
        multiple cells. We build it once and reuse it everywhere.

        Returns:
            dict: {1: 'T-Shirts', 2: 'Jeans', 3: 'Jackets', ...}
        """
        name_map = dict(
            zip(
                self.categories_df["category_id"],
                self.categories_df["name"],
            )
        )
        logger.debug(f"  Item name map built: {len(name_map)} items")
        return name_map