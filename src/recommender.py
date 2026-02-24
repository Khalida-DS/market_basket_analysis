"""
recommender.py — Product Recommendation Engine
===============================================

WHY THIS FILE EXISTS:
---------------------
The original notebook had no recommendation engine.
It found association rules but never used them to actually
recommend products to customers.

This module takes the rules from Analyzer and turns them
into actionable recommendations:
    "Customer has T-Shirts in basket"
    → "Recommend Jeans (confidence: 0.71, lift: 1.8)"

HOW TO USE:
-----------
    from src.recommender import Recommender

    recommender = Recommender(rules_df)
    recommendations = recommender.recommend(["T-Shirts", "Jeans"])
"""

import pandas as pd
from loguru import logger
from typing import List, Dict

from src.config import (
    RECOMMENDER_TOP_N,
    RECOMMENDER_MIN_CONFIDENCE,
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
# RECOMMENDER CLASS
# ===========================================================

class Recommender:
    """
    Generates product recommendations from association rules.

    HOW IT WORKS:
    -------------
    1. Takes a customer's current basket as input
    2. Finds all rules where antecedents are a SUBSET
       of the current basket
    3. Returns consequents of those rules as recommendations
    4. Ranks by confidence (most reliable first)
    5. Filters out items already in the basket

    Example:
        basket = ["T-Shirts", "Jeans"]

        Matching rules:
            {T-Shirts} → {Jackets}  confidence=0.71
            {Jeans}    → {Shoes}    confidence=0.65
            {T-Shirts, Jeans} → {Belts} confidence=0.60

        Recommendations: [Jackets, Shoes, Belts]

    Attributes:
        rules_df (pd.DataFrame): Output from Analyzer.run()
    """

    def __init__(self, rules_df: pd.DataFrame):
        """
        Initialize Recommender with association rules.

        Args:
            rules_df: Output from Analyzer.run()
                      Must have columns:
                      antecedents, consequents, confidence, lift
        """
        self.rules_df = rules_df.copy()

        logger.info("Recommender initialized")
        logger.info(f"  Rules loaded: {len(self.rules_df):,}")

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def recommend(
        self,
        basket: List[str],
        top_n: int = RECOMMENDER_TOP_N,
        min_confidence: float = RECOMMENDER_MIN_CONFIDENCE,
    ) -> pd.DataFrame:
        """
        Generate product recommendations for a given basket.

        Args:
            basket:         List of item names currently in basket
            top_n:          Max number of recommendations to return
            min_confidence: Minimum confidence threshold

        Returns:
            pd.DataFrame with columns:
                item        : recommended product name
                confidence  : reliability of the recommendation
                lift        : how much better than random chance
                zhangs_metric: association strength
                rule        : the rule that triggered recommendation

        Example:
            recommender.recommend(["T-Shirts", "Jeans"])

            item      confidence  lift   zhangs_metric
            Jackets   0.71        1.8    0.31
            Shoes     0.65        1.6    0.28
            Belts     0.60        1.5    0.22
        """
        logger.info(f"Generating recommendations for: {basket}")

        if not basket:
            logger.warning("  Empty basket — no recommendations")
            return pd.DataFrame()

        if len(self.rules_df) == 0:
            logger.warning("  No rules available")
            return pd.DataFrame()

        basket_set = set(basket)

        # Find all matching rules
        # A rule matches if ALL its antecedents are in the basket
        matching_rules = self.rules_df[
            self.rules_df["antecedents"].apply(
                lambda antecedents: antecedents.issubset(basket_set)
            ) &
            (self.rules_df["confidence"] >= min_confidence)
        ].copy()

        logger.info(
            f"  Matching rules found: {len(matching_rules)}"
        )

        if len(matching_rules) == 0:
            logger.info("  No matching rules for this basket")
            return pd.DataFrame()

        # Extract recommendations from consequents
        recommendations = []

        for _, rule in matching_rules.iterrows():
            for item in rule["consequents"]:
                # Skip items already in basket
                if item not in basket_set:
                    recommendations.append({
                        "item"         : item,
                        "confidence"   : round(rule["confidence"], 4),
                        "lift"         : round(rule["lift"], 4),
                        "zhangs_metric": round(rule["zhangs_metric"], 4),
                        "rule"         : (
                            f"{set(rule['antecedents'])} "
                            f"→ {set(rule['consequents'])}"
                        ),
                    })

        if not recommendations:
            logger.info(
                "  All consequents already in basket"
            )
            return pd.DataFrame()

        # Convert to DataFrame
        rec_df = pd.DataFrame(recommendations)

        # Remove duplicates — keep highest confidence per item
        rec_df = (
            rec_df
            .sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["item"])
            .head(top_n)
            .reset_index(drop=True)
        )

        logger.info(
            f"  Recommendations returned: {len(rec_df)}"
        )
        for _, row in rec_df.iterrows():
            logger.info(
                f"    → {row['item']:<20} "
                f"confidence={row['confidence']:.2f}  "
                f"lift={row['lift']:.2f}"
            )

        return rec_df

    def recommend_for_customer(
        self,
        customer_id: int,
        baskets_df: pd.DataFrame,
        top_n: int = RECOMMENDER_TOP_N,
    ) -> pd.DataFrame:
        """
        Generate recommendations based on a customer's
        most recent basket.

        Args:
            customer_id: Customer ID to generate recs for
            baskets_df:  DataFrame from DataLoader.load_baskets()
            top_n:       Max recommendations to return

        Returns:
            pd.DataFrame with recommendations
            Empty DataFrame if customer not found
        """
        logger.info(
            f"Generating recommendations for "
            f"customer {customer_id}..."
        )

        # Find customer's most recent basket
        customer_baskets = baskets_df[
            baskets_df["customer_id"] == customer_id
        ]

        if customer_baskets.empty:
            logger.warning(
                f"  Customer {customer_id} not found"
            )
            return pd.DataFrame()

        # Get the most recent basket (last row)
        latest_basket = customer_baskets.iloc[-1]["basket_items"]

        logger.info(
            f"  Latest basket size: {len(latest_basket)} items"
        )

        return self.recommend(latest_basket, top_n=top_n)

    def get_popular_recommendations(
        self,
        top_n: int = RECOMMENDER_TOP_N,
    ) -> pd.DataFrame:
        """
        Return the most commonly recommended items across
        all rules — useful for cold start (new customers
        with no purchase history).

        Args:
            top_n: Number of popular items to return

        Returns:
            pd.DataFrame with columns:
                item, avg_confidence, avg_lift, rule_count
        """
        logger.info("Getting popular recommendations...")

        if len(self.rules_df) == 0:
            return pd.DataFrame()

        # Explode consequents into individual items
        all_consequents = []
        for _, rule in self.rules_df.iterrows():
            for item in rule["consequents"]:
                all_consequents.append({
                    "item"      : item,
                    "confidence": rule["confidence"],
                    "lift"      : rule["lift"],
                })

        if not all_consequents:
            return pd.DataFrame()

        cons_df = pd.DataFrame(all_consequents)

        # Aggregate by item
        popular = (
            cons_df
            .groupby("item")
            .agg(
                avg_confidence=("confidence", "mean"),
                avg_lift      =("lift",       "mean"),
                rule_count    =("confidence", "count"),
            )
            .reset_index()
            .sort_values("rule_count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        popular["avg_confidence"] = popular["avg_confidence"].round(4)
        popular["avg_lift"]       = popular["avg_lift"].round(4)

        logger.info(
            f"  Top {len(popular)} popular items found"
        )

        return popular