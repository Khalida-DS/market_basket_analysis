"""
analyzer.py — Association Rule Mining
======================================

WHY THIS FILE EXISTS:
---------------------
The original notebook ran Apriori but did minimal filtering.
This module runs the full pipeline:
  1. Run Apriori to find frequent itemsets
  2. Generate association rules
  3. Calculate Zhang's metric
  4. Filter by confidence, lift, and zhang
  5. Return clean, ranked rules DataFrame

WHAT THIS FILE FIXES FROM THE ORIGINAL:
----------------------------------------
1. No Zhang's metric           → added as a proper filter
2. Only top 5 rules shown      → full ranked DataFrame returned
3. No confidence filtering     → proper threshold from config
4. Magic numbers everywhere    → all from config.py

HOW TO USE:
-----------
    from src.analyzer import Analyzer

    analyzer = Analyzer(one_hot_df)
    rules_df = analyzer.run()
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional

from mlxtend.frequent_patterns import apriori, association_rules

from src.config import (
    APRIORI_MIN_SUPPORT,
    APRIORI_MIN_CONFIDENCE,
    APRIORI_MIN_LIFT,
    APRIORI_MIN_ZHANG,
    APRIORI_METRIC,
    APRIORI_MIN_THRESHOLD,
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
# ANALYZER CLASS
# ===========================================================

class Analyzer:
    """
    Runs the full association rule mining pipeline.

    Steps:
        1. Run Apriori → frequent itemsets
        2. Generate rules → raw association rules
        3. Add Zhang's metric → better filtering
        4. Filter rules → only strong rules remain
        5. Sort by zhang → best rules first

    Attributes:
        one_hot_df  (pd.DataFrame): Output from Preprocessor
        rules_df    (pd.DataFrame): Final filtered rules
        itemsets_df (pd.DataFrame): Frequent itemsets found

    Usage:
        analyzer = Analyzer(one_hot_df)
        rules_df  = analyzer.run()
    """

    def __init__(self, one_hot_df: pd.DataFrame):
        """
        Initialize Analyzer with one-hot encoded matrix.

        Args:
            one_hot_df: Boolean DataFrame from
                        Preprocessor.build_one_hot_matrix()
        """
        self.one_hot_df  = one_hot_df
        self.rules_df    = None
        self.itemsets_df = None

        logger.info("Analyzer initialized")
        logger.info(
            f"  Matrix shape: "
            f"{one_hot_df.shape[0]:,} transactions × "
            f"{one_hot_df.shape[1]} items"
        )

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full association rule mining pipeline.

        Returns:
            pd.DataFrame with columns:
                antecedents   : frozenset of items in the IF part
                consequents   : frozenset of items in the THEN part
                support       : % of transactions containing both
                confidence    : P(consequents | antecedents)
                lift          : how much better than random chance
                zhangs_metric : zhang's metric (-1 to +1)

        Example:
            rules_df = analyzer.run()
            # antecedents={T-Shirts}, consequents={Jeans}
            # support=0.042, confidence=0.71, lift=1.8, zhang=0.31
        """
        logger.info("=" * 55)
        logger.info("Starting association rule mining pipeline")
        logger.info("=" * 55)

        # Step 1 — Find frequent itemsets
        self.itemsets_df = self._find_frequent_itemsets()

        # Step 2 — Generate association rules
        raw_rules = self._generate_rules()

        # Step 3 — Add Zhang's metric
        raw_rules = self._add_zhangs_metric(raw_rules)

        # Step 4 — Filter rules
        filtered_rules = self._filter_rules(raw_rules)

        # Step 5 — Sort by Zhang's metric descending
        self.rules_df = filtered_rules.sort_values(
            "zhangs_metric", ascending=False
        ).reset_index(drop=True)

        logger.info("=" * 55)
        logger.info(
            f"Pipeline complete: "
            f"{len(self.rules_df)} rules found"
        )
        logger.info("=" * 55)

        return self.rules_df

    def get_summary(self) -> dict:
        """
        Return summary statistics about the rules found.

        Returns:
            dict with keys:
                total_rules, avg_confidence, avg_lift,
                avg_zhang, avg_support,
                top_rule_antecedents, top_rule_consequents
        """
        if self.rules_df is None or len(self.rules_df) == 0:
            logger.warning("No rules found. Run analyzer.run() first.")
            return {}

        top_rule = self.rules_df.iloc[0]

        summary = {
            "total_rules"          : len(self.rules_df),
            "avg_confidence"       : round(
                self.rules_df["confidence"].mean(), 4
            ),
            "avg_lift"             : round(
                self.rules_df["lift"].mean(), 4
            ),
            "avg_zhang"            : round(
                self.rules_df["zhangs_metric"].mean(), 4
            ),
            "avg_support"          : round(
                self.rules_df["support"].mean(), 4
            ),
            "top_rule_antecedents" : list(top_rule["antecedents"]),
            "top_rule_consequents" : list(top_rule["consequents"]),
            "top_rule_confidence"  : round(top_rule["confidence"], 4),
            "top_rule_lift"        : round(top_rule["lift"], 4),
            "top_rule_zhang"       : round(top_rule["zhangs_metric"], 4),
        }

        logger.info("Rule Mining Summary:")
        logger.info(f"  Total rules      : {summary['total_rules']}")
        logger.info(f"  Avg confidence   : {summary['avg_confidence']}")
        logger.info(f"  Avg lift         : {summary['avg_lift']}")
        logger.info(f"  Avg zhang        : {summary['avg_zhang']}")
        logger.info(
            f"  Top rule         : "
            f"{summary['top_rule_antecedents']} → "
            f"{summary['top_rule_consequents']}"
        )

        return summary

    # -------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------

    def _find_frequent_itemsets(self) -> pd.DataFrame:
        """
        Run Apriori algorithm to find frequent itemsets.

        WHY min_support matters:
            Too low  (0.001) → millions of itemsets → slow
            Too high (0.10)  → almost nothing found
            We use 0.01      → items appearing in 1%+ of baskets

        Returns:
            pd.DataFrame with columns:
                support    : % of transactions containing itemset
                itemsets   : frozenset of items
        """
        logger.info(
            f"Running Apriori "
            f"(min_support={APRIORI_MIN_SUPPORT})..."
        )

        itemsets = apriori(
            self.one_hot_df,
            min_support=APRIORI_MIN_SUPPORT,
            use_colnames=True,   # ← use names not column indices
            verbose=0,
        )

        logger.info(
            f"  Frequent itemsets found: {len(itemsets):,}"
        )
        logger.info(
            f"  Support range: "
            f"{itemsets['support'].min():.4f} – "
            f"{itemsets['support'].max():.4f}"
        )

        return itemsets

    def _generate_rules(self) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        WHY metric="lift" with threshold=0.1:
            We use a LOW threshold here intentionally.
            We want ALL possible rules first.
            Then we filter properly in _filter_rules().
            This gives us full control over filtering.

        Returns:
            pd.DataFrame with association rules
        """
        logger.info("Generating association rules...")

        rules = association_rules(
            self.itemsets_df,
            metric=APRIORI_METRIC,
            min_threshold=APRIORI_MIN_THRESHOLD,
            num_itemsets=len(self.itemsets_df),
        )

        logger.info(f"  Raw rules generated: {len(rules):,}")

        return rules

    def _add_zhangs_metric(
        self, rules: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate and add Zhang's metric to the rules DataFrame.

        WHY Zhang's metric:
            Lift can be misleading for very common items.
            Zhang's metric measures GENUINE association strength.

        Formula:
            zhang = (P(A∩B) - P(A)×P(B)) /
                    max(
                        P(A∩B) × (1 - P(A)),
                        P(A) × (P(B) - P(A∩B))
                    )

        Interpretation:
            +1.0 = perfect positive association
             0.0 = no association
            -1.0 = perfect negative association

        Args:
            rules: DataFrame with support, antecedent support,
                   consequent support columns

        Returns:
            rules DataFrame with new 'zhangs_metric' column
        """
        logger.info("Calculating Zhang's metric...")

        # Rename for shorter variable names
        p_ab = rules["support"]
        p_a  = rules["antecedent support"]
        p_b  = rules["consequent support"]

        # Numerator: P(A∩B) - P(A)×P(B)
        numerator = p_ab - (p_a * p_b)

        # Denominator: max(P(A∩B)×(1-P(A)), P(A)×(P(B)-P(A∩B)))
        denominator = np.maximum(
            p_ab * (1 - p_a),
            p_a  * (p_b - p_ab)
        )

        # Avoid division by zero
        rules["zhangs_metric"] = np.where(
            denominator == 0,
            0,
            numerator / denominator
        )

        logger.info(
            f"  Zhang's metric range: "
            f"{rules['zhangs_metric'].min():.4f} – "
            f"{rules['zhangs_metric'].max():.4f}"
        )

        return rules

    def _filter_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters to keep only strong rules.

        Filters applied:
            confidence >= 0.60  → rule is reliable
            lift       >  1.00  → items appear together more than chance
            zhang      >  0.00  → genuine positive association

        WHY these thresholds (from config.py):
            confidence < 0.60 = rule is wrong more than 40% of time
            lift <= 1.0       = no real relationship
            zhang <= 0.0      = A does not increase P(B)

        Args:
            rules: DataFrame with all generated rules

        Returns:
            Filtered DataFrame with only strong rules
        """
        logger.info("Filtering rules...")
        logger.info(f"  Rules before filtering : {len(rules):,}")

        filtered = rules[
            (rules["confidence"]    >= APRIORI_MIN_CONFIDENCE) &
            (rules["lift"]          >  APRIORI_MIN_LIFT)       &
            (rules["zhangs_metric"] >  APRIORI_MIN_ZHANG)
        ].copy()

        logger.info(
            f"  Rules after filtering  : {len(filtered):,}"
        )
        logger.info(
            f"  Rules removed          : "
            f"{len(rules) - len(filtered):,}"
        )

        if len(filtered) == 0:
            logger.warning(
                "  No rules passed filtering. "
                "Consider lowering thresholds in config.py"
            )

        return filtered