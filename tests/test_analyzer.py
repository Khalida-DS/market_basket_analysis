"""
test_analyzer.py — Unit Tests for Analyzer
===========================================

HOW TO RUN:
    pytest tests/test_analyzer.py -v

WHAT WE TEST:
    - Analyzer initializes correctly
    - run() returns a DataFrame
    - run() returns expected columns
    - Zhang's metric is calculated correctly
    - Filtering removes weak rules
    - get_summary() returns correct structure

NOTE:
    We use a small synthetic one-hot matrix so tests
    run fast and don't depend on real data files.
"""

import pytest
import pandas as pd
import numpy as np
from src.analyzer import Analyzer


# ===========================================================
# FIXTURES
# ===========================================================

@pytest.fixture
def small_one_hot_df() -> pd.DataFrame:
    """
    Small one-hot encoded DataFrame for testing.

    We create 100 rows with controlled patterns so
    we know exactly what rules should be found.

    Pattern:
        T-Shirts + Jeans appear together frequently → strong rule
        Shoes + Socks appear together frequently    → strong rule
        Jackets appears randomly                    → weak/no rule
    """
    np.random.seed(42)
    n = 200

    # Create strongly correlated pairs
    t_shirts = np.random.choice([True, False], size=n, p=[0.6, 0.4])
    jeans    = np.where(t_shirts, 
                        np.random.choice([True, False], p=[0.8, 0.2]),
                        np.random.choice([True, False], p=[0.2, 0.8]))
    shoes    = np.random.choice([True, False], size=n, p=[0.5, 0.5])
    socks    = np.where(shoes,
                        np.random.choice([True, False], p=[0.8, 0.2]),
                        np.random.choice([True, False], p=[0.2, 0.8]))
    jackets  = np.random.choice([True, False], size=n, p=[0.3, 0.7])

    return pd.DataFrame({
        "T-Shirts" : t_shirts,
        "Jeans"    : jeans,
        "Shoes"    : shoes,
        "Socks"    : socks,
        "Jackets"  : jackets,
    })


@pytest.fixture
def analyzer(small_one_hot_df) -> Analyzer:
    """Creates an Analyzer with small test data."""
    return Analyzer(small_one_hot_df)


@pytest.fixture
def analyzer_with_rules(analyzer) -> Analyzer:
    """Creates an Analyzer that has already run."""
    analyzer.run()
    return analyzer


# ===========================================================
# TESTS: Initialization
# ===========================================================

class TestAnalyzerInit:
    """Tests for Analyzer.__init__"""

    def test_init_stores_dataframe(self, small_one_hot_df):
        """Analyzer should store the one-hot DataFrame."""
        a = Analyzer(small_one_hot_df)
        assert len(a.one_hot_df) == len(small_one_hot_df)

    def test_init_rules_df_is_none(self, small_one_hot_df):
        """rules_df should be None before run() is called."""
        a = Analyzer(small_one_hot_df)
        assert a.rules_df is None

    def test_init_itemsets_df_is_none(self, small_one_hot_df):
        """itemsets_df should be None before run() is called."""
        a = Analyzer(small_one_hot_df)
        assert a.itemsets_df is None


# ===========================================================
# TESTS: run()
# ===========================================================

class TestRun:
    """Tests for Analyzer.run()"""

    def test_returns_dataframe(self, analyzer):
        """run() should return a pandas DataFrame."""
        result = analyzer.run()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, analyzer):
        """Result should have all expected columns."""
        result = analyzer.run()
        assert "antecedents"   in result.columns
        assert "consequents"   in result.columns
        assert "support"       in result.columns
        assert "confidence"    in result.columns
        assert "lift"          in result.columns
        assert "zhangs_metric" in result.columns

    def test_rules_df_is_set_after_run(self, analyzer):
        """rules_df attribute should be set after run()."""
        analyzer.run()
        assert analyzer.rules_df is not None

    def test_itemsets_df_is_set_after_run(self, analyzer):
        """itemsets_df attribute should be set after run()."""
        analyzer.run()
        assert analyzer.itemsets_df is not None

    def test_sorted_by_zhang_descending(self, analyzer):
        """Rules should be sorted by zhang descending."""
        result = analyzer.run()
        if len(result) > 1:
            zhang_values = result["zhangs_metric"].tolist()
            assert zhang_values == sorted(zhang_values, reverse=True)

    def test_all_confidence_above_threshold(self, analyzer):
        """All rules should meet minimum confidence threshold."""
        from src.config import APRIORI_MIN_CONFIDENCE
        result = analyzer.run()
        if len(result) > 0:
            assert (result["confidence"] >= APRIORI_MIN_CONFIDENCE).all()

    def test_all_lift_above_threshold(self, analyzer):
        """All rules should have lift > 1.0."""
        from src.config import APRIORI_MIN_LIFT
        result = analyzer.run()
        if len(result) > 0:
            assert (result["lift"] > APRIORI_MIN_LIFT).all()

    def test_all_zhang_above_threshold(self, analyzer):
        """All rules should have zhang > 0.0."""
        from src.config import APRIORI_MIN_ZHANG
        result = analyzer.run()
        if len(result) > 0:
            assert (result["zhangs_metric"] > APRIORI_MIN_ZHANG).all()


# ===========================================================
# TESTS: _add_zhangs_metric()
# ===========================================================

# ===========================================================
# TESTS: _add_zhangs_metric()
# ===========================================================

class TestZhangsMetric:
    """Tests for Zhang's metric calculation."""

    def test_zhang_range_is_valid(self, analyzer):
        """
        Zhang's metric should be between -1 and +1.
        We use a small tolerance for floating point edge cases.
        """
        result = analyzer.run()
        if len(result) > 0:
            assert (result["zhangs_metric"] >= -1.01).all()
            assert (result["zhangs_metric"] <= 1.01).all()

    def test_zhang_positive_for_correlated_items(
        self, small_one_hot_df
    ):
        """
        T-Shirts and Jeans are correlated in our fixture.
        Their Zhang's metric should be positive.
        """
        analyzer = Analyzer(small_one_hot_df)
        result   = analyzer.run()

        if len(result) > 0:
            tshirt_rules = result[
                result["antecedents"].apply(
                    lambda x: "T-Shirts" in x
                ) &
                result["consequents"].apply(
                    lambda x: "Jeans" in x
                )
            ]
            if len(tshirt_rules) > 0:
                assert (tshirt_rules["zhangs_metric"] > 0).all()

    def test_zhang_calculated_manually(self, small_one_hot_df):
        """
        Verify Zhang's metric formula against manual calculation.
        """
        analyzer = Analyzer(small_one_hot_df)
        result   = analyzer.run()

        if len(result) > 0:
            row  = result.iloc[0]
            p_ab = row["support"]
            p_a  = row["antecedent support"]
            p_b  = row["consequent support"]

            numerator   = p_ab - (p_a * p_b)
            denominator = max(
                p_ab * (1 - p_a),
                p_a  * (p_b - p_ab)
            )
            expected_zhang = (
                0 if denominator == 0
                else numerator / denominator
            )

            assert abs(
                row["zhangs_metric"] - expected_zhang
            ) < 0.0001

# ===========================================================
# TESTS: get_summary()
# ===========================================================

class TestGetSummary:
    """Tests for Analyzer.get_summary()"""

    def test_returns_dict(self, analyzer_with_rules):
        """get_summary() should return a dictionary."""
        result = analyzer_with_rules.get_summary()
        assert isinstance(result, dict)

    def test_has_required_keys(self, analyzer_with_rules):
        """Summary should contain all expected keys."""
        result = analyzer_with_rules.get_summary()
        if result:
            assert "total_rules"          in result
            assert "avg_confidence"       in result
            assert "avg_lift"             in result
            assert "avg_zhang"            in result
            assert "top_rule_antecedents" in result
            assert "top_rule_consequents" in result

    def test_total_rules_matches_rules_df(
        self, analyzer_with_rules
    ):
        """total_rules should match length of rules_df."""
        result = analyzer_with_rules.get_summary()
        if result:
            assert result["total_rules"] == len(
                analyzer_with_rules.rules_df
            )

    def test_returns_empty_dict_before_run(
        self, small_one_hot_df
    ):
        """get_summary() before run() should return empty dict."""
        a      = Analyzer(small_one_hot_df)
        result = a.get_summary()
        assert result == {}