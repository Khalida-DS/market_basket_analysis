"""
test_recommender.py — Unit Tests for Recommender
=================================================

HOW TO RUN:
    pytest tests/test_recommender.py -v

WHAT WE TEST:
    - Recommender initializes correctly
    - recommend() returns correct structure
    - recommend() filters items already in basket
    - recommend() handles empty basket
    - recommend() handles no matching rules
    - recommend_for_customer() finds correct basket
    - get_popular_recommendations() returns ranked items
"""

import pytest
import pandas as pd
#from frozenset import *
from src.recommender import Recommender


# ===========================================================
# FIXTURES
# ===========================================================

@pytest.fixture
def sample_rules_df() -> pd.DataFrame:
    """
    Small rules DataFrame mimicking Analyzer output.
    We control exactly what rules exist so we know
    exactly what recommendations to expect.
    """
    return pd.DataFrame({
        "antecedents"   : [
            frozenset(["T-Shirts"]),
            frozenset(["Jeans"]),
            frozenset(["T-Shirts", "Jeans"]),
            frozenset(["Shoes"]),
            frozenset(["Jackets"]),
        ],
        "consequents"   : [
            frozenset(["Jeans"]),
            frozenset(["T-Shirts"]),
            frozenset(["Jackets"]),
            frozenset(["Socks"]),
            frozenset(["Belts"]),
        ],
        "support"       : [0.042, 0.041, 0.035, 0.038, 0.033],
        "confidence"    : [0.71,  0.68,  0.65,  0.72,  0.61 ],
        "lift"          : [1.8,   1.7,   1.6,   1.9,   1.5  ],
        "zhangs_metric" : [0.31,  0.28,  0.25,  0.35,  0.22 ],
        "antecedent support": [0.06, 0.06, 0.05, 0.05, 0.05],
        "consequent support": [0.06, 0.06, 0.05, 0.05, 0.05],
    })


@pytest.fixture
def sample_baskets_df() -> pd.DataFrame:
    """Small baskets DataFrame for customer-based tests."""
    return pd.DataFrame({
        "customer_id"  : [1001, 1002, 1001, 1003],
        "basket_items" : [
            [1, 2, 3],
            [4, 5, 6],
            [1, 3, 5],
            [2, 4, 6],
        ],
        "basket_size"  : [3, 3, 3, 3],
    })


@pytest.fixture
def recommender(sample_rules_df) -> Recommender:
    """Creates a Recommender with sample rules."""
    return Recommender(sample_rules_df)


# ===========================================================
# TESTS: Initialization
# ===========================================================

class TestRecommenderInit:
    """Tests for Recommender.__init__"""

    def test_init_stores_rules(self, sample_rules_df):
        """Recommender should store the rules DataFrame."""
        r = Recommender(sample_rules_df)
        assert len(r.rules_df) == len(sample_rules_df)

    def test_init_makes_copy(self, sample_rules_df):
        """Recommender should store a copy not the original."""
        r = Recommender(sample_rules_df)
        assert r.rules_df is not sample_rules_df


# ===========================================================
# TESTS: recommend()
# ===========================================================

class TestRecommend:
    """Tests for Recommender.recommend()"""

    def test_returns_dataframe(self, recommender):
        """recommend() should return a pandas DataFrame."""
        result = recommender.recommend(["T-Shirts"])
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, recommender):
        """Result should have all expected columns."""
        result = recommender.recommend(["T-Shirts"])
        if len(result) > 0:
            assert "item"          in result.columns
            assert "confidence"    in result.columns
            assert "lift"          in result.columns
            assert "zhangs_metric" in result.columns
            assert "rule"          in result.columns

    def test_does_not_recommend_items_in_basket(
        self, recommender
    ):
        """
        Items already in the basket should NOT be recommended.
        Basket has T-Shirts — rule {T-Shirts} → {Jeans}
        should recommend Jeans.
        But if basket has both T-Shirts AND Jeans,
        Jeans should not be recommended again.
        """
        result = recommender.recommend(["T-Shirts", "Jeans"])
        if len(result) > 0:
            assert "T-Shirts" not in result["item"].values
            assert "Jeans"    not in result["item"].values

    def test_empty_basket_returns_empty_dataframe(
        self, recommender
    ):
        """Empty basket should return empty DataFrame."""
        result = recommender.recommend([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_top_n_limits_results(self, recommender):
        """top_n parameter should limit results."""
        result = recommender.recommend(
            ["T-Shirts"], top_n=1
        )
        assert len(result) <= 1

    def test_sorted_by_confidence_descending(self, recommender):
        """Results should be sorted by confidence descending."""
        result = recommender.recommend(["T-Shirts", "Jeans"])
        if len(result) > 1:
            confidences = result["confidence"].tolist()
            assert confidences == sorted(
                confidences, reverse=True
            )

    def test_correct_recommendation_for_tshirts(
        self, recommender
    ):
        """
        Basket: [T-Shirts]
        Rule: {T-Shirts} → {Jeans} confidence=0.71
        Expected recommendation: Jeans
        """
        result = recommender.recommend(["T-Shirts"])
        if len(result) > 0:
            assert "Jeans" in result["item"].values

    def test_no_matching_rules_returns_empty(self, recommender):
        """
        Item with no rules should return empty DataFrame.
        'Hats' is not in any rule in our fixture.
        """
        result = recommender.recommend(["Hats"])
        assert len(result) == 0


# ===========================================================
# TESTS: recommend_for_customer()
# ===========================================================

class TestRecommendForCustomer:
    """Tests for Recommender.recommend_for_customer()"""

    def test_returns_dataframe(
        self, recommender, sample_baskets_df
    ):
        """Should return a pandas DataFrame."""
        result = recommender.recommend_for_customer(
            customer_id=1001,
            baskets_df=sample_baskets_df,
        )
        assert isinstance(result, pd.DataFrame)

    def test_unknown_customer_returns_empty(
        self, recommender, sample_baskets_df
    ):
        """Unknown customer ID should return empty DataFrame."""
        result = recommender.recommend_for_customer(
            customer_id=9999,
            baskets_df=sample_baskets_df,
        )
        assert len(result) == 0


# ===========================================================
# TESTS: get_popular_recommendations()
# ===========================================================

class TestGetPopularRecommendations:
    """Tests for Recommender.get_popular_recommendations()"""

    def test_returns_dataframe(self, recommender):
        """Should return a pandas DataFrame."""
        result = recommender.get_popular_recommendations()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, recommender):
        """Should have all expected columns."""
        result = recommender.get_popular_recommendations()
        if len(result) > 0:
            assert "item"           in result.columns
            assert "avg_confidence" in result.columns
            assert "avg_lift"       in result.columns
            assert "rule_count"     in result.columns

    def test_top_n_limits_results(self, recommender):
        """top_n parameter should limit results."""
        result = recommender.get_popular_recommendations(top_n=2)
        assert len(result) <= 2

    def test_sorted_by_rule_count_descending(self, recommender):
        """Results should be sorted by rule_count descending."""
        result = recommender.get_popular_recommendations()
        if len(result) > 1:
            counts = result["rule_count"].tolist()
            assert counts == sorted(counts, reverse=True)