"""
test_preprocessor.py — Unit Tests for Preprocessor
====================================================

HOW TO RUN:
    pytest tests/test_preprocessor.py -v

WHAT WE TEST:
    - Preprocessor initializes correctly
    - build_one_hot_matrix() returns correct structure
    - get_basket_stats() returns correct values
    - get_item_frequency() returns correct counts
    - get_item_frequency() fixes the off-by-one bug
    - get_top_customer_per_item() returns correct structure
"""

import pytest
import pandas as pd
from src.preprocessor import Preprocessor


# ===========================================================
# FIXTURES
# ===========================================================

@pytest.fixture
def sample_baskets_df() -> pd.DataFrame:
    """
    Small baskets DataFrame mimicking DataLoader output.
    Matches real data format: basket_items is list of ints.
    """
    return pd.DataFrame({
        "customer_id"  : [1001, 1002, 1003, 1001],
        "basket"       : [
            "1,2,3",
            "2,3,4",
            "1,3,5",
            "1,2,5",
        ],
        "basket_items" : [
            [1, 2, 3],
            [2, 3, 4],
            [1, 3, 5],
            [1, 2, 5],
        ],
        "basket_size"  : [3, 3, 3, 3],
    })


@pytest.fixture
def sample_categories_df() -> pd.DataFrame:
    """
    Small categories DataFrame mimicking DataLoader output.
    """
    return pd.DataFrame({
        "category_id" : [1, 2, 3, 4, 5],
        "name"        : [
            "T-Shirts",
            "Jeans",
            "Jackets",
            "Shoes",
            "Socks",
        ],
        "description" : [
            "Casual shirts",
            "Denim trousers",
            "Outerwear",
            "Footwear",
            "Hosiery",
        ],
    })


@pytest.fixture
def preprocessor(
    sample_baskets_df, sample_categories_df
) -> Preprocessor:
    """Creates a Preprocessor with sample data."""
    return Preprocessor(
        baskets_df=sample_baskets_df,
        categories_df=sample_categories_df,
    )


# ===========================================================
# TESTS: Initialization
# ===========================================================

class TestPreprocessorInit:
    """Tests for Preprocessor.__init__"""

    def test_init_stores_dataframes(
        self, sample_baskets_df, sample_categories_df
    ):
        """Preprocessor should store both DataFrames."""
        p = Preprocessor(sample_baskets_df, sample_categories_df)
        assert len(p.baskets_df)    == len(sample_baskets_df)
        assert len(p.categories_df) == len(sample_categories_df)

    def test_init_builds_item_name_map(
        self, sample_baskets_df, sample_categories_df
    ):
        """Item name map should be built on initialization."""
        p = Preprocessor(sample_baskets_df, sample_categories_df)
        assert isinstance(p._item_name_map, dict)
        assert len(p._item_name_map) == 5

    def test_item_name_map_correct_values(
        self, sample_baskets_df, sample_categories_df
    ):
        """Item name map should map IDs to correct names."""
        p = Preprocessor(sample_baskets_df, sample_categories_df)
        assert p._item_name_map[1] == "T-Shirts"
        assert p._item_name_map[2] == "Jeans"
        assert p._item_name_map[3] == "Jackets"


# ===========================================================
# TESTS: build_one_hot_matrix()
# ===========================================================

class TestBuildOneHotMatrix:
    """Tests for Preprocessor.build_one_hot_matrix()"""

    def test_returns_dataframe(self, preprocessor):
        """Should return a pandas DataFrame."""
        result = preprocessor.build_one_hot_matrix()
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_rows(self, preprocessor):
        """Should have same number of rows as transactions."""
        result = preprocessor.build_one_hot_matrix()
        assert len(result) == 4

    def test_columns_are_category_names(self, preprocessor):
        """Columns should be category names not item IDs."""
        result = preprocessor.build_one_hot_matrix()
        assert "T-Shirts" in result.columns
        assert "Jeans"    in result.columns
        assert "Jackets"  in result.columns

    def test_values_are_boolean(self, preprocessor):
        """All values should be boolean True/False."""
        result = preprocessor.build_one_hot_matrix()
        for col in result.columns:
            assert result[col].dtype == bool

    def test_correct_encoding(self, preprocessor):
        """
        First basket [1,2,3] = T-Shirts, Jeans, Jackets.
        T-Shirts should be True, Shoes should be False.
        """
        result = preprocessor.build_one_hot_matrix()
        assert result["T-Shirts"].iloc[0] == True
        assert result["Jeans"].iloc[0]    == True
        assert result["Jackets"].iloc[0]  == True
        assert result["Shoes"].iloc[0]    == False


# ===========================================================
# TESTS: get_basket_stats()
# ===========================================================

class TestGetBasketStats:
    """Tests for Preprocessor.get_basket_stats()"""

    def test_returns_dict(self, preprocessor):
        """Should return a dictionary."""
        result = preprocessor.get_basket_stats()
        assert isinstance(result, dict)

    def test_has_required_keys(self, preprocessor):
        """Should contain all expected statistical keys."""
        result = preprocessor.get_basket_stats()
        assert "mean"              in result
        assert "median"            in result
        assert "std"               in result
        assert "min"               in result
        assert "max"               in result
        assert "skew"              in result
        assert "kurtosis"          in result
        assert "total_transactions" in result

    def test_correct_total_transactions(self, preprocessor):
        """Should count 4 total transactions."""
        result = preprocessor.get_basket_stats()
        assert result["total_transactions"] == 4

    def test_correct_min_max(self, preprocessor):
        """All baskets have size 3 so min and max should be 3."""
        result = preprocessor.get_basket_stats()
        assert result["min"] == 3
        assert result["max"] == 3

    def test_correct_mean(self, preprocessor):
        """All baskets have size 3 so mean should be 3.0."""
        result = preprocessor.get_basket_stats()
        assert result["mean"] == 3.0


# ===========================================================
# TESTS: get_item_frequency()
# ===========================================================

class TestGetItemFrequency:
    """Tests for Preprocessor.get_item_frequency()"""

    def test_returns_dataframe(self, preprocessor):
        """Should return a pandas DataFrame."""
        result = preprocessor.get_item_frequency()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, preprocessor):
        """Should have all expected columns."""
        result = preprocessor.get_item_frequency()
        assert "category_id" in result.columns
        assert "name"        in result.columns
        assert "frequency"   in result.columns
        assert "percentage"  in result.columns

    def test_sorted_by_frequency_descending(self, preprocessor):
        """Results should be sorted highest frequency first."""
        result = preprocessor.get_item_frequency()
        frequencies = result["frequency"].tolist()
        assert frequencies == sorted(frequencies, reverse=True)

    def test_correct_item_counts(self, preprocessor):
        """
        From our sample data:
        Item 1 (T-Shirts) appears in baskets: [1,2,3],[1,3,5],[1,2,5] = 3
        Item 2 (Jeans)    appears in baskets: [1,2,3],[2,3,4],[1,2,5] = 3
        Item 3 (Jackets)  appears in baskets: [1,2,3],[2,3,4],[1,3,5] = 3
        Item 4 (Shoes)    appears in baskets: [2,3,4]                 = 1
        Item 5 (Socks)    appears in baskets: [1,3,5],[1,2,5]         = 2
        """
        result = preprocessor.get_item_frequency()
        tshirts = result[result["name"] == "T-Shirts"]["frequency"].values[0]
        shoes   = result[result["name"] == "Shoes"]["frequency"].values[0]
        socks   = result[result["name"] == "Socks"]["frequency"].values[0]
        assert tshirts == 3
        assert shoes   == 1
        assert socks   == 2

    def test_percentage_sums_to_100(self, preprocessor):
        """All percentages should sum to 100."""
        result = preprocessor.get_item_frequency()
        assert abs(result["percentage"].sum() - 100.0) < 0.01


# ===========================================================
# TESTS: get_top_customer_per_item()
# ===========================================================

class TestGetTopCustomerPerItem:
    """Tests for Preprocessor.get_top_customer_per_item()"""

    def test_returns_dataframe(self, preprocessor):
        """Should return a pandas DataFrame."""
        result = preprocessor.get_top_customer_per_item()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, preprocessor):
        """Should have all expected columns."""
        result = preprocessor.get_top_customer_per_item()
        assert "category_id"    in result.columns
        assert "name"           in result.columns
        assert "customer_id"    in result.columns
        assert "purchase_count" in result.columns

    def test_one_row_per_item(self, preprocessor):
        """Should return exactly one top customer per item."""
        result = preprocessor.get_top_customer_per_item()
        assert len(result) == 5

    def test_correct_top_customer_for_item1(self, preprocessor):
        """
        Item 1 (T-Shirts):
        Customer 1001 bought it in 2 baskets ([1,2,3] and [1,2,5])
        Customer 1003 bought it in 1 basket  ([1,3,5])
        So top customer = 1001 with count 2
        """
        result = preprocessor.get_top_customer_per_item()
        tshirts_row = result[result["name"] == "T-Shirts"]
        assert tshirts_row["customer_id"].values[0]    == 1001
        assert tshirts_row["purchase_count"].values[0] == 2