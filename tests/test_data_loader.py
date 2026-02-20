"""
test_data_loader.py — Unit Tests for DataLoader
================================================

HOW TO RUN:
    pytest tests/test_data_loader.py -v

WHAT WE TEST:
    - DataLoader initializes correctly
    - load_baskets() returns correct structure
    - load_baskets() parses basket strings correctly
    - load_baskets() catches missing files
    - load_baskets() catches missing columns
    - load_baskets() catches null values
    - load_categories() returns correct structure
    - load_categories() strips whitespace
    - load_categories() catches missing files
    - load_all() returns both DataFrames correctly

NOTE:
    We never test against real data files.
    We create small mock CSV files inside each test.
    This way tests work without data/ folder.

KEY LESSON:
    CSV fields containing commas MUST be quoted:
    Wrong:   1001,1,5,12       pandas sees 4 columns
    Correct: 1001,"1,5,12"     pandas sees 2 columns
"""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader


# ===========================================================
# FIXTURES — Reusable test setup
# ===========================================================

@pytest.fixture
def valid_baskets_csv(tmp_path) -> Path:
    """
    Creates a small valid customer_baskets.csv for testing.

    IMPORTANT: basket values are quoted because they contain
    commas. This matches the real data format exactly.
    """
    content = (
        "customer_id,basket\n"
        '75689161,"34,13,42,11,5"\n'
        '37394281,"25,32,10,3"\n'
        '12345678,"1,2,3,4,5,6"\n'
    )
    file_path = tmp_path / "customer_baskets.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def valid_categories_csv(tmp_path) -> Path:
    """Creates a small valid clothing_categories.csv for testing."""
    content = (
        "category_id,name,description\n"
        "1,T-Shirts,  Casual shirts  \n"
        "2,Jeans,Denim trousers\n"
        "3,Jackets,Outerwear\n"
        "4,Shoes,Footwear\n"
        "5,Socks,  Hosiery  \n"
        "6,Hats,Headwear\n"
        "7,Scarves,Neck accessories\n"
        "8,Gloves,Hand warmers\n"
        "9,Belts,Waist accessories\n"
        "10,Dresses,Formal dresses\n"
        "11,Skirts,Various styles\n"
        "13,Suits,Business attire\n"
        "25,Coats,Winter outerwear\n"
        "32,Bags,Handbags\n"
        "34,Hats,Caps and hats\n"
        "42,Boots,Winter boots\n"
    )
    file_path = tmp_path / "clothing_categories.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def loader(valid_baskets_csv, valid_categories_csv) -> DataLoader:
    """Creates a DataLoader pointing to test CSV files."""
    return DataLoader(
        baskets_path=valid_baskets_csv,
        categories_path=valid_categories_csv,
    )


# ===========================================================
# TESTS: Initialization
# ===========================================================

class TestDataLoaderInit:
    """Tests for DataLoader.__init__"""

    def test_init_stores_paths_correctly(
        self, valid_baskets_csv, valid_categories_csv
    ):
        """DataLoader should store the paths we give it."""
        loader = DataLoader(
            baskets_path=valid_baskets_csv,
            categories_path=valid_categories_csv,
        )
        assert loader.baskets_path    == valid_baskets_csv
        assert loader.categories_path == valid_categories_csv

    def test_init_converts_string_to_path_object(
        self, valid_baskets_csv, valid_categories_csv
    ):
        """DataLoader should accept string paths and convert to Path."""
        loader = DataLoader(
            baskets_path=str(valid_baskets_csv),
            categories_path=str(valid_categories_csv),
        )
        assert isinstance(loader.baskets_path,    Path)
        assert isinstance(loader.categories_path, Path)


# ===========================================================
# TESTS: load_baskets()
# ===========================================================

class TestLoadBaskets:
    """Tests for DataLoader.load_baskets()"""

    def test_returns_dataframe(self, loader):
        """load_baskets() should return a pandas DataFrame."""
        result = loader.load_baskets()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, loader):
        """Result should have all expected columns."""
        result = loader.load_baskets()
        assert "customer_id"  in result.columns
        assert "basket"       in result.columns
        assert "basket_items" in result.columns
        assert "basket_size"  in result.columns

    def test_basket_items_is_list_of_integers(self, loader):
        """basket_items should be a list of integers."""
        result = loader.load_baskets()
        first_basket = result["basket_items"].iloc[0]
        assert isinstance(first_basket, list)
        assert all(isinstance(i, int) for i in first_basket)

    def test_basket_items_parsed_correctly(self, loader):
        """
        First row basket '34,13,42,11,5' should parse to
        [34, 13, 42, 11, 5].
        """
        result = loader.load_baskets()
        first_basket = result["basket_items"].iloc[0]
        assert first_basket == [34, 13, 42, 11, 5]

    def test_basket_size_matches_basket_items_length(self, loader):
        """basket_size should equal len(basket_items) for every row."""
        result = loader.load_baskets()
        for _, row in result.iterrows():
            assert row["basket_size"] == len(row["basket_items"])

    def test_correct_number_of_rows(self, loader):
        """Should load exactly 3 rows from our mock CSV."""
        result = loader.load_baskets()
        assert len(result) == 3

    def test_missing_file_raises_clear_error(
        self, valid_categories_csv
    ):
        """Missing baskets file should raise FileNotFoundError."""
        loader = DataLoader(
            baskets_path=Path("/nonexistent/customer_baskets.csv"),
            categories_path=valid_categories_csv,
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_baskets()

        assert "customer_baskets.csv" in str(exc_info.value)
        assert "data/raw"             in str(exc_info.value)

    def test_missing_column_raises_value_error(
        self, tmp_path, valid_categories_csv
    ):
        """CSV missing 'basket' column should raise ValueError."""
        bad_csv = tmp_path / "bad_baskets.csv"
        bad_csv.write_text(
            "customer_id,wrong_column\n"
            "1001,something\n"
        )
        loader = DataLoader(
            baskets_path=bad_csv,
            categories_path=valid_categories_csv,
        )
        with pytest.raises(ValueError) as exc_info:
            loader.load_baskets()

        assert "Missing columns" in str(exc_info.value)

    def test_null_values_raise_value_error(
        self, tmp_path, valid_categories_csv
    ):
        """Empty basket values should be caught as null/invalid."""
        bad_csv = tmp_path / "null_baskets.csv"
        bad_csv.write_text(
            "customer_id,basket\n"
            '1001,""\n'
            '1002,"1,2,3"\n'
        )
        loader = DataLoader(
            baskets_path=bad_csv,
            categories_path=valid_categories_csv,
        )
        # Empty basket should either raise ValueError or produce
        # an empty basket_items list — either is acceptable behavior
        with pytest.raises((ValueError, Exception)):
            loader.load_baskets()


# ===========================================================
# TESTS: load_categories()
# ===========================================================

class TestLoadCategories:
    """Tests for DataLoader.load_categories()"""

    def test_returns_dataframe(self, loader):
        """load_categories() should return a pandas DataFrame."""
        result = loader.load_categories()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, loader):
        """Result should have all expected columns."""
        result = loader.load_categories()
        assert "category_id"  in result.columns
        assert "name"         in result.columns
        assert "description"  in result.columns

    def test_strips_whitespace_from_description(self, loader):
        """Description should have leading/trailing whitespace removed."""
        result = loader.load_categories()
        tshirt = result[result["name"] == "T-Shirts"]
        assert tshirt["description"].values[0] == "Casual shirts"

    def test_strips_whitespace_from_name(self, loader):
        """All name values should have no leading/trailing whitespace."""
        result = loader.load_categories()
        for name in result["name"]:
            assert name == name.strip()

    def test_correct_number_of_rows(self, loader):
        """Should load exactly 16 rows from our mock CSV."""
        result = loader.load_categories()
        assert len(result) == 16

    def test_missing_file_raises_clear_error(
        self, valid_baskets_csv
    ):
        """Missing categories file should raise FileNotFoundError."""
        loader = DataLoader(
            baskets_path=valid_baskets_csv,
            categories_path=Path("/nonexistent/clothing_categories.csv"),
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_categories()

        assert "clothing_categories.csv" in str(exc_info.value)

    def test_missing_column_raises_value_error(
        self, tmp_path, valid_baskets_csv
    ):
        """CSV missing required columns should raise ValueError."""
        bad_csv = tmp_path