"""
test_visualizer.py — Unit Tests for Visualizer
===============================================

HOW TO RUN:
    pytest tests/test_visualizer.py -v

WHAT WE TEST:
    - Visualizer initializes correctly
    - plot_item_frequency() returns a Figure
    - plot_basket_distribution() returns a Figure
    - plot_rules_scatter() returns a Figure
    - plot_network_graph() returns a Figure
    - Empty data is handled gracefully

NOTE:
    We test that charts are CREATED correctly.
    We do NOT test visual appearance — that's manual review.
    We verify: correct type, correct data, no crashes.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.visualizer import Visualizer


# ===========================================================
# FIXTURES
# ===========================================================

@pytest.fixture
def visualizer() -> Visualizer:
    """Creates a Visualizer with save_figures=False."""
    return Visualizer(save_figures=False)


@pytest.fixture
def sample_freq_df() -> pd.DataFrame:
    """Small item frequency DataFrame for testing."""
    return pd.DataFrame({
        "category_id": [1, 2, 3, 4, 5],
        "name"       : [
            "T-Shirts", "Jeans", "Jackets", "Shoes", "Socks"
        ],
        "description": [
            "Casual", "Denim", "Outer", "Foot", "Hosiery"
        ],
        "frequency"  : [1200, 1100, 950, 800, 750],
        "percentage" : [24.0, 22.0, 19.0, 16.0, 15.0],
    })


@pytest.fixture
def sample_baskets_df() -> pd.DataFrame:
    """Small baskets DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "customer_id" : range(100),
        "basket_size" : np.random.randint(1, 15, size=100),
    })


@pytest.fixture
def sample_rules_df() -> pd.DataFrame:
    """Small rules DataFrame for testing."""
    return pd.DataFrame({
        "antecedents"       : [
            frozenset(["T-Shirts"]),
            frozenset(["Jeans"]),
            frozenset(["Shoes"]),
        ],
        "consequents"       : [
            frozenset(["Jeans"]),
            frozenset(["Jackets"]),
            frozenset(["Socks"]),
        ],
        "support"           : [0.042, 0.038, 0.035],
        "confidence"        : [0.71,  0.68,  0.65 ],
        "lift"              : [1.8,   1.7,   1.6  ],
        "zhangs_metric"     : [0.31,  0.28,  0.22 ],
        "antecedent support": [0.06,  0.06,  0.05 ],
        "consequent support": [0.06,  0.05,  0.05 ],
    })


@pytest.fixture
def empty_rules_df() -> pd.DataFrame:
    """Empty rules DataFrame for edge case testing."""
    return pd.DataFrame(columns=[
        "antecedents", "consequents", "support",
        "confidence", "lift", "zhangs_metric",
        "antecedent support", "consequent support",
    ])


# ===========================================================
# TESTS: Initialization
# ===========================================================

class TestVisualizerInit:
    """Tests for Visualizer.__init__"""

    def test_init_stores_colors(self, visualizer):
        """Visualizer should store color palette."""
        assert isinstance(visualizer.colors, dict)
        assert "primary"    in visualizer.colors
        assert "secondary"  in visualizer.colors
        assert "accent"     in visualizer.colors
        assert "background" in visualizer.colors

    def test_init_stores_dimensions(self, visualizer):
        """Visualizer should store width and height."""
        assert visualizer.width  > 0
        assert visualizer.height > 0

    def test_init_save_figures_false(self, visualizer):
        """save_figures should be False by default in tests."""
        assert visualizer.save_figures == False


# ===========================================================
# TESTS: plot_item_frequency()
# ===========================================================

class TestPlotItemFrequency:
    """Tests for Visualizer.plot_item_frequency()"""

    def test_returns_figure(self, visualizer, sample_freq_df):
        """Should return a plotly Figure."""
        result = visualizer.plot_item_frequency(sample_freq_df)
        assert isinstance(result, go.Figure)

    def test_figure_has_data(self, visualizer, sample_freq_df):
        """Figure should contain chart data."""
        result = visualizer.plot_item_frequency(sample_freq_df)
        assert len(result.data) > 0

    def test_top_n_limits_bars(self, visualizer, sample_freq_df):
        """top_n=3 should show only 3 items."""
        result = visualizer.plot_item_frequency(
            sample_freq_df, top_n=3
        )
        assert isinstance(result, go.Figure)

    def test_custom_title(self, visualizer, sample_freq_df):
        """Custom title should appear in figure."""
        result = visualizer.plot_item_frequency(
            sample_freq_df,
            title="My Custom Title"
        )
        assert "My Custom Title" in result.layout.title.text


# ===========================================================
# TESTS: plot_basket_distribution()
# ===========================================================

class TestPlotBasketDistribution:
    """Tests for Visualizer.plot_basket_distribution()"""

    def test_returns_figure(
        self, visualizer, sample_baskets_df
    ):
        """Should return a plotly Figure."""
        result = visualizer.plot_basket_distribution(
            sample_baskets_df
        )
        assert isinstance(result, go.Figure)

    def test_figure_has_data(
        self, visualizer, sample_baskets_df
    ):
        """Figure should contain chart data."""
        result = visualizer.plot_basket_distribution(
            sample_baskets_df
        )
        assert len(result.data) > 0

    def test_custom_title(
        self, visualizer, sample_baskets_df
    ):
        """Custom title should appear in figure."""
        result = visualizer.plot_basket_distribution(
            sample_baskets_df,
            title="My Distribution"
        )
        assert "My Distribution" in result.layout.title.text


# ===========================================================
# TESTS: plot_rules_scatter()
# ===========================================================

class TestPlotRulesScatter:
    """Tests for Visualizer.plot_rules_scatter()"""

    def test_returns_figure(
        self, visualizer, sample_rules_df
    ):
        """Should return a plotly Figure."""
        result = visualizer.plot_rules_scatter(sample_rules_df)
        assert isinstance(result, go.Figure)

    def test_empty_rules_returns_figure(
        self, visualizer, empty_rules_df
    ):
        """Empty rules should return empty Figure gracefully."""
        result = visualizer.plot_rules_scatter(empty_rules_df)
        assert isinstance(result, go.Figure)

    def test_figure_has_data(
        self, visualizer, sample_rules_df
    ):
        """Figure should contain scatter data."""
        result = visualizer.plot_rules_scatter(sample_rules_df)
        assert len(result.data) > 0


# ===========================================================
# TESTS: plot_network_graph()
# ===========================================================

class TestPlotNetworkGraph:
    """Tests for Visualizer.plot_network_graph()"""

    def test_returns_figure(
        self, visualizer, sample_rules_df
    ):
        """Should return a plotly Figure."""
        result = visualizer.plot_network_graph(sample_rules_df)
        assert isinstance(result, go.Figure)

    def test_empty_rules_returns_figure(
        self, visualizer, empty_rules_df
    ):
        """Empty rules should return empty Figure gracefully."""
        result = visualizer.plot_network_graph(empty_rules_df)
        assert isinstance(result, go.Figure)

    def test_figure_has_data(
        self, visualizer, sample_rules_df
    ):
        """Figure should contain network data."""
        result = visualizer.plot_network_graph(sample_rules_df)
        assert len(result.data) > 0