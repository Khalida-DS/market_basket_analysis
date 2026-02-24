"""
visualizer.py — Data Visualization
=====================================

WHY THIS FILE EXISTS:
---------------------
The original notebook had scattered matplotlib calls
with no consistent styling, no reusable functions,
and no way to generate charts programmatically.

This module centralizes all visualization logic:
  - Consistent color palette (from config.py)
  - Reusable chart functions
  - Charts saved to outputs/figures/
  - Clean interface for dashboard/app.py

CHARTS PRODUCED:
----------------
1. plot_item_frequency()     → bar chart of popular items
2. plot_basket_distribution() → histogram of basket sizes
3. plot_rules_scatter()      → confidence vs lift scatter
4. plot_network_graph()      → association rules network

HOW TO USE:
-----------
    from src.visualizer import Visualizer

    viz = Visualizer()
    fig = viz.plot_item_frequency(freq_df)
    fig = viz.plot_basket_distribution(stats, baskets_df)
    fig = viz.plot_rules_scatter(rules_df)
    fig = viz.plot_network_graph(rules_df)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
from loguru import logger
from typing import Optional

from src.config import (
    VIZ_PRIMARY_COLOR,
    VIZ_SECONDARY_COLOR,
    VIZ_ACCENT_COLOR,
    VIZ_NEUTRAL_COLOR,
    VIZ_BACKGROUND,
    VIZ_FIGURE_WIDTH,
    VIZ_FIGURE_HEIGHT,
    FIGURES_DIR,
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
# VISUALIZER CLASS
# ===========================================================

class Visualizer:
    """
    Creates all charts for the Market Basket Analysis project.

    WHY A CLASS:
    - Stores color palette and styling once
    - All charts have consistent look and feel
    - Easy to change theme in one place
    - Clean interface for Streamlit dashboard

    Usage:
        viz = Visualizer()
        fig = viz.plot_item_frequency(freq_df)
        fig.show()   # opens in browser
    """

    def __init__(self, save_figures: bool = False):
        """
        Initialize Visualizer with styling configuration.

        Args:
            save_figures: If True, saves charts to
                         outputs/figures/ as HTML files
        """
        self.save_figures = save_figures
        self.colors = {
            "primary"   : VIZ_PRIMARY_COLOR,
            "secondary" : VIZ_SECONDARY_COLOR,
            "accent"    : VIZ_ACCENT_COLOR,
            "neutral"   : VIZ_NEUTRAL_COLOR,
            "background": VIZ_BACKGROUND,
        }
        self.width  = VIZ_FIGURE_WIDTH
        self.height = VIZ_FIGURE_HEIGHT

        if save_figures:
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Visualizer initialized")

    # -------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------

    def plot_item_frequency(
        self,
        freq_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Most Popular Item Categories",
    ) -> go.Figure:
        """
        Bar chart of item frequency — most popular items.

        Args:
            freq_df: Output from Preprocessor.get_item_frequency()
            top_n:   Number of top items to show
            title:   Chart title

        Returns:
            plotly Figure object
        """
        logger.info(f"Plotting item frequency (top {top_n})...")

        # Take top N items
        plot_df = freq_df.head(top_n).copy()

        fig = px.bar(
            plot_df,
            x="frequency",
            y="name",
            orientation="h",
            color="percentage",
            color_continuous_scale=[
                self.colors["primary"],
                self.colors["accent"],
            ],
            title=title,
            labels={
                "frequency" : "Number of Transactions",
                "name"      : "Item Category",
                "percentage": "% of Total",
            },
            text="frequency",
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
        )

        fig.update_layout(
            **self._base_layout(),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=True,
        )

        logger.info("  ✓ Item frequency chart created")
        self._save_figure(fig, "item_frequency.html")
        return fig

    def plot_basket_distribution(
        self,
        baskets_df: pd.DataFrame,
        title: str = "Basket Size Distribution",
    ) -> go.Figure:
        """
        Histogram of basket sizes with statistics overlay.

        Args:
            baskets_df: Output from DataLoader.load_baskets()
            title:      Chart title

        Returns:
            plotly Figure object
        """
        logger.info("Plotting basket size distribution...")

        sizes = baskets_df["basket_size"]

        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(
            x=bins,
            name="Basket Sizes",
            marker_color=self.colors["primary"],
            opacity=0.8,
            nbinsx=30,
        ))

        # Mean line
        fig.add_vline(
            x=sizes.mean(),
            line_dash="dash",
            line_color=self.colors["accent"],
            annotation_text=f"Mean: {sizes.mean():.1f}",
            annotation_position="top right",
        )

        # Median line
        fig.add_vline(
            x=sizes.median(),
            line_dash="dot",
            line_color=self.colors["secondary"],
            annotation_text=f"Median: {sizes.median():.1f}",
            annotation_position="top left",
        )

        fig.update_layout(
            **self._base_layout(),
            title=title,
            xaxis_title="Number of Items in Basket",
            yaxis_title="Number of Transactions",
            showlegend=False,
        )

        logger.info("  ✓ Basket distribution chart created")
        self._save_figure(fig, "basket_distribution.html")
        return fig

    def plot_rules_scatter(
        self,
        rules_df: pd.DataFrame,
        title: str = "Association Rules — Confidence vs Lift",
    ) -> go.Figure:
        """
        Scatter plot of association rules.
        X axis: confidence
        Y axis: lift
        Color:  Zhang's metric
        Size:   support

        This is the signature chart of the project.
        Shows the quality of every rule at a glance.

        Args:
            rules_df: Output from Analyzer.run()
            title:    Chart title

        Returns:
            plotly Figure object
        """
        logger.info("Plotting rules scatter chart...")

        if len(rules_df) == 0:
            logger.warning("  No rules to plot")
            return go.Figure()

        # Convert frozensets to strings for display
        plot_df = rules_df.copy()
        plot_df["antecedents_str"] = plot_df["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        plot_df["consequents_str"] = plot_df["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        plot_df["rule_str"] = (
            plot_df["antecedents_str"]
            + " → "
            + plot_df["consequents_str"]
        )

        fig = px.scatter(
            plot_df,
            x="confidence",
            y="lift",
            color="zhangs_metric",
            size="support",
            hover_name="rule_str",
            hover_data={
                "confidence"   : ":.3f",
                "lift"         : ":.3f",
                "zhangs_metric": ":.3f",
                "support"      : ":.4f",
            },
            color_continuous_scale=[
                self.colors["primary"],
                self.colors["accent"],
            ],
            title=title,
            labels={
                "confidence"   : "Confidence",
                "lift"         : "Lift",
                "zhangs_metric": "Zhang's Metric",
                "support"      : "Support",
            },
        )

        # Add reference lines
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Lift = 1.0 (random chance)",
        )
        fig.add_vline(
            x=0.6,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Min confidence",
        )

        fig.update_layout(**self._base_layout())

        logger.info(
            f"  ✓ Rules scatter chart created "
            f"({len(rules_df)} rules)"
        )
        self._save_figure(fig, "rules_scatter.html")
        return fig

    def plot_network_graph(
        self,
        rules_df: pd.DataFrame,
        min_confidence: float = 0.65,
        title: str = "Association Rules Network",
    ) -> go.Figure:
        """
        Network graph showing item associations.
        Each node = item category
        Each edge = association rule
        Edge thickness = confidence strength

        This is the most visually impressive chart.
        Shows which items cluster together naturally.

        Args:
            rules_df:       Output from Analyzer.run()
            min_confidence: Only show rules above this threshold
            title:          Chart title

        Returns:
            plotly Figure object
        """
        logger.info("Plotting association rules network...")

        if len(rules_df) == 0:
            logger.warning("  No rules to plot")
            return go.Figure()

        # Filter to strongest rules only
        strong_rules = rules_df[
            rules_df["confidence"] >= min_confidence
        ].copy()

        if len(strong_rules) == 0:
            logger.warning(
                f"  No rules above confidence {min_confidence}"
            )
            return go.Figure()

        # Build network graph
        G = nx.DiGraph()

        for _, rule in strong_rules.iterrows():
            for ant in rule["antecedents"]:
                for con in rule["consequents"]:
                    G.add_edge(
                        ant, con,
                        weight=rule["confidence"],
                        lift=rule["lift"],
                    )

        # Calculate layout
        pos = nx.spring_layout(G, seed=42, k=2)

        # Build edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight  = edge[2]["weight"]

            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=weight * 3,
                    color=self.colors["primary"],
                ),
                opacity=0.6,
                hoverinfo="none",
            ))

        # Build node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = list(G.nodes())
        node_sizes = [
            20 + G.degree(node) * 10
            for node in G.nodes()
        ]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=self.colors["accent"],
                line=dict(
                    width=2,
                    color=self.colors["primary"],
                ),
            ),
        )

        # Combine traces
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                plot_bgcolor=self.colors["background"],
                paper_bgcolor=self.colors["background"],
                width=self.width,
                height=self.height,
            )
        )

        logger.info(
            f"  ✓ Network graph created "
            f"({G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges)"
        )
        self._save_figure(fig, "network_graph.html")
        return fig

    # -------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------

    def _base_layout(self) -> dict:
        """
        Returns base layout settings applied to all charts.
        Ensures consistent styling across every figure.
        """
        return {
            "width"        : self.width,
            "height"       : self.height,
            "plot_bgcolor" : self.colors["background"],
            "paper_bgcolor": self.colors["background"],
            "font"         : dict(
                family="Arial, sans-serif",
                size=13,
                color="#333333",
            ),
            "title_font"   : dict(size=18, color="#222222"),
            "margin"       : dict(l=60, r=60, t=80, b=60),
        }

    def _save_figure(
        self, fig: go.Figure, filename: str
    ) -> None:
        """
        Save figure to outputs/figures/ if save_figures=True.

        Args:
            fig:      Plotly figure to save
            filename: Output filename (e.g. 'item_frequency.html')
        """
        if self.save_figures:
            output_path = FIGURES_DIR / filename
            fig.write_html(str(output_path))
            logger.debug(f"  Figure saved: {output_path}")