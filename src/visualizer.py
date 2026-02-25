"""
visualizer.py — Data Visualization
=====================================

CHARTS — 4 business-relevant visualizations:
----------------------------------------------
1. plot_item_frequency()       → what sells most (inventory + promotions)
2. plot_top_customers()        → who buys most (loyalty + retailer interviews)
3. plot_rules_scatter()        → cross-sell rule quality (confidence vs lift)
4. plot_cooccurrence_heatmap() → which items cluster (shelf placement + bundles)

HOW TO USE:
-----------
    from src.visualizer import Visualizer

    viz = Visualizer()
    fig = viz.plot_item_frequency(freq_df)
    fig = viz.plot_top_customers(baskets_df)
    fig = viz.plot_rules_scatter(rules_df)
    fig = viz.plot_cooccurrence_heatmap(rules_df)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger

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
    Creates business-relevant charts for Market Basket Analysis.

    4 charts — each answers a specific business question:
      1. What sells most?
      2. Who are the top customers?
      3. Which cross-sell rules are strongest?
      4. Which items naturally cluster together?

    Usage:
        viz = Visualizer()
        fig = viz.plot_item_frequency(freq_df)
        fig.show()
    """

    def __init__(self, save_figures: bool = False):
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
    # CHART 1 — Item Frequency
    # Business question: What sells most?
    # Original notebook: px.bar df_1 x="name" y="item_frequency"
    # -------------------------------------------------------

    def plot_item_frequency(
        self,
        freq_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Most Popular Item Categories",
    ) -> go.Figure:
        """
        Horizontal bar chart of item frequency.
        Shows which categories appear in the most transactions.
        Directly answers: what should we promote and stock more of?

        Args:
            freq_df: Output from Preprocessor.get_item_frequency()
            top_n:   Number of top items to show
            title:   Chart title

        Returns:
            plotly Figure object
        """
        logger.info(f"Plotting item frequency (top {top_n})...")

        plot_df = freq_df.head(top_n).copy()

        fig = px.bar(
            plot_df,
            x="frequency",
            y="name",
            orientation="h",
            color="percentage",
            color_continuous_scale=[
                "#90cdf4",
                self.colors["primary"],
                "#1a365d",
            ],
            title=title,
            labels={
                "frequency" : "Number of Transactions",
                "name"      : "Item Category",
                "percentage": "% of Total Sales",
            },
            text=plot_df["percentage"].apply(lambda x: f"{x:.1f}%"),
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(size=11),
        )

        fig.update_layout(
            **self._base_layout(),
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=True,
        )

        logger.info("  ✓ Item frequency chart created")
        self._save_figure(fig, "item_frequency.html")
        return fig

    # -------------------------------------------------------
    # CHART 2 — Top Customers
    # Business question: Who buys the most?
    # Original notebook: px.bar df_total_10 x="customer_id_str"
    # -------------------------------------------------------

    def plot_top_customers(
        self,
        baskets_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Top Customers by Total Items Purchased",
    ) -> go.Figure:
        """
        Bar chart of top customers ranked by total items bought.

        Directly from original notebook logic:
            df_total = df.groupby('customer_id')['basket_size'].sum()

        Business value:
            - Retailer wants to interview top customers
            - Identify loyalty program candidates
            - Understand high-value customer behaviour

        Args:
            baskets_df: Output from DataLoader.load_baskets()
            top_n:      Number of top customers to show
            title:      Chart title

        Returns:
            plotly Figure object
        """
        logger.info(f"Plotting top {top_n} customers...")

        df_total = (
            baskets_df
            .groupby("customer_id")["basket_size"]
            .sum()
            .reset_index()
            .rename(columns={"basket_size": "total_items"})
            .sort_values("total_items", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        total_all = df_total["total_items"].sum()
        df_total["percentage"] = (
            df_total["total_items"] / total_all * 100
        ).round(3)

        df_total["customer_id_str"] = (
            "#" + df_total.index.astype(str).str.zfill(2)
            + " — " + df_total["customer_id"].astype(str)
        )
        df_total["rank"] = df_total.index + 1

        fig = px.bar(
            df_total,
            x="customer_id_str",
            y="total_items",
            color="total_items",
            color_continuous_scale=[
                "#90cdf4",
                self.colors["primary"],
                "#1a365d",
            ],
            title=title,
            labels={
                "customer_id_str": "Customer ID",
                "total_items"    : "Total Items Purchased",
            },
            text=df_total["percentage"].apply(
                lambda x: f"{x:.2f}%"
            ),
            hover_data={
                "customer_id_str": True,
                "total_items"    : True,
                "percentage"     : ":.3f",
                "rank"           : True,
            },
        )

        fig.update_traces(
            textposition="outside",
            textfont=dict(size=10),
        )

        fig.update_layout(
            **self._base_layout(),
            xaxis=dict(
                title="Customer ID",
                tickangle=-45,
                tickfont=dict(size=10),
            ),
            yaxis_title="Total Items Purchased",
            coloraxis_showscale=False,
            showlegend=False,
        )

        logger.info(
            f"  ✓ Top customers chart created — "
            f"top {top_n} of "
            f"{baskets_df['customer_id'].nunique():,} customers"
        )
        self._save_figure(fig, "top_customers.html")
        return fig

    # -------------------------------------------------------
    # CHART 3 — Rules Scatter
    # Business question: Which cross-sell rules are strongest?
    # -------------------------------------------------------

    def plot_rules_scatter(
        self,
        rules_df: pd.DataFrame,
        title: str = "Association Rules — Confidence vs Lift",
    ) -> go.Figure:
        """
        Scatter plot of all association rules.
        X axis: confidence — how reliable is the rule?
        Y axis: lift       — how much better than random?
        Color:  Zhang's metric — is the association genuine?
        Size:   support    — how common is this pattern?

        Business value:
            Rules in the top-right corner (high confidence + high lift)
            are the best cross-sell opportunities.
            Color (Zhang) confirms the association is real, not popularity bias.

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

        plot_df = rules_df.copy()

        plot_df["antecedents_str"] = plot_df["antecedents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
        )
        plot_df["consequents_str"] = plot_df["consequents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
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
                "#90cdf4",
                self.colors["primary"],
                "#1a365d",
            ],
            title=title,
            labels={
                "confidence"   : "Confidence",
                "lift"         : "Lift",
                "zhangs_metric": "Zhang's Metric",
                "support"      : "Support",
            },
        )

        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            opacity=0.4,
            annotation_text="Lift = 1.0 (random chance)",
            annotation_position="bottom right",
        )

        fig.update_layout(**self._base_layout())

        logger.info(
            f"  ✓ Rules scatter chart created — "
            f"{len(rules_df):,} rules"
        )
        self._save_figure(fig, "rules_scatter.html")
        return fig

    # -------------------------------------------------------
    # CHART 4 — Co-occurrence Heatmap
    # Business question: Which items cluster together?
    # Replaces network graph — more readable at scale
    # -------------------------------------------------------

    def plot_cooccurrence_heatmap(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Item Co-occurrence — Confidence Heatmap",
    ) -> go.Figure:
        """
        Heatmap of confidence between every item pair.
        Row = antecedent (IF customer bought this)
        Col = consequent (THEN they also bought this)
        Color = confidence of that rule

        Business value:
            Darker cells = strongest cross-sell opportunities
            Cluster patterns reveal natural product groupings
            Directly informs shelf placement and bundle pricing

        Args:
            rules_df: Output from Analyzer.run()
            top_n:    Number of top items to include
            title:    Chart title

        Returns:
            plotly Figure object
        """
        logger.info("Plotting co-occurrence heatmap...")

        if len(rules_df) == 0:
            logger.warning("  No rules to plot")
            return go.Figure()

        # Single-item rules only → clean readable heatmap
        single_rules = rules_df[
            (rules_df["antecedents"].apply(len) == 1) &
            (rules_df["consequents"].apply(len) == 1)
        ].copy()

        if len(single_rules) == 0:
            logger.warning("  No single-item rules found")
            return go.Figure()

        single_rules["ant"] = single_rules["antecedents"].apply(
            lambda x: str(list(x)[0])
        )
        single_rules["con"] = single_rules["consequents"].apply(
            lambda x: str(list(x)[0])
        )

        # Top N items by appearance in rules
        top_items = (
            pd.concat([single_rules["ant"], single_rules["con"]])
            .value_counts()
            .head(top_n)
            .index.tolist()
        )

        filtered = single_rules[
            single_rules["ant"].isin(top_items) &
            single_rules["con"].isin(top_items)
        ]

        pivot = (
            filtered
            .pivot_table(
                index="ant",
                columns="con",
                values="confidence",
                aggfunc="max",
            )
            .fillna(0)
            .reindex(index=top_items, columns=top_items, fill_value=0)
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#f0f4f8"],
                [0.3, "#90cdf4"],
                [0.6, self.colors["primary"]],
                [1.0, "#1a365d"],
            ],
            hoverongaps=False,
            hovertemplate=(
                "<b>IF: %{y}</b><br>"
                "<b>THEN: %{x}</b><br>"
                "Confidence: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Confidence",
                thickness=15,
                len=0.8,
            ),
        ))

        fig.update_layout(
            **self._base_layout(),
            title=title,
            xaxis=dict(
                title="THEN buy this →",
                tickangle=-45,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                title="← IF bought this",
                tickfont=dict(size=10),
            ),
        )

        logger.info(
            f"  ✓ Co-occurrence heatmap created — "
            f"{len(top_items)} × {len(top_items)} items"
        )
        self._save_figure(fig, "cooccurrence_heatmap.html")
        return fig

    # -------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------

    def _base_layout(self) -> dict:
        """Consistent layout applied to all charts."""
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
            "margin"       : dict(l=60, r=60, t=80, b=80),
        }

    def _save_figure(self, fig: go.Figure, filename: str) -> None:
        """Save figure to outputs/figures/ if save_figures=True."""
        if self.save_figures:
            output_path = FIGURES_DIR / filename
            fig.write_html(str(output_path))
            logger.debug(f"  Figure saved: {output_path}")
