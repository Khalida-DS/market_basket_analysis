"""
app.py — Streamlit Dashboard
=============================

HOW TO RUN:
-----------
    cd market_basket_analysis
    conda activate market_basket
    streamlit run dashboard/app.py

PAGES:
------
    1. Overview         — item popularity + top customers
    2. Association Rules — rules scatter + co-occurrence heatmap
    3. Recommender      — live product recommendations
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader   import DataLoader
from src.preprocessor  import Preprocessor
from src.analyzer      import Analyzer
from src.recommender   import Recommender
from src.visualizer    import Visualizer
from src.config        import RECOMMENDER_TOP_N

# ===========================================================
# PAGE CONFIGURATION
# ===========================================================

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #2E86AB;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2E86AB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================
# DATA LOADING — cached so pipeline runs once
# ===========================================================

@st.cache_data
def load_data():
    loader                    = DataLoader()
    baskets_df, categories_df = loader.load_all()

    preprocessor  = Preprocessor(baskets_df, categories_df)
    one_hot_df    = preprocessor.build_one_hot_matrix()
    freq_df       = preprocessor.get_item_frequency()
    stats         = preprocessor.get_basket_stats()
    top_customers = preprocessor.get_top_customer_per_item()

    analyzer  = Analyzer(one_hot_df)
    rules_df  = analyzer.run()
    summary   = analyzer.get_summary()

    return (
        baskets_df,
        categories_df,
        freq_df,
        stats,
        top_customers,
        rules_df,
        summary,
    )


# ===========================================================
# SIDEBAR
# ===========================================================

st.sidebar.image(
    "https://img.icons8.com/fluency/96/shopping-cart.png",
    width=80,
)
st.sidebar.title("Market Basket Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Association Rules", "🎯 Recommender"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About this project:**
End-to-end association rule mining pipeline
built with Python, mlxtend, and Streamlit.

**Techniques used:**
- Apriori algorithm
- Zhang's metric
- Co-occurrence analysis
""")


# ===========================================================
# LOAD DATA
# ===========================================================

with st.spinner("Loading data and running pipeline..."):
    try:
        (
            baskets_df,
            categories_df,
            freq_df,
            stats,
            top_customers,
            rules_df,
            summary,
        ) = load_data()

        viz         = Visualizer(save_figures=False)
        data_loaded = True

    except FileNotFoundError as e:
        data_loaded = False
        st.error(f"""
        **Data files not found.**
        Place CSV files in `data/raw/`:
        - `customer_baskets.csv`
        - `clothing_categories.csv`

        Error: {e}
        """)


# ===========================================================
# PAGE 1 — OVERVIEW
# ===========================================================

if data_loaded and page == "🏠 Overview":

    st.title("🛒 Market Basket Analysis — Overview")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_transactions']:,}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(categories_df):,}</div>
            <div class="metric-label">Item Categories</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['mean']}</div>
            <div class="metric-label">Avg Basket Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_rules', 0):,}</div>
            <div class="metric-label">Association Rules</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Chart 1 — Item Frequency
    st.subheader("📦 Most Popular Item Categories")
    st.markdown(
        "Which categories appear in the most transactions. "
        "Drives inventory and promotional decisions."
    )

    top_n_items = st.slider(
        "Number of categories to show",
        min_value=5,
        max_value=min(48, len(freq_df)),
        value=20,
        step=5,
        key="freq_slider",
    )

    fig_freq = viz.plot_item_frequency(freq_df, top_n=top_n_items)
    st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("---")

    # Chart 2 — Top Customers
    st.subheader("👥 Top Customers by Total Items Purchased")
    st.markdown(
        "Customers ranked by total items bought across all transactions. "
        "The retailer uses this to identify customers for interviews."
    )

    top_n_cust = st.slider(
        "Number of customers to show",
        min_value=5,
        max_value=30,
        value=20,
        step=5,
        key="cust_slider",
    )

    fig_customers = viz.plot_top_customers(
        baskets_df, top_n=top_n_cust
    )
    st.plotly_chart(fig_customers, use_container_width=True)

    st.markdown("---")

    # Top customer per item table
    st.subheader("🏆 Top Customer Per Item Category")
    st.markdown(
        "For each item category, the customer who purchased it most. "
        "Direct input for retailer interview list."
    )
    st.dataframe(
        top_customers,
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================
# PAGE 2 — ASSOCIATION RULES
# ===========================================================

elif data_loaded and page == "📊 Association Rules":

    st.title("📊 Association Rules Explorer")
    st.markdown("---")

    if len(rules_df) == 0:
        st.warning(
            "No rules found. Try lowering thresholds in config.py"
        )
    else:

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules",    f"{summary.get('total_rules', 0):,}")
        with col2:
            st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.3f}")
        with col3:
            st.metric("Avg Lift",       f"{summary.get('avg_lift', 0):.3f}")
        with col4:
            st.metric("Avg Zhang",      f"{summary.get('avg_zhang', 0):.3f}")

        st.markdown("---")

        # Chart 3 — Rules Scatter
        st.subheader("🎯 Cross-sell Rule Quality — Confidence vs Lift")
        st.markdown(
            "Each dot = one rule. "
            "**Top-right** = high confidence + high lift = best cross-sell opportunity. "
            "**Color** = Zhang's metric (darker = more genuine association). "
            "Hover any dot for rule details."
        )
        fig_scatter = viz.plot_rules_scatter(rules_df)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Chart 4 — Co-occurrence Heatmap
        st.subheader("🗺️ Item Co-occurrence Heatmap")
        st.markdown(
            "Darker = higher confidence that row item leads to column item. "
            "Use this for **shelf placement** and **bundle promotions**."
        )

        top_n_heat = st.slider(
            "Number of items to include",
            min_value=10,
            max_value=48,
            value=20,
            step=5,
            key="heat_slider",
        )

        fig_heatmap = viz.plot_cooccurrence_heatmap(
            rules_df, top_n=top_n_heat
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")

        # Filterable rules table
        st.subheader("📋 All Rules — Filterable Table")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_conf = st.slider(
                "Min confidence", 0.2, 0.65, 0.2, 0.05
            )
        with col2:
            min_lift = st.slider(
                "Min lift", 1.0, 5.0, 1.0, 0.1
            )
        with col3:
            min_zhang = st.slider(
                "Min Zhang's metric", 0.0, 1.0, 0.0, 0.05
            )

        filtered = rules_df[
            (rules_df["confidence"]    >= min_conf)  &
            (rules_df["lift"]          >= min_lift)  &
            (rules_df["zhangs_metric"] >= min_zhang)
        ].copy()

        filtered["antecedents"] = filtered["antecedents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
        )
        filtered["consequents"] = filtered["consequents"].apply(
            lambda x: ", ".join(sorted(str(i) for i in x))
        )

        st.info(f"Showing {len(filtered):,} rules")
        st.dataframe(
            filtered[[
                "antecedents", "consequents",
                "support", "confidence",
                "lift", "zhangs_metric",
            ]].round(4),
            use_container_width=True,
            hide_index=True,
        )


# ===========================================================
# PAGE 3 — RECOMMENDER
# ===========================================================

elif data_loaded and page == "🎯 Recommender":

    st.title("🎯 Live Product Recommender")
    st.markdown("---")

    recommender = Recommender(rules_df)

    all_items = sorted(
        freq_df["name"].dropna().astype(str).tolist()
    )

    st.subheader("Build Your Basket")
    st.markdown(
        "Select items a customer currently has in their basket. "
        "The engine finds matching association rules and recommends "
        "what to add next."
    )

    selected_items = st.multiselect(
        "Current basket items",
        options=all_items,
        default=[],
        placeholder="Select items...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        top_n = st.number_input(
            "Max recommendations",
            min_value=1,
            max_value=20,
            value=RECOMMENDER_TOP_N,
        )

    if selected_items:
        st.markdown("---")
        st.subheader(f"Basket: {', '.join(selected_items)}")

        recommendations = recommender.recommend(
            selected_items,
            top_n=int(top_n),
        )

        if len(recommendations) > 0:
            st.success(
                f"Found {len(recommendations)} recommendations"
            )

            cols = st.columns(min(len(recommendations), 3))
            for idx, (_, row) in enumerate(
                recommendations.iterrows()
            ):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value"
                             style="font-size:1.2rem">
                            {row['item']}
                        </div>
                        <div class="metric-label">
                            Confidence: {row['confidence']:.0%}<br>
                            Lift: {row['lift']:.2f}<br>
                            Zhang: {row['zhangs_metric']:.3f}
                        </div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Recommendation Details")
            st.dataframe(
                recommendations,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No recommendations found for this basket. "
                "Try adding more items or different combinations."
            )

    else:
        st.info("Select items above to get recommendations.")

    st.markdown("---")

    st.subheader("🌟 Popular Items — Cold Start")
    st.markdown(
        "For new customers with no history — "
        "recommend universally popular items:"
    )

    popular = recommender.get_popular_recommendations(top_n=10)
    if len(popular) > 0:
        st.dataframe(
            popular,
            use_container_width=True,
            hide_index=True,
        )
