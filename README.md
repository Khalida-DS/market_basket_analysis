# Market Basket Analysis — Retail Recommendation Engine

> End-to-end association rule mining pipeline transforming raw transaction data
> into actionable product recommendations via a live Streamlit dashboard.

---

## Overview

This project refactors a Jupyter notebook proof-of-concept into a
production-grade Python pipeline. It applies the Apriori algorithm
with Zhang's metric to identify genuine product associations in a retail
clothing dataset of **36,316 transactions across 48 item categories**,
then surfaces those associations through a real-time recommendation engine
and an interactive Streamlit dashboard.

The key engineering decisions — replacing `iterrows()` with vectorized
operations, fixing an off-by-one bug in frequency counting, introducing
Zhang's metric to correct lift's popularity bias — are documented
alongside every module. **90 unit tests** verify all functionality.

---

## Live Demo

```bash
conda activate market_basket
streamlit run dashboard/app.py
```

| Page | What It Shows |
|---|---|
| Overview | Transaction volume, item popularity bar chart, basket size distribution |
| Association Rules | Confidence vs lift scatter, network graph, filterable rules table |
| Recommender | Live basket builder — select items, get ranked recommendations instantly |

---

## Architecture

```
market_basket_analysis/
│
├── src/
│   ├── config.py          # Single source of truth — all thresholds and paths
│   ├── data_loader.py     # CSV ingestion, schema validation, logging
│   ├── preprocessor.py    # One-hot encoding, basket stats, item frequency
│   ├── analyzer.py        # Apriori + Zhang's metric + rule filtering
│   ├── recommender.py     # Rule-based recommendation engine
│   └── visualizer.py      # Plotly charts — bar, histogram, scatter, network
│
├── dashboard/
│   └── app.py             # Streamlit dashboard — 3-page interactive app
│
├── tests/
│   ├── test_data_loader.py    # 18 tests
│   ├── test_preprocessor.py   # 22 tests
│   ├── test_analyzer.py       # 18 tests
│   ├── test_recommender.py    # 16 tests
│   └── test_visualizer.py     # 16 tests
│
├── data/
│   └── raw/
│       ├── customer_baskets.csv
│       └── clothing_categories.csv
│
├── main.py                # Pipeline entry point
├── conftest.py            # pytest path resolution
└── requirements.txt
```

**Design principle:** each module has one job and one reason to change.
`data_loader.py` loads and validates. `preprocessor.py` transforms.
`analyzer.py` mines rules. `recommender.py` generates recommendations.
`visualizer.py` renders charts. Nothing overlaps.

---

## Pipeline

```
customer_baskets.csv          clothing_categories.csv
        │                               │
        └──────────┬────────────────────┘
                   ▼
            DataLoader
         (validate + load)
                   │
                   ▼
            Preprocessor
    ┌──────────────────────────┐
    │  build_one_hot_matrix()  │  → TransactionEncoder (replaces iterrows)
    │  get_basket_stats()      │  → mean, median, skew, kurtosis
    │  get_item_frequency()    │  → dictionary-based (fixes off-by-one)
    │  get_top_customer()      │  → explode() + groupby() + idxmax()
    └──────────────────────────┘
                   │
                   ▼
              Analyzer
    ┌──────────────────────────┐
    │  Apriori algorithm       │  → 5,305 frequent itemsets
    │  association_rules()     │  → 29,562 raw rules
    │  Zhang's metric          │  → corrects lift's popularity bias
    │  filter_rules()          │  → confidence ≥ 0.2, lift > 1.0
    └──────────────────────────┘
                   │
                   ▼
            Recommender
    ┌──────────────────────────┐
    │  recommend()             │  → antecedents ⊆ basket → consequents
    │  recommend_for_customer()│  → lookup by customer_id
    │  get_popular_items()     │  → cold-start fallback
    └──────────────────────────┘
                   │
                   ▼
            Visualizer + Dashboard
```

---

## Key Engineering Decisions

### 1. Replacing iterrows() — 90× Performance Improvement

The original notebook built the one-hot matrix using nested Python loops:

```python
# Original — 1,743,168 Python iterations
for index, row in df.iterrows():
    for i in range(1, 49):
        if i in row['basket_items']:
            df.loc[index, col_item] = 1
```

Replaced with `mlxtend.TransactionEncoder` — compiled C operations
that process the entire array at once:

```python
# Senior — vectorized, ~0.5s vs ~45s
encoder = TransactionEncoder()
matrix  = encoder.fit(transactions).transform(transactions)
```

| Approach | Time |
|---|---|
| Original iterrows() | ~45 seconds |
| TransactionEncoder | ~0.5 seconds |
| **Improvement** | **~90×** |

---

### 2. Fixing the Off-by-One Bug in Item Frequency

The original code stored counts in a list indexed from 0, but item IDs
started at 1. The `- 1` correction subtracted from the wrong index,
silently returning the wrong item with no error thrown:

```python
# Original — always returns wrong item
poplr_item = [0] * 49
poplr_item = df_1.loc[[poplr_item.index(max(poplr_item)) - 1]]
#                                                          ^^^
```

Fixed with a dictionary where the key IS the item_id — no arithmetic needed:

```python
# Senior — key is the item_id, no index confusion possible
item_counts = {}
for basket in self.baskets_df["basket_items"]:
    for item_id in basket:
        item_counts[item_id] = item_counts.get(item_id, 0) + 1
```

---

### 3. Zhang's Metric — Correcting Lift's Popularity Bias

Lift alone can mislead when items are universally popular.
Zhang's metric measures whether item A **genuinely increases** the
probability of item B, rather than reflecting background popularity:

```
Zhang(A → B) = (P(A∩B) − P(A)·P(B)) /
               max(P(A∩B)·(1−P(A)), P(A)·(P(B)−P(A∩B)))

Range:  +1 = perfect positive association
         0 = no association (independent)
        −1 = perfect negative association

Filter: zhang > 0.0 keeps only rules where A genuinely increases P(B)
```

---

### 4. Top Customer Per Item — 3 Vectorized Operations vs 48 Loops

Original used 48 separate groupby operations in a loop.
Replaced with a single pipeline:

```python
# Senior — explode() + groupby() + idxmax()
exploded = (
    baskets_df[["customer_id", "basket_items"]]
    .explode("basket_items")
    .rename(columns={"basket_items": "item_id"})
)
purchase_counts = (
    exploded
    .groupby(["item_id", "customer_id"])
    .size()
    .reset_index(name="purchase_count")
)
top_customers = purchase_counts.loc[
    purchase_counts
    .groupby("item_id")["purchase_count"]
    .idxmax()
]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | pandas 2.x, numpy |
| ML | mlxtend (Apriori, TransactionEncoder) |
| Visualisation | Plotly, NetworkX |
| Dashboard | Streamlit |
| Logging | loguru |
| Testing | pytest (90 tests) |
| Environment | Conda |
| Version control | Git — feature branch per phase, PR per merge |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Khalida-DS/market_basket_analysis.git
cd market_basket_analysis

# Create and activate environment
conda create -n market_basket python=3.11 -y
conda activate market_basket

# Install dependencies
pip install -r requirements.txt

# Add your data files
cp your_data/customer_baskets.csv     data/raw/
cp your_data/clothing_categories.csv  data/raw/
```

---

## Usage

**Run the full pipeline (terminal output):**
```bash
python main.py
```

**Launch the interactive dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run all tests:**
```bash
pytest tests/ -v
# 90 passed in ~2s
```

---

## Data Format

**customer_baskets.csv**
```
customer_id,basket
75689161,"34,13,42,11,5"
37394281,"25,32,10,3"
```

**clothing_categories.csv**
```
category_id,name,description
1,T-Shirts,Casual shirts
2,Jeans,Denim trousers
```

---

## Test Coverage

```
tests/test_data_loader.py    18 tests  — loading, validation, error handling
tests/test_preprocessor.py   22 tests  — encoding, stats, frequency, bug fix
tests/test_analyzer.py       18 tests  — Apriori, Zhang's metric, filtering
tests/test_recommender.py    16 tests  — recommendations, edge cases
tests/test_visualizer.py     16 tests  — chart creation, empty data handling
─────────────────────────────────────────────────────────────────────────────
Total                        90 tests  — all passing in 1.93s
```

Every test uses controlled fixtures — not the real data files.
This keeps tests fast, stable, and CI-friendly.

---

## Project Build Log

| Phase | Module | What Was Built |
|---|---|---|
| 1 | `data_loader.py` | CSV ingestion, schema validation, loguru logging |
| 2 | `preprocessor.py` | One-hot encoding, bug fixes, vectorized transforms |
| 3 | `analyzer.py` | Apriori pipeline + Zhang's metric implementation |
| 4 | `recommender.py` | Rule-based recommendation engine |
| 5 | `visualizer.py` | 4 interactive Plotly charts |
| 6 | `dashboard/app.py` | 3-page Streamlit dashboard |

Each phase was developed on a feature branch, tested before merging,
and merged to `main` via pull request. The Git history reflects
the exact build order.

---

## Configuration

All thresholds and paths live in `src/config.py` — one place to change,
everything adjusts:

```python
APRIORI_MIN_SUPPORT     = 0.01   # items in 1%+ of transactions
APRIORI_MIN_CONFIDENCE  = 0.20   # rule correct 20%+ of the time
APRIORI_MIN_LIFT        = 1.0    # better than random chance
APRIORI_MIN_ZHANG       = 0.0    # genuine positive association only
RECOMMENDER_TOP_N       = 5      # max recommendations per basket
```

---

## What This Refactoring Demonstrates

| Skill | Evidence |
|---|---|
| Performance engineering | 90× speedup replacing iterrows() with vectorized ops |
| Bug identification | Found and fixed silent off-by-one error with test proof |
| Statistical depth | Zhang's metric implemented from formula, not just used |
| Software design | Single responsibility — 5 modules, zero overlap |
| Test discipline | 90 tests, fixtures not real data, all under 2 seconds |
| Git workflow | Feature branches, conventional commits, PR per phase |
| Full-stack DS | Data → ML → API → Dashboard in one coherent codebase |

---

## Author

**Khalida** — Data Scientist
[GitHub: Khalida-DS](https://github.com/Khalida-DS)

---

*Built February 2026 — Market Basket Analysis Capstone Project*
