# Market Basket Analysis — Retail Recommendation Engine

> End-to-end association rule mining pipeline transforming 36,316 retail transactions
> into actionable cross-sell recommendations via a live interactive dashboard.

**🚀 Live Demo:** [marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app](https://marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app)
**💻 Repository:** [github.com/Khalida-DS/market_basket_analysis](https://github.com/Khalida-DS/market_basket_analysis)

---

## Business Context

A clothing retailer needs to understand which products customers buy together.
Without this insight, cross-selling is guesswork and shelf placement is arbitrary.

This project answers 4 business questions:

| Business Question | Answer Delivered |
|---|---|
| What sells most? | Item frequency chart — top 48 categories ranked |
| Who are our highest-value customers? | Top customers bar chart — ranked by total items |
| Which products should we cross-sell? | 12,670 association rules ranked by Zhang's metric |
| Which items naturally cluster? | Co-occurrence heatmap — confidence between every item pair |

---

## Business Insights

### 1. Belt is the Highest-Frequency Item

Belt appears in **13,301 transactions** — the most purchased category
across all 36,316 transactions. This makes it the strongest candidate for:
- Featured placement at checkout
- Bundle promotions with related items
- Cross-sell trigger in recommendation systems

### 2. Strong Cross-Sell Clusters Identified

The co-occurrence heatmap reveals distinct item clusters:

```
Cluster 1 — Outerwear:
  Coat → Tracksuit    confidence = 0.31   lift = 1.36
  Coat → Belt         confidence = 0.31   lift = 1.37

Cluster 2 — Occasionwear:
  Skirt + Sweatshirt → Dressing Gown + Stockings
  confidence = 0.35   lift = 1.52   zhang = 0.49

Cluster 3 — Accessories:
  Belt → Top          confidence = 0.31   lift = 1.37
  Belt → Coat         confidence = 0.31   lift = 1.36
  Belt → Tracksuit    confidence = 0.31   lift = 1.36
```

### 3. Zhang's Metric Reveals Genuine Associations

Top rules by Zhang's metric (max = 0.735):

```
Rules with high Zhang (genuine influence):
  Zhang > 0.5 → A strongly increases probability of B
  Zhang > 0.3 → A moderately increases probability of B
  Zhang > 0.0 → A has some genuine positive influence on B

Rules filtered OUT (popularity bias):
  High lift but Zhang ≈ 0 → B is just universally popular
  These rules would generate irrelevant recommendations
```

This filtering removed **16,892 rules** (57% of raw rules) that would have
generated misleading recommendations based on item popularity rather than
genuine customer behaviour.

### 4. Average Basket Contains 8 Items

```
Mean basket size  : 8.11 items
Median basket size: 8.0  items
Min basket size   : 1    item
Max basket size   : 27   items
```

With 8 items per basket on average, customers are open to broad purchases.
This supports bundle promotions of 3–5 related items rather than single add-ons.

### 5. Top Customer Insights

The top 20 customers by total items purchased each bought **55–70 items**
across multiple transactions. These customers are identified for retailer
interviews to understand purchasing motivations and inform future stocking decisions.

### 6. Recommendation Engine Performance

```
Input:  ["Belt"]
Output: Top, Thong, Coat, Tracksuit, Ball Gown
        All with confidence ≥ 0.28, lift ≥ 1.22

Input:  ["Coat", "Skirt"]
Output: Multi-item basket matching finds rules where
        ALL antecedent items are present
        Higher precision than single-item matching
```

---

## Live Dashboard

**URL:** [marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app](https://marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app)

### Page 1 — Overview
- 4 KPI cards: transactions, categories, avg basket size, association rules
- Item frequency bar chart with adjustable top N
- Top customers bar chart ranked by total items
- Top customer per item category table

### Page 2 — Association Rules
- Confidence vs Lift scatter plot — hover any dot for rule details
- Item co-occurrence heatmap — darker = stronger cross-sell signal
- Filterable rules table with confidence, lift and Zhang's metric sliders

### Page 3 — Live Recommender
- Select any items from 48 categories
- Engine matches basket against 12,670 rules in real time
- Recommendations ranked by confidence with Zhang's metric displayed
- Cold-start fallback for new customers with no history

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
│   └── visualizer.py      # 4 Plotly charts — frequency, customers, scatter, heatmap
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
│       ├── clothing_categories.csv
│       └── precomputed_rules.csv      ← pre-run for cloud deployment
│
├── main.py                # Pipeline entry point
└── requirements.txt
```

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
    │  build_one_hot_matrix()  │  → TransactionEncoder (replaces iterrows, 90× faster)
    │  get_basket_stats()      │  → mean=8.11, median=8, skew=0.42
    │  get_item_frequency()    │  → dictionary-based (fixes off-by-one bug)
    │  get_top_customer()      │  → explode() + groupby() + idxmax()
    └──────────────────────────┘
                   │
                   ▼
              Analyzer
    ┌──────────────────────────┐
    │  Apriori algorithm       │  → 5,305 frequent itemsets
    │  association_rules()     │  → 29,562 raw rules
    │  Zhang's metric          │  → corrects lift's popularity bias
    │  filter_rules()          │  → 12,670 quality rules (57% removed)
    └──────────────────────────┘
                   │
              ┌────┴────┐
              ▼         ▼
        Recommender   Visualizer
        issubset()    4 Plotly charts
        cold start    heatmap + scatter
              │         │
              └────┬────┘
                   ▼
            Streamlit Dashboard
            3 pages — live at streamlit.app
```

---

## Key Engineering Decisions

### 1. Replacing iterrows() — 90× Performance Improvement

```python
# Original — 1,743,168 Python iterations (~45 seconds)
for index, row in df.iterrows():
    for i in range(1, 49):
        if i in row['basket_items']:
            df.loc[index, col_item] = 1

# Senior — vectorized C operations (~0.5 seconds)
encoder = TransactionEncoder()
matrix  = encoder.fit(transactions).transform(transactions)
```

### 2. Fixing the Off-by-One Bug in Item Frequency

```python
# Original — always returns wrong item (no error thrown)
poplr_item = [0] * 49
poplr_item = df_1.loc[[poplr_item.index(max(poplr_item)) - 1]]
#                                                          ^^^  BUG

# Senior — dictionary key IS the item_id, no arithmetic possible
item_counts = {}
for basket in baskets:
    for item_id in basket:
        item_counts[item_id] = item_counts.get(item_id, 0) + 1
```

### 3. Zhang's Metric — Correcting Lift's Popularity Bias

```
Zhang(A → B) = (P(A∩B) − P(A)·P(B)) /
               max(P(A∩B)·(1−P(A)), P(A)·(P(B)−P(A∩B)))

Range: +1 = perfect positive association
        0 = no association
       −1 = perfect negative association

Result: 16,892 popularity-biased rules removed
        12,670 genuine association rules retained
```

### 4. Precomputed Results for Cloud Deployment

Apriori on 36,316 transactions exceeds Streamlit Cloud's 1GB RAM limit.
Solution: run Apriori locally, commit results as CSV, load on cloud.

```python
# Cloud loads precomputed CSV instead of running Apriori
rules_df = pd.read_csv("data/raw/precomputed_rules.csv")
rules_df["antecedents"] = rules_df["antecedents"].apply(
    lambda x: frozenset(x.split("|"))
)
```

---

## Installation

```bash
git clone https://github.com/Khalida-DS/market_basket_analysis.git
cd market_basket_analysis

conda create -n market_basket python=3.11 -y
conda activate market_basket
pip install -r requirements.txt
```

---

## Usage

```bash
# Run full pipeline (terminal)
python main.py

# Launch interactive dashboard
streamlit run dashboard/app.py

# Run all tests
pytest tests/ -v
# 90 passed in ~2s
```

---

## Test Coverage

```
tests/test_data_loader.py    18 tests
tests/test_preprocessor.py   22 tests
tests/test_analyzer.py       18 tests
tests/test_recommender.py    16 tests
tests/test_visualizer.py     16 tests
─────────────────────────────────────
Total                        90 tests — all passing in 2.37s
```

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `customer_baskets.csv` | 36,316 | Transaction-level data, basket as comma-separated item IDs |
| `clothing_categories.csv` | 48 | Item ID → name → description mapping |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | pandas 2.x, numpy |
| ML | mlxtend (Apriori, TransactionEncoder) |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Logging | loguru |
| Testing | pytest (90 tests) |
| Deployment | Streamlit Cloud |
| Version Control | Git — feature branch per phase, PR per merge |

---

## Project Build Log

| Phase | Module | What Was Built |
|---|---|---|
| 1 | `data_loader.py` | CSV ingestion, schema validation, loguru logging |
| 2 | `preprocessor.py` | One-hot encoding, 2 bug fixes, vectorized transforms |
| 3 | `analyzer.py` | Apriori pipeline + Zhang's metric implementation |
| 4 | `recommender.py` | Rule-based recommendation engine + cold start |
| 5 | `visualizer.py` | 4 business-driven Plotly charts |
| 6 | `dashboard/app.py` | 3-page Streamlit dashboard + cloud deployment |

Each phase was developed on a feature branch, tested before merging,
and merged to `main` via pull request. 6 PRs total.

---

## What This Project Demonstrates

| Skill | Evidence |
|---|---|
| Performance engineering | 90× speedup replacing iterrows() |
| Bug identification | Silent off-by-one fixed with test proof |
| Statistical depth | Zhang's metric from formula, not just applied |
| Business thinking | 4 charts chosen to answer retailer questions |
| Software design | 5 modules, single responsibility, zero overlap |
| Test discipline | 90 tests, fixtures, all under 2.5 seconds |
| Git workflow | Feature branches, conventional commits, 6 PRs |
| Cloud deployment | Streamlit Cloud with precomputed data strategy |

---

## Author

**Khalida** — Data Scientist
[GitHub: Khalida-DS](https://github.com/Khalida-DS)

---

*Built February 2026 — Market Basket Analysis Capstone Project*
*Live at: [marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app](https://marketbasketanalysis-bqjt3cnnrxkp4cnvb2b2wj.streamlit.app)*
