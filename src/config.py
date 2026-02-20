"""
config.py — Single Source of Truth for All Project Settings
============================================================
WHY THIS FILE EXISTS:
In the original notebook, values like range(1,49), min_support=0.01,
and file paths were scattered across the entire script.
A config.py solves this: every setting lives in ONE place.
"""

from pathlib import Path

# ===========================================================
# PROJECT PATHS
# ===========================================================
# Path(__file__) = location of THIS file → src/config.py
# .parent        = src/ folder
# .parent        = project root folder
# Works correctly no matter where you run the script from.

PROJECT_ROOT        = Path(__file__).parent.parent
DATA_RAW_DIR        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
BASKETS_FILE        = DATA_RAW_DIR / "customer_baskets.csv"
CATEGORIES_FILE     = DATA_RAW_DIR / "clothing_categories.csv"
OUTPUTS_DIR         = PROJECT_ROOT / "outputs"
RULES_DIR           = OUTPUTS_DIR / "rules"
FIGURES_DIR         = OUTPUTS_DIR / "figures"
DB_PATH             = RULES_DIR / "association_rules.db"

# ===========================================================
# DATA SCHEMA
# ===========================================================
# Defines expected columns so data_loader.py can validate
# incoming data. If the CSV changes structure, we catch it
# immediately with a clear error.

BASKETS_SCHEMA = {
    "required_columns" : ["customer_id", "basket"],
    "basket_col"       : "basket",
}

CATEGORIES_SCHEMA = {
    "required_columns" : ["category_id", "name", "description"],
}

# ===========================================================
# ITEM CATALOG
# ===========================================================
# Original code had range(1, 49) hardcoded in 4 places.
# If catalog grows to 60 items you'd miss 12 silently.
# Now there is ONE place to update.

ITEM_ID_MIN = 1
ITEM_ID_MAX = 48

# ===========================================================
# APRIORI PARAMETERS
# ===========================================================
# min_support=0.01  → item combo must appear in 1%+ of transactions
# min_confidence=0.6 → given A, B must follow 60%+ of the time
# min_lift=1.0      → A and B appear together MORE than by chance
# min_zhang=0.0     → A genuinely INCREASES probability of B

APRIORI_MIN_SUPPORT    = 0.01
APRIORI_MIN_CONFIDENCE = 0.60
APRIORI_MIN_LIFT       = 1.0
APRIORI_MIN_ZHANG      = 0.0
APRIORI_METRIC         = "lift"
APRIORI_MIN_THRESHOLD  = 0.1

# ===========================================================
# RECOMMENDATION ENGINE
# ===========================================================

RECOMMENDER_TOP_N          = 5
RECOMMENDER_MIN_CONFIDENCE = 0.6

# ===========================================================
# LOGGING
# ===========================================================
# loguru gives: timestamp + level + file + function + line
# print() gives: nothing except the text

LOG_LEVEL  = "INFO"
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# ===========================================================
# VISUALIZATION
# ===========================================================
# Centralized colors = every chart looks consistent

VIZ_PRIMARY_COLOR   = "#2E86AB"
VIZ_SECONDARY_COLOR = "#A23B72"
VIZ_ACCENT_COLOR    = "#F18F01"
VIZ_NEUTRAL_COLOR   = "#C73E1D"
VIZ_BACKGROUND      = "#F8F9FA"
VIZ_FIGURE_WIDTH    = 1000
VIZ_FIGURE_HEIGHT   = 500