"""
main.py — Pipeline Entry Point
================================
This is the single command to run the entire pipeline.

HOW TO RUN:
    cd market_basket_analysis
    conda activate market_basket
    python main.py
"""

from loguru import logger
from src.data_loader import DataLoader


def run_pipeline():
    """Execute the full Market Basket Analysis pipeline."""

    logger.info("🛒 Market Basket Analysis Pipeline — Starting")
    logger.info("Phase 1: Data Loading & Validation")

    # --------------------------------------------------
    # PHASE 1 — Load and validate data
    # --------------------------------------------------
    loader = DataLoader()
    baskets_df, categories_df = loader.load_all()

    # Quick summary
    logger.info(f"Transactions loaded : {len(baskets_df):,}")
    logger.info(f"Unique customers    : {baskets_df['customer_id'].nunique():,}")
    logger.info(f"Item categories     : {len(categories_df):,}")
    logger.info(f"Avg basket size     : {baskets_df['basket_size'].mean():.2f}")

    # --------------------------------------------------
    # PHASES 2–8 — Coming soon
    # --------------------------------------------------
    logger.info("Phases 2–8: Coming in next phases...")
    logger.info("✅ Pipeline complete")


if __name__ == "__main__":
    run_pipeline()