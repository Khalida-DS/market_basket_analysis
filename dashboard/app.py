import streamlit as st
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

try:
    from src.data_loader import DataLoader
    from src.preprocessor import Preprocessor
    from src.analyzer import Analyzer

    loader = DataLoader()
    baskets_df, categories_df = loader.load_all()
    st.write("✅ Data loaded")

    preprocessor = Preprocessor(baskets_df, categories_df)
    one_hot_df = preprocessor.build_one_hot_matrix()
    st.write(f"✅ One-hot matrix: {one_hot_df.shape}")

    st.write("⏳ Running Apriori — this takes ~8 seconds...")
    
    from mlxtend.frequent_patterns import apriori
    frequent_itemsets = apriori(
        one_hot_df, 
        min_support=0.01, 
        use_colnames=True
    )
    st.write(f"✅ Frequent itemsets: {len(frequent_itemsets):,}")

except Exception as e:
    import traceback
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())

st.stop()
