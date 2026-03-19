import pandas as pd
from src import config

def clean_data(df):
    """
    Generic data cleaning: Duplicates, Currency, Boolean, Dates, Artifacts.
    """
    df = df.copy()
    initial_rows = len(df)
    print("- Cleaning data...")

    # 1. Remove exact duplicates
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        print(f"  - Dropped {initial_rows - len(df)} duplicate rows.")

    # 2. Clean Currency (Generic)
    if 'claim_amount' in df.columns:
        df['claim_amount'] = (
            df['claim_amount']
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce').fillna(0)

    # 3. Clean Boolean (Generic)
    if 'digital' in df.columns:
        df['digital'] = (df['digital'].astype(str).str.upper().str.strip() == 'VERDADEIRO').astype(int)

    # 4. Clean Dates
    for col in config.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # 5. Filter Date Artifacts (Generic Year Range 1990-2030)
    if config.COL_DATE in df.columns:
        mask_valid = (df[config.COL_DATE].dt.year >= 1990) & (df[config.COL_DATE].dt.year <= 2030)
        invalid_count = (~mask_valid).sum()
        if invalid_count > 0:
            print(f"  - Dropped {invalid_count} events with invalid dates.")
            df = df[mask_valid]

    # 6. Sorting
    sort_cols = [c for c in [config.COL_CASE_ID, config.COL_DATE, 'order'] if c in df.columns]
    if len(sort_cols) >= 2:
        df = df.sort_values(by=sort_cols)

    return df

def cluster_activities_hybrid(df, top_n=20):
    """
    Hybrid approach: Keeps the exact labels for the top N most frequent activities,
    and clusters the remaining "long tail" of activities into broader phases.
    """
    print(f"- Applying Hybrid Activity Clustering (Keeping top {top_n} exact)...")
    df = df.copy()

    # 1. Find the top N most frequent activities across the dataset
    top_activities = df[config.COL_ACTIVITY].value_counts().nlargest(top_n).index.tolist()

    def assign_cluster(movement_str):
        if not isinstance(movement_str, str):
            return "Administrative/Other"

        # Rule A: If it's a top, high-frequency activity, KEEP its exact name
        if movement_str in top_activities:
            return movement_str

        # Rule B: If it's a rare activity, cluster it using our dictionary
        for cluster, keywords in config.ACTIVITY_CLUSTERS.items():
            if any(kw.lower() in movement_str.lower() for kw in keywords):
                return cluster

        # Rule C: Fallback for unmapped rare activities
        return "Administrative/Other"

    # Apply the mapping
    df['movement_cluster'] = df[config.COL_ACTIVITY].apply(assign_cluster)

    # Overwrite the original movement column so models use the hybrid labels
    df[config.COL_ACTIVITY] = df['movement_cluster']

    num_unique = df['movement_cluster'].nunique()
    print(f"  - Reduced activities to {num_unique} unique labels ({top_n} exact + {num_unique - top_n} clusters).")

    return df