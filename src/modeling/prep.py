import pandas as pd
from src import config
from sklearn.preprocessing import StandardScaler


def target_encode(train_df, test_df, cat_cols, target, m=10):
    """Encodes categorical columns based on the average target value (Smoothed)."""
    global_mean = train_df[target].mean()
    for col in cat_cols:
        # Calculate mean target per category on TRAIN set only
        agg = train_df.groupby(col)[target].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        # Smooth
        smooth_mean = (counts * means + m * global_mean) / (counts + m)
        mapping = smooth_mean.to_dict()

        # Apply to Train and Test
        train_df[f"{col}_te"] = train_df[col].map(mapping).fillna(global_mean)
        test_df[f"{col}_te"] = test_df[col].map(mapping).fillna(global_mean)

    return train_df, test_df


def split_and_prepare_data(df):
    print("- Preparing Training Data (Using ALL cases)...")
    target = 'remaining_time_days'

    # 1. Clean Target
    df = df.dropna(subset=[target]).copy()

    # 2. Define Feature Columns
    # Numeric: Counts, Time, Workload
    # REMOVED Month/Weekday from here
    num_cols = [c for c in df.columns if c.startswith('Count_') or c in [
        'elapsed_time_days', 'time_since_last_event', 'prefix_length',
        'judge_changed', 'judge_workload'
    ]]

    # Categorical: Last Event, Judge, AND Month/Weekday (for Target Encoding)
    # ADDED Month/Weekday here
    cat_cols = ['Last_event_ID', 'Second_last_event_ID', 'Month', 'Weekday'] + \
               [c for c in config.CASE_ATTRIBUTES if c in df.columns]

    # 3. Temporal Split
    cases = df.groupby(config.COL_CASE_ID)['case_start'].min().sort_values().index.tolist()
    split_idx = int(len(cases) * 0.8)  # 80/20 Split

    train_ids = cases[:split_idx]
    test_ids = cases[split_idx:]

    train_df = df[df[config.COL_CASE_ID].isin(train_ids)].copy()
    test_df = df[df[config.COL_CASE_ID].isin(test_ids)].copy()

    print(f"  - Train Cases: {len(train_ids)}")
    print(f"  - Test Cases:  {len(test_ids)}")

    # 4. Encoders (Target Encode Cats + Scale Numerics)
    train_df, test_df = target_encode(train_df, test_df, cat_cols, target)

    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols].fillna(0))
    test_df[num_cols] = scaler.transform(test_df[num_cols].fillna(0))

    # Return full dataframes
    return {
        "train_df": train_df,
        "test_df": test_df,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": num_cols + [f"{c}_te" for c in cat_cols]
    }