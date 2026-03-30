import pandas as pd
from src import config


def add_temporal_features(df):
    """
    Adds time-based features: Weekday, Month, Elapsed Time.
    Calculates the target variable: remaining_time_days (ONLY FOR CLOSED CASES).
    """
    df = df.copy()
    print("  - Adding Temporal Features...")

    if config.COL_DATE in df.columns:
        df[config.COL_DATE] = pd.to_datetime(df[config.COL_DATE])

    # Sort to ensure correct time calculations
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    # 1. Cyclic/Calendar features
    df['Weekday'] = df[config.COL_DATE].dt.dayofweek
    df['Month'] = df[config.COL_DATE].dt.month

    # 2. Duration features
    df['case_start'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('min')
    df['elapsed_time_days'] = (df[config.COL_DATE] - df['case_start']).dt.total_seconds().div(86400)
    df['time_since_last_event'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].diff().dt.total_seconds().div(
        86400).fillna(0)

    print("  - IMPORTANT: Filtering for officially closed cases ('Extinct' or 'Canceled')...")

    # Search for the translations 'Extinct' or 'Canceled' in the status column
    closed_mask = df[config.COL_STATUS].astype(str).str.contains('Extinct|Canceled|closed', case=False, na=False,regex=True)
    closed_case_ids = df[closed_mask][config.COL_CASE_ID].unique()

    # Filter the dataframe to keep ONLY events belonging to closed cases
    df = df[df[config.COL_CASE_ID].isin(closed_case_ids)].copy()

    print(f"  - Kept {len(closed_case_ids)} closed cases for training.")

    # 3. Target Variable: Remaining Time
    # Now it is safe to use 'max' because we know the last event is truly the closure of the case
    df['case_end'] = df.groupby(config.COL_CASE_ID)[config.COL_DATE].transform('max')
    df['remaining_time_days'] = (df['case_end'] - df[config.COL_DATE]).dt.total_seconds().div(86400)

    return df


def add_control_flow_features(df, top_n_events=20):
    """
    Simplified Control Flow to AVOID LEAKAGE:
    Replaced Global K-Means (Leaky) with Event Frequencies (Safe).
    """
    print(f"  - Adding Control Flow Features (Top {top_n_events} events)...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    # 1. Immediate State (Last Event & Second Last)
    df['Last_event_ID'] = df[config.COL_ACTIVITY]
    df['Second_last_event_ID'] = df.groupby(config.COL_CASE_ID)[config.COL_ACTIVITY].shift(1)

    # 2. History (Prefix Length)
    df['prefix_length'] = df.groupby(config.COL_CASE_ID).cumcount() + 1

    # 3. Process Memory (Count of specific events so far)
    # We only take the top N most frequent events to avoid explosion
    event_counts = df[config.COL_ACTIVITY].value_counts()
    top_events = event_counts.head(top_n_events).index.tolist()

    # Create dummy columns for these events
    dummies = pd.get_dummies(df[config.COL_ACTIVITY])
    # Keep only top events, fill missing with 0
    dummies = dummies.reindex(columns=top_events, fill_value=0)
    dummies.columns = [f"Count_{col}" for col in dummies.columns]

    # Calculate cumulative sum per case (History so far)
    # This is SAFE because it uses cumsum()
    dummies = dummies.groupby(df[config.COL_CASE_ID]).cumsum()

    df = pd.concat([df, dummies], axis=1)

    return df


def add_judge_change_feature(df):
    """
    Detects true judge handovers by ignoring 'Unknown' gaps and calculates
    the cumulative number of handovers for the case.
    """
    if config.COL_RESOURCE not in df.columns:
        return df

    print("  - Adding Judge Change (Handover) Detection...")
    df = df.sort_values([config.COL_CASE_ID, config.COL_DATE])

    # 1. To avoid 'Unknown' triggering false handovers (e.g., A -> Unknown -> A),
    # we temporarily replace 'Unknown' with NaN and forward-fill within each case.
    # This tracks the true "active" judge over time.
    active_judge = df[config.COL_RESOURCE].replace('Unknown', pd.NA)
    active_judge = active_judge.groupby(df[config.COL_CASE_ID]).ffill()

    # 2. Shift the active judge by 1 to compare
    prev_active_judge = active_judge.groupby(df[config.COL_CASE_ID]).shift(1)

    # 3. A handover occurs when the active judge changes (and the previous active judge wasn't NaN)
    df['judge_changed'] = ((active_judge != prev_active_judge) & prev_active_judge.notna()).astype(int)

    # 4. Cumulative handovers (highly predictive of delays)
    df['Count_handovers'] = df.groupby(config.COL_CASE_ID)['judge_changed'].cumsum()

    return df