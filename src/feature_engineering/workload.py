import pandas as pd
from src import config


def add_inter_case_features(df):
    """
    Calculates the 'Workload': Number of active cases a judge has,
    and refines it by subject matter to capture complexity.
    """
    print("- Calculating Refined Judge Workload...")

    if config.COL_RESOURCE not in df.columns or 'subject_matter' not in df.columns:
        return df

    df['subject_matter'] = df['subject_matter'].fillna('Unknown')
    df = df.dropna(subset=[config.COL_DATE]).sort_values(by=[config.COL_DATE])

    # 1. Define Case Lifespan
    case_info = df.groupby(config.COL_CASE_ID).agg(
        start_date=(config.COL_DATE, 'min'),
        end_date=(config.COL_DATE, 'max'),
        judge=(config.COL_RESOURCE, 'first'),
        subject_matter=('subject_matter', 'first')
    ).reset_index()

    case_info.rename(columns={'judge': config.COL_RESOURCE}, inplace=True)

    # 2. Create Timeline
    starts = case_info[[config.COL_RESOURCE, 'subject_matter', 'start_date']].copy()
    starts.columns = [config.COL_RESOURCE, 'subject_matter', config.COL_DATE]
    starts['change'] = 1

    ends = case_info[[config.COL_RESOURCE, 'subject_matter', 'end_date']].copy()
    ends.columns = [config.COL_RESOURCE, 'subject_matter', config.COL_DATE]
    ends[config.COL_DATE] = ends[config.COL_DATE] + pd.Timedelta(seconds=1)
    ends['change'] = -1

    timeline = pd.concat([starts, ends]).sort_values(config.COL_DATE)

    # 3. Calculate Running Totals
    timeline['judge_workload'] = timeline.groupby(config.COL_RESOURCE)['change'].cumsum()
    timeline['workload_by_subject'] = timeline.groupby([config.COL_RESOURCE, 'subject_matter'])['change'].cumsum()

    # 4. Merge back to main dataframe safely
    # Explicitly sort after dropping duplicates to satisfy merge_asof constraints
    global_timeline = timeline[[config.COL_DATE, config.COL_RESOURCE, 'judge_workload']].drop_duplicates(
        subset=[config.COL_DATE, config.COL_RESOURCE], keep='last'
    ).sort_values(config.COL_DATE)

    subject_timeline = timeline[
        [config.COL_DATE, config.COL_RESOURCE, 'subject_matter', 'workload_by_subject']].drop_duplicates(
        subset=[config.COL_DATE, config.COL_RESOURCE, 'subject_matter'], keep='last'
    ).sort_values(config.COL_DATE)

    df = pd.merge_asof(
        df,
        global_timeline,
        on=config.COL_DATE,
        by=config.COL_RESOURCE,
        direction='backward'
    )

    df = pd.merge_asof(
        df,
        subject_timeline,
        on=config.COL_DATE,
        by=[config.COL_RESOURCE, 'subject_matter'],
        direction='backward'
    )

    df['judge_workload'] = df['judge_workload'].fillna(0).astype(int)
    df['workload_by_subject'] = df['workload_by_subject'].fillna(0).astype(int)

    return df