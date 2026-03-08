import pandas as pd
from src import config


def add_inter_case_features(df):
    """
    Calculates the 'Workload': Number of active cases a judge has at the specific moment of the event.
    """
    print("- Calculating Judge Workload...")

    if config.COL_RESOURCE not in df.columns:
        return df

    df = df.sort_values(by=[config.COL_DATE])

    # 1. Define Case Lifespan (Start to End)
    case_spans = df.groupby(config.COL_CASE_ID)[config.COL_DATE].agg(['min', 'max']).reset_index()
    case_spans.columns = [config.COL_CASE_ID, 'start_date', 'end_date']

    # 2. Map Cases to Judges
    # We assume the judge assigned at the first event is the judge for the case
    judge_map = df.drop_duplicates(config.COL_CASE_ID, keep='first')[[config.COL_CASE_ID, config.COL_RESOURCE]]
    active_cases = case_spans.merge(judge_map, on=config.COL_CASE_ID)

    # 3. Create a Timeline of +1 (New Case) and -1 (Case Closed)
    # Event: Case Starts
    starts = active_cases[[config.COL_RESOURCE, 'start_date']].copy()
    starts.columns = [config.COL_RESOURCE, config.COL_DATE]
    starts['change'] = 1

    # Event: Case Ends
    ends = active_cases[[config.COL_RESOURCE, 'end_date']].copy()
    ends.columns = [config.COL_RESOURCE, config.COL_DATE]
    # Add a second so the drop happens AFTER the event
    ends[config.COL_DATE] = ends[config.COL_DATE] + pd.Timedelta(seconds=1)
    ends['change'] = -1

    # 4. Calculate Running Total (Workload)
    timeline = pd.concat([starts, ends]).sort_values(config.COL_DATE)
    timeline['judge_workload'] = timeline.groupby(config.COL_RESOURCE)['change'].cumsum()

    # 5. Merge Workload back onto the main Event Log
    # merge_asof finds the closest past value in the timeline for each event in df
    df = pd.merge_asof(
        df.sort_values(config.COL_DATE),
        timeline[[config.COL_DATE, config.COL_RESOURCE, 'judge_workload']],
        on=config.COL_DATE,
        by=config.COL_RESOURCE,
        direction='backward'
    )

    df['judge_workload'] = df['judge_workload'].fillna(0).astype(int)

    return df