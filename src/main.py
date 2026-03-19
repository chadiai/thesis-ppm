import pandas as pd
import config
from preprocessing import loader, cleaner, translator
from feature_engineering import transformers, workload
from analysis import visualizer, stats
from modeling import prep, train, dl_prep, dl_train


def run_preprocessing():
    print("\n Cleaning & Translation")
    p1_path = config.DATA_PROCESSED_DIR / config.TRANSLATED_FILENAME
    df = loader.load_data()
    df = cleaner.clean_data(df)
    df = translator.translate_data(df)
    df = cleaner.cluster_activities_hybrid(df, top_n=20)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p1_path, index=False)
    return df


def run_feature_engineering(df_phase1):
    print("\n Feature Engineering")
    p2_path = config.DATA_PROCESSED_DIR / config.FEATURED_FILENAME
    # Calculating workload on ALL cases before dropping active ones
    df_feat = workload.add_inter_case_features(df_phase1)

    # Now calculate target variable (which filters out open cases)
    df_feat = transformers.add_temporal_features(df_feat)
    df_feat = transformers.add_control_flow_features(df_feat)
    df_feat = transformers.add_judge_change_feature(df_feat)

    df_feat.to_csv(p2_path, index=False)
    return df_feat


def run_modeling(df_feat):
    print("\n Predictive Modeling (Tabular)")
    data_dict = prep.split_and_prepare_data(df_feat)

    results_df, best_model, X_test, y_test = train.run_experiment(data_dict)

    print("\n Predictive Modeling (Deep Learning - LSTM with Embeddings)")

    # Reconstruct raw train/test split (mirroring tabular exactly to ensure 1-to-1 comparison)
    train_ids = data_dict['train_df']['lawsuit_id'].unique()
    test_ids = data_dict['test_df']['lawsuit_id'].unique()

    raw_train_df = df_feat[df_feat['lawsuit_id'].isin(train_ids)].dropna(subset=['remaining_time_days']).copy()
    raw_test_df = df_feat[df_feat['lawsuit_id'].isin(test_ids)].dropna(subset=['remaining_time_days']).copy()

    # Base Case Attributes (Always included)
    base_cat = [c for c in config.CASE_ATTRIBUTES if c != 'claim_amount' and c in df_feat.columns]
    base_cont = ['claim_amount'] if 'claim_amount' in df_feat.columns else []

    # Define the 5 LSTM Scenarios
    lstm_scenarios = {
        "Case Attributes (Baseline)": {
            "cat": base_cat,
            "cont": base_cont
        },
        "Control Flow (Sequence)": {
            "cat": base_cat + ['movement'],
            "cont": base_cont
        },
        "Temporal Features": {
            "cat": base_cat + ['Weekday', 'Month'],
            "cont": base_cont + ['elapsed_time_days', 'time_since_last_event']
        },
        "Workload Features": {
            "cat": base_cat,
            "cont": base_cont + ['judge_workload', 'workload_by_subject']
        },
        "All Features": {
            "cat": base_cat + ['movement', 'Weekday', 'Month'],
            "cont": base_cont + ['elapsed_time_days', 'time_since_last_event', 'judge_workload', 'workload_by_subject']
        }
    }

    print("\n" + "=" * 60)
    print(" RUNNING LSTM SCENARIOS")
    print("=" * 60)

    for scenario_name, cols in lstm_scenarios.items():
        cat_cols = [c for c in cols["cat"] if c in df_feat.columns]
        cont_cols = [c for c in cols["cont"] if c in df_feat.columns]

        print(f"\n--- Model: LSTM | Scenario: {scenario_name} ---")

        train_loader, test_loader, embedding_sizes = dl_prep.prepare_dl_data(
            raw_train_df,
            raw_test_df,
            cat_cols=cat_cols,
            cont_cols=cont_cols,
            batch_size=64
        )

        lstm_mae = dl_train.train_and_evaluate_lstm(
            train_loader,
            test_loader,
            embedding_sizes=embedding_sizes,
            num_continuous=len(cont_cols),
            epochs=25
        )

        lstm_row = pd.DataFrame([{
            "Model": "LSTM",
            "Scenario": scenario_name,
            "MAE": lstm_mae,
            "Num_Features": len(cat_cols) + len(cont_cols)
        }])
        results_df = pd.concat([results_df, lstm_row], ignore_index=True)

    print("\n=== Final Results Table ===")
    print(results_df.tail(10))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.MODEL_RESULTS_FILE, index=False)

    if best_model and X_test is not None:
        visualizer.run_model_plots(best_model, X_test, y_test)
        visualizer.plot_error_by_workload_severity(X_test, y_test)


def run_pipeline():
    print("=== PIPELINE START ===")
    df = run_preprocessing()
    stats.print_stats(stats.get_process_stats(df))
    df_feat = run_feature_engineering(df)
    stats.print_stats(stats.get_process_stats(df_feat))
    visualizer.run_eda_plots(df_feat)
    run_modeling(df_feat)
    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    run_pipeline()