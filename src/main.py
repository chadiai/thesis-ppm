import argparse
import pathlib

import pandas as pd
from src import config
from preprocessing import loader, cleaner, translator
from feature_engineering import transformers, workload
from analysis import visualizer, stats
from modeling import prep, train, dl_train


def run_preprocessing():
    print("\n Cleaning & Translation")
    df = loader.load_data()
    df = cleaner.clean_data(df)
    df = translator.translate_data(df)
    df = cleaner.cluster_activities_hybrid(df, top_n=20)
    return df


def run_feature_engineering(df_phase1):
    print("\n Feature Engineering")
    # Calculating workload on ALL cases before dropping active ones
    df_feat = workload.add_inter_case_features(df_phase1)

    # Now calculate target variable (which filters out open cases)
    df_feat = transformers.add_temporal_features(df_feat)
    df_feat = transformers.add_control_flow_features(df_feat)
    df_feat = transformers.add_judge_change_feature(df_feat)
    return df_feat


def run_modeling(df_feat):
    print("\n Predictive Modeling (Tabular)")
    data_dict = prep.split_and_prepare_data(df_feat)

    # 1. Run Tabular Experiments (RF / XGB)
    tabular_results, best_model, X_test, y_test = train.run_experiment(data_dict)

    # 2. Run DL Experiments (LSTM)
    dl_results = dl_train.run_experiment(df_feat, data_dict)

    # 3. Combine results
    results_df = pd.concat([tabular_results, dl_results], ignore_index=True)

    print("\n=== Final Results Table ===")
    print(results_df.tail(10))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.MODEL_RESULTS_FILE, index=False)

    if best_model and X_test is not None:
        visualizer.run_model_plots(best_model, X_test, y_test)
        visualizer.plot_error_by_workload_severity(X_test, y_test)

    visualizer.plot_thesis_feature_progression()
    visualizer.plot_thesis_final_showdown(results_df)

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
    parser = argparse.ArgumentParser(description="Run the Judicial Workload PPM pipeline.")

    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help="The full or relative path to the CSV dataset (e.g., C:/data/my_dataset.csv)"
    )

    args = parser.parse_args()

    # Resolve the path exactly
    custom_path = pathlib.Path(args.filepath).resolve()
    print(f"[*] Target dataset: {custom_path}")

    # 1. Update the base file parameters
    config.DATASET_PATH = custom_path
    config.RAW_FILENAME = custom_path.name
    print("update",config.RAW_FILENAME)
    config.NAME = custom_path.stem
    config.MODEL_RESULTS_FILE = config.REPORTS_DIR / f"{config.NAME}_results.csv"
    config.TRANSLATED_FILENAME = f"{config.NAME}_translated.csv"
    config.FEATURED_FILENAME = f"{config.NAME}_features.csv"

    # Run the pipeline
    run_pipeline()