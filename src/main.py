import sys
import pandas as pd
import config
from preprocessing import loader, cleaner, translator
from feature_engineering import transformers, workload
from analysis import visualizer, stats
from modeling import prep, train


def run_preprocessing():
    print("\n Cleaning & Translation")
    p1_path = config.DATA_PROCESSED_DIR / config.TRANSLATED_FILENAME
    df = loader.load_data()
    df = cleaner.clean_data(df)
    if hasattr(cleaner, 'remove_outliers'):
        df = cleaner.remove_outliers(df)
    df = translator.translate_data(df)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p1_path, index=False)
    return df


def run_feature_engineering(df_phase1):
    print("\n Feature Engineering")
    p2_path = config.DATA_PROCESSED_DIR / config.FEATURED_FILENAME
    df_feat = transformers.add_temporal_features(df_phase1)
    df_feat = transformers.add_control_flow_features(df_feat)
    df_feat = transformers.add_judge_change_feature(df_feat)
    df_feat = workload.add_inter_case_features(df_feat)

    df_feat.to_csv(p2_path, index=False)
    return df_feat


def run_modeling(df_feat):
    print("\n Predictive Modeling")
    data_dict = prep.split_and_prepare_data(df_feat)
    results_df, best_model, X_test, y_test = train.run_experiment(data_dict)
    print(results_df)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.MODEL_RESULTS_FILE, index=False)
    if best_model and X_test is not None:
        visualizer.run_model_plots(best_model, X_test, y_test)


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