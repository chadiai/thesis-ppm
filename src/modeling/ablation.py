import pandas as pd
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src import config


def run_workload_ablation(data_dict):
    print("\n" + "=" * 50)
    print("STARTING WORKLOAD ABLATION STUDY (Targeted 18-Feature XGBoost)")
    print("=" * 50)

    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    target_col = 'remaining_time_days'

    # EXACTLY the 18 features from your winning "Workload Features" scenario
    winning_features = [
        'order', 'digital', 'claim_amount', 'Weekday', 'Month',
        'judge_changed', 'movement_te', 'status_te', 'area_te',
        'subject_matter_te', 'control_te', 'class_te',
        'distribution_date_te', 'court_department_te', 'judge_te',
        'judge_type_te', 'judge_workload', 'workload_by_subject'
    ]

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # Isolate ONLY the 18 winning features
    X_train = train_df[winning_features]
    X_test = test_df[winning_features]

    # UPDATE THESE if your train.py uses different XGBoost parameters!
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42,
        'n_jobs': -1
    }

    # 1. Calculate TRUE Baseline (The 18-Feature Model)
    print("Training winning Workload XGBoost model...")
    baseline_model = xgb.XGBRegressor(**xgb_params)
    baseline_model.fit(X_train, y_train)

    baseline_preds = baseline_model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    print(f"Baseline MAE (18 Features): {baseline_mae:.2f} days\n")

    # 2. Ablate the workload features
    workload_features = ['judge_workload', 'workload_by_subject']
    ablation_results = {'Baseline (Workload Scenario)': baseline_mae}

    for feature in workload_features:
        print(f"Ablating (removing) feature: {feature}...")

        X_train_ablated = X_train.drop(columns=[feature])
        X_test_ablated = X_test.drop(columns=[feature])

        ablated_model = xgb.XGBRegressor(**xgb_params)
        ablated_model.fit(X_train_ablated, y_train)

        ablated_preds = ablated_model.predict(X_test_ablated)
        mae = mean_absolute_error(y_test, ablated_preds)

        penalty = mae - baseline_mae
        ablation_results[f"Without {feature}"] = mae

        print(f"-> Result MAE: {mae:.2f} days | Penalty: +{penalty:.2f} days\n")

    # 3. Print Summary
    print("-" * 50)
    print("ABLATION STUDY SUMMARY:")
    for scenario, mae in ablation_results.items():
        if scenario == 'Baseline (Workload Scenario)':
            print(f"{scenario}: {mae:.2f} days")
        else:
            diff = mae - baseline_mae
            print(f"{scenario}: {mae:.2f} days (Error increased by {diff:.2f} days)")
    print("-" * 50)

    # 4. Save Results
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.REPORTS_DIR / f"{config.NAME}_ablation_results.csv"

    ablation_df = pd.DataFrame(list(ablation_results.items()), columns=['Scenario', 'MAE'])
    ablation_df.to_csv(save_path, index=False)

    return ablation_df