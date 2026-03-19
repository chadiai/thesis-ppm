import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from src import config


def train_model(model_type, X_train, y_train):
    """
    Fits the specified model type (RF or XGB).
    """
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'xgb':
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def run_experiment(data_dict):
    """
    Runs the 8-Scenario Ablation Study for BOTH Random Forest and XGBoost.
    """
    train_df = data_dict['train_df']
    test_df = data_dict['test_df']
    all_cols = data_dict['feature_names']
    target = 'remaining_time_days'

    y_train = train_df[target]
    y_test = test_df[target]

    # --- Feature Groups ---
    # 1. Baseline: Case Attributes
    f_attrs = [c for c in all_cols if any(attr in c for attr in config.CASE_ATTRIBUTES) and '_te' in c]

    # 2. Control Flow: Counts (Frequency)
    f_counts = [c for c in all_cols if c.startswith('Count_')]

    # 3. Control Flow: State (Last Event + Second Last)
    f_state = ['Last_event_ID_te']
    f_last_two = ['Last_event_ID_te', 'Second_last_event_ID_te']

    # 4. Temporal
    # Added prefix_length to give context to elapsed_time
    # Uses Month_te and Weekday_te (Target Encoded) instead of raw
    f_temporal = ['elapsed_time_days', 'time_since_last_event', 'prefix_length', 'Month_te', 'Weekday_te']

    # 5. Workload
    f_workload = [c for c in all_cols if 'workload' in c]

    # Helper to ensure cols exist
    def get_cols(col_list):
        return [c for c in col_list if c in train_df.columns]

    scenarios = {
        "Case Attributes (Baseline)": get_cols(f_attrs),
        "Control Flow: Events (Freq + State)": get_cols(f_attrs + f_counts + f_state),
        "Control Flow: Frequency Only": get_cols(f_attrs + f_counts),
        "Control Flow: Last Two": get_cols(f_attrs + f_last_two),
        "Full Control Flow": get_cols(f_attrs + f_counts + f_last_two),
        "Temporal Features": get_cols(f_attrs + f_temporal),
        "Workload Features": get_cols(f_attrs + f_workload),
        "All Features": all_cols
    }

    results = []
    best_model, min_mae = None, float('inf')
    X_test_best = None

    print("\n" + "=" * 60)
    print(" RUNNING 8 SCENARIOS (RF vs XGB)")
    print("=" * 60)

    for model_name in ['rf', 'xgb']:
        print(f"\n--- Model: {model_name.upper()} ---")

        for scenario_name, features in scenarios.items():
            if not features:
                continue

            # Train
            X_train_sub = train_df[features]
            X_test_sub = test_df[features]

            model = train_model(model_name, X_train_sub, y_train)
            y_pred = model.predict(X_test_sub)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)

            print(f"  {scenario_name:40s} | MAE: {mae:.2f} days")

            results.append({
                "Model": model_name.upper(),
                "Scenario": scenario_name,
                "MAE": mae,
                "Num_Features": len(features)
            })

            if mae < min_mae:
                min_mae = mae
                best_model = model
                X_test_best = test_df.copy()
                X_test_best['predicted_remaining'] = y_pred
                print(f"\t-> New Global Best Model!")

    print("=" * 60)
    return pd.DataFrame(results), best_model, X_test_best, y_test