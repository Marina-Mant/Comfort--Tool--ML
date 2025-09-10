import os
import math
import json
import joblib
import yaml
import numpy as np
import pandas as pd

# CONFIGURATION
CONFIG_PATH = 'config.yml'
PERF_CSV = 'model_performance_summary.csv'
INPUT_CSV = r'C:\Users\manteniotim\Documents\projects\airtech_data\airtech_data\certh_house_v01__ext_prediction.csv'
OUTPUT_CSV = 'predictions_all_targets.csv'

# LOAD CONFIG & PERFORMANCE SUMMARY
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
perf_df = pd.read_csv(PERF_CSV)

# LOAD DATA
df = pd.read_csv(INPUT_CSV, parse_dates=["time_end"])
df['time_end'] = pd.to_datetime(df['time_end'], utc=True)
df = df.set_index('time_end')

# FEATURE ENGINEERING
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month
df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)
doy = df.index.dayofyear
hod = df.index.hour + df.index.minute / 60
df["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
df["hour_sin"] = np.sin(2 * math.pi * hod / 24)
df["hour_cos"] = np.cos(2 * math.pi * hod / 24)



# PREDICT FOR EACH TARGET
predictions = pd.DataFrame({'time_end': df.index})


for target, target_cfg in config['models'].items():
    print(f"Predicting for target: {target} ...")
    best_row = perf_df[perf_df['target'] == target].sort_values('R2', ascending=False).iloc[0]
    best_model = best_row['model']
    model_path = f"models/{target}_{best_model}.pkl"
    features_path = f"models/{target}_{best_model}_features.json"
    with open(features_path, 'r') as f:
        features = json.load(f)
    model = joblib.load(model_path)
    X_pred = df[features]
    y_pred = model.predict(X_pred)
    predictions[f'{target}_pred'] = y_pred
    print(f"Predicted {target} using {best_model}")

    # Load model
    model = joblib.load(model_path)
    # Predict
    X_pred = df[features]
    y_pred = model.predict(X_pred)
    predictions[f'{target}_pred'] = y_pred
    print(f"Predicted {target} using {best_model}")

# Format loop for pred targets to 1 decimal point
for col in predictions.columns:
    if col.endswith('_pred'):
        predictions[col] = predictions[col].round(1)


# SAVE ALL PREDICTIONS
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions for all targets saved to {OUTPUT_CSV}")