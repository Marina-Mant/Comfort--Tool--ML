import os
import math
import yaml
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# load the yaml config file
path_yaml = 'config.yml'
with open(path_yaml, 'r') as file:
    config = yaml.safe_load(file)

# Get path to data

data_path = Path(config.get('data_path'))
print(f"Data path: {data_path}")

# Load model / target configuration
tasks = config.get('models', {})
print(f"Model configuration: {tasks}")

# Helper: feature importance extractor
def get_feature_importance(model, feature_names):
    # Tree-based models, CatBoost
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()
    # Linear models
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)

# Load and preprocess the dataset
df = pd.read_csv(data_path, parse_dates=["time_end"])
df['time_end'] = pd.to_datetime(df['time_end'], utc=True)
df = df.set_index('time_end')

# Time-based features
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month
df['is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)

### TO ADAPT ###
# doy = df["time_end"].dt.dayofyear
# hod = df["time_end"].dt.hour + df["time_end"].dt.minute / 60
# if "doy_sin" in feats and "doy_sin" not in df.columns:
#     df["doy_sin"] = np.sin(2 * math.pi * doy / 365)
# if "doy_cos" in feats and "doy_cos" not in df.columns:
#     df["doy_cos"] = np.cos(2 * math.pi * doy / 365)
# if "hour_sin" in feats and "hour_sin" not in df.columns:
#     df["hour_sin"] = np.sin(2 * math.pi * hod / 24)
# if "hour_cos" in feats and "hour_cos" not in df.columns:
#     df["hour_cos"] = np.cos(2 * math.pi * hod / 24)
################################


# Cyclical time features
doy = df.index.dayofyear
hod = df.index.hour + df.index.minute / 60

df["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
df["hour_sin"] = np.sin(2 * math.pi * hod / 24)
df["hour_cos"] = np.cos(2 * math.pi * hod / 24)

# Edit: ignoring the empty columns instead of dropping them
df = df.loc[:, df.notna().any()]

# model inventory
model_inv = {  # NEEDS COMPLETION # EDIT: done😎
    'RF': RandomForestRegressor(),
    'GBM': GradientBoostingRegressor(),
    'CatBoost': cb.CatBoostRegressor(verbose=0),
    # 'SVR_poly': SVR(kernel='poly'),
    'Lasso': Lasso(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': lgb.LGBMRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'KNN': KNeighborsRegressor(),
    'DT': DecisionTreeRegressor(),
#    'GPR': GaussianProcessRegressor(),
}

# Define tasks: each target with its features and models
tasks = {
    'temperature_c': (
        [#'direct_radiation', 'shortwave_radiation', 'precipitation',
         'temperature_outdoor_c', 'rh_outdoor_percent', #'wind_speed_10m', 'wind_gusts_10m',
         'hour', 'weekday', 'month' #'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
         ],
        {
            'RF',
            'GBM',
            'CatBoost',
            # 'SVR_poly': SVR(kernel='poly'),
        #    'Lasso',
            'XGBoost',
            'LightGBM',
            'AdaBoost',
            
        }
    ),
    'rh_percent': (
        ['rh_outdoor_percent', # 'direct_radiation', 'shortwave_radiation', 
        # 'hour', 'weekday',
        'month', 'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'],
        {
            'RF',
            # 'SVR': SVR(),
            'Lasso',
            'CatBoost',
        }
    ),
    'luminance_lux': (
        ['shortwave_radiation', 'direct_radiation', #'rh_outdoor_percent', 
        # 'cloud_cover', 
        # 'hour',
        #'weekday',
        'month',
        'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'],
        {
            'RF',
            'GBM',
            'XGBoost',
            'CatBoost'
        }
    ),
    'average_noise_db': (
        ['rain', 'wind_gusts_10m', 'wind_speed_10m', 
         'hour',
        #'weekday',
        'month', 'is_weekend',
        'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'],
        {
            'RF',
            'LightGBM',
            'KNN',
            'GBM',
            'CatBoost',
        }
    ),

# Pollutants' proper insertion
    'co2_ppm': (
['temperature_outdoor_c',
 # 'wind_speed_10m', 'wind_gusts_10m',
 'rh_outdoor_percent',
'hour',
'weekday',
'month',
#'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
],

{'RF',
 'DT',
 # 'SVR': SVR(),
 'GBM',
# 'GPR',
 'CatBoost'}
),

    'tvoc_ppb': (
['temperature_outdoor_c',
#'wind_speed_10m', 'wind_gusts_10m',
'rh_outdoor_percent',
#'hour',
'weekday',
'month',
#'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
],
{'RF',
    'DT',
    # 'SVR': SVR(),
    'GBM',
    #'GPR',
    'CatBoost'}
    ),
    
        'pm2_5': (
    ['temperature_outdoor_c', 'wind_speed_10m', 'wind_gusts_10m', 'rh_outdoor_percent',
    'hour', 'weekday', 'month'
    #'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
    ],
    {'RF',
    'DT',
    # 'SVR': SVR(),
    'GBM',
    #'GPR',
    'CatBoost'}
    ),
    
        'pm10': (
    ['temperature_outdoor_c', 'wind_speed_10m', 'wind_gusts_10m', 'rh_outdoor_percent',
      'hour', 'weekday', 'month'
    # , 'doy_sin', 'doy_cos', 'hour_sin', 'hour_cos'
      ],
    {'RF',
    'DT',
    # 'SVR': SVR(),
    'GBM',
 #   'GPR',
    'CatBoost'}
    )

}


# Define hyperparameter grids for GridSearchCV
grid_params = {
    'RF': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
    'GBM': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'LightGBM': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'CatBoost': {'iterations': [100, 200], 'depth': [4, 6], 'learning_rate': [0.03, 0.1]},
    # 'SVR': {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
    # 'SVR_poly': {'C': [0.1, 1], 'degree': [2, 3]},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'AdaBoost': {'n_estimators': [50, 100]},
    'Lasso': {'alpha': [0.1, 1.0]},
    'DT': {'max_depth': [None, 5, 10]},
    'GBR': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'GPR': {}
}

# Prepare results container and output
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)  # folder to save trained models
results = []
fi_records = []  # to collect feature importances and output
os.makedirs('plots', exist_ok=True)
results = []
fi_records = []  # to collect feature importances and output
results = []

# Loop through each target and its models
print("Starting full forecasting run...")
for target, (features, models) in tasks.items():
    print(f"=== TARGET: {target} ===")
for target, (features, models) in tasks.items():
    # Drop NA for target
    df_task = df.dropna(subset=[target]).copy()
    # Engineer weekend flag only for noise
    if target == 'average_noise_db':
        df_task['is_weekend'] = df_task.index.weekday.isin([5, 6]).astype(int)

    X = df_task[features]
    y = df_task[target]
    # Time-based train/test split (80/20)
    split_idx = int(len(df_task) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


    # Scale features for Lasso and KNN
    scale_models = {'Lasso', 'KNN'}
    if any(name in scale_models for name in models):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


    for name in models:
        model = model_inv[name]
        print(f"-- Model: {name} --")
        # 1) Grid Search
        if name in grid_params and grid_params[name]:
            print(f"[{target}] Running GridSearchCV for {name}...")
            gs = GridSearchCV(model, grid_params[name], cv=3, scoring='neg_root_mean_squared_error')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            print(f"Best params for {name}: {best_params}")
        else:
            best_model = model
            best_params = {}
            # 2) Cross-Validation RMSE on training
        # print(f"Performing 5-fold CV for {name}...")
        # cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        # cv_rmse = -cv_scores.mean()
        # print(f"CV RMSE for {name}: {cv_rmse:.4f}")
        # 3) Fit on full train and predict on test
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Save the trained model to disk
        model_path = os.path.join('models', f"{target}_{name}.pkl")
        joblib.dump(best_model, model_path)

        #   NEW!
        # Save features used for this model
        features_path = os.path.join('models', f"{target}_{name}_features.json")
        with open(features_path, 'w') as f:
            json.dump(features, f)
        print(f"Model {name} trained and saved to {model_path}")


        # Collect feature importances
        fi = get_feature_importance(best_model, features)
        if fi is not None:
             for feat, val in fi.items():
                 fi_records.append({
                     'target': target,
                     'model': name,
                     'feature': feat,
                     'importance': val
                 })
        print(f"Finished training and prediction for {name}.")
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Record results
        results.append({
            'target': target,
            'model': name,
            'best_params': best_params,
            'CV_RMSE': np.nan,  #cv_rmse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        # Plot actual vs predicted
        plt.figure(figsize=(8,4))
        plt.plot(y_test.index, y_test, label='Actual')
        plt.plot(y_test.index, y_pred, '--', label='Predicted')
        plt.title(f"{target} - {name}")
        plt.xlabel('Time')
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{target}_{name}.png")
        plt.close()

# Create and save summary DataFrame
print("Saving model performance summary...")
path_to_model_perform = 'model_performance_summary.csv'
results_df = pd.DataFrame(results)
results_df.to_csv(path_to_model_perform, index=False)
print(f"Model performance summary saved to {path_to_model_perform}")
print(results_df)

# Build and save feature-importance DataFrame
path_to_importance_csv = 'feature_importances.csv'
fi_df = pd.DataFrame(fi_records)
fi_df.to_csv(path_to_importance_csv, index=False)
print(f"Feature importances saved to {path_to_importance_csv}")














