import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import math
import altair as alt
import yaml

st.title("Comfortness Tool - Prediction Dashboard")

# User-friendly labels for each target
TARGET_LABELS = {
    "temperature_c": "Indoor Temperature °C (Predicted)",
    "rh_percent": "Indoor Relative Humidity % (Predicted)",
    "luminance_lux": "Indoor Luminance (lux, Predicted)",
    "average_noise_db": "Indoor Noise (dB, Predicted)",
    "co2_ppm": "Indoor CO₂ (ppm, Predicted)",
    "pm10": "Indoor PM10 (μg/m³, Predicted)",
    "tvoc_ppb": "Indoor TVOC (ppb, Predicted)",
    "pm2_5": "Indoor PM2.5 (μg/m³, Predicted)"
}

tab1, tab2, tab3 = st.tabs(["Data Insertion", "Data", "Visualization"])

with tab1:
    uploaded_file = st.file_uploader("Upload input CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["time_end"])
        df['time_end'] = pd.to_datetime(df['time_end'], utc=True)
        df = df.set_index('time_end')
        st.write("Preview of uploaded data:", df.head())

        # Feature Engineering
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


    # Prediction
        perf_df = pd.read_csv("model_performance_summary.csv")
        with open("config.yml", "r") as f:
            config = yaml.safe_load(f)

        predictions = pd.DataFrame({'time_end': df.index})

        for target, target_cfg in config['models'].items():
            best_row = perf_df[perf_df['target'] == target].sort_values('R2', ascending=False).iloc[0]
            best_model = best_row['model']
            model_path = f"models/{target}_{best_model}.pkl"
            features_path = f"models/{target}_{best_model}_features.json"
            with open(features_path, 'r') as f:
                features = json.load(f)
            model = joblib.load(model_path)
            X_pred = df[features]
            y_pred = model.predict(X_pred)
            predictions[f'{target}_pred'] = np.round(y_pred, 1)
            st.success(f"Predicted {TARGET_LABELS.get(target, target)} using {best_model}")

        st.session_state['predictions'] = predictions.copy()

with tab2:
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        display_df = predictions.copy()
        # Rename columns for user-friendly labels
        for target, label in TARGET_LABELS.items():
            col = f"{target}_pred"
            if col in display_df.columns:
                display_df.rename(columns={col: label}, inplace=True)
        st.write("Predictions Table:")
        st.dataframe(display_df)

with tab3:
    if 'predictions' in st.session_state:
        predictions = st.session_state['predictions']
        st.write("Prediction Visualizations")
        for target, label in TARGET_LABELS.items():
            col = f"{target}_pred"
            if col in predictions.columns:
                chart = alt.Chart(predictions.reset_index()).mark_line().encode(
                    x=alt.X('time_end:T', title="Timestamp"),
                    y=alt.Y(col, title=label),
                    tooltip=['time_end', col]
                ).properties(
                    title=label
                ).interactive()
                st.altair_chart(chart, use_container_width=True)        