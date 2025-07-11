import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


SCORE_FILE = r"D:\abnormal_trace_scores-mae.txt"
METRICS_DIR = r"D:\HeMiRCA\Metrics-generated"
FAULT_JSON = r"D:\injected_faults.json"
OUTPUT_FILE = "./correlation_result_spearman_mae.csv"

# Load anomaly scores with timestamp
def load_anomaly_scores(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values(by="timestamp")
    return df

# Load fault injection intervals
def load_fault_intervals(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    faults = []
    for i, fault in enumerate(data['faults']):
        start = int(fault['start']) - 100
        end = int(fault['end'])
        faults.append((i+1, start, end))
    return faults

# Load and normalize metric file
def load_and_prepare_metrics(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='timestamp')
    cols_to_scale = ["latency"]
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

# Correlation calculation per fault window
def calculate_spearman_correlation(score_df, metric_df, service_name, fault_windows):
    results = []
    score_df['timestamp'] = score_df['timestamp'].astype(int)
    metric_df['timestamp'] = metric_df['timestamp'].astype(int)

    merged = pd.merge(score_df, metric_df, on="timestamp", how="inner")
    for fault_index, start, end in fault_windows:
        window_df = merged[(merged['timestamp'] >= start) & (merged['timestamp'] <= end)]
        for col in metric_df.columns:
            if col in ['timestamp']: continue
            try:
                corr, _ = spearmanr(window_df['score'], window_df[col])
            except:
                corr = 0.0
            results.append({
                'fault_index': fault_index,
                'microservice': service_name,
                'metric': col,
                'correlation': abs(corr)
            })
    print(results)
    return results


def main():

    score_df = load_anomaly_scores(SCORE_FILE)
    fault_windows = load_fault_intervals(FAULT_JSON)
    all_results = []

    for file in os.listdir(METRICS_DIR):
        if file.endswith(".csv"):
            service_path = os.path.join(METRICS_DIR, file)
            service_name = file[:-4]
            print(f"Processing {service_name}...")
            metric_df = load_and_prepare_metrics(service_path)
            service_results = calculate_spearman_correlation(score_df, metric_df, service_name, fault_windows)
            all_results.extend(service_results)


    result_df = pd.DataFrame(all_results)
    result_df.sort_values(by=["fault_index", "correlation"], ascending=[True, False], inplace=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Correlation results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
