import json
import pandas as pd
import os
import re

# === FILE PATHS ===
injected_faults_path = r"D:\HeMiRCA\InjectedFaults\injected_faults.json"
correlation_result_path = r"./correlation_result_spearman_mae.csv"
output_csv_path = r"./output_mae_dt2.csv"

# === LOAD DATA ===
with open(injected_faults_path, "r") as f:
    injected_faults = json.load(f)

correlation_df = pd.read_csv(correlation_result_path)

# === Normalize microservice scores ===
# Keep the max correlation score for each (fault_index, microservice)
aggregated_df = (
    correlation_df.groupby(['fault_index', 'microservice'])['correlation']
    .max()
    .reset_index()
)

results = []

for fault_index, fault in enumerate(injected_faults['faults'], start=1):
    fault_name = fault['name']
    # Extract expected service name
    match = re.match(r"\d+(.*)", fault_name)
    if match:
        expected_service = match.group(1)# e.g. docker_007
    else:
        expected_service=""

    # Filter by fault index
    fault_rows = aggregated_df[aggregated_df["fault_index"] == fault_index]
    # Sort by correlation descending
    fault_rows_sorted = fault_rows.sort_values(by="correlation", ascending=False).reset_index(drop=True)

    # Get the rank of the expected service
    rank = None
    equal_score_count = 0

    for idx, row in fault_rows_sorted.iterrows():
        if row["microservice"] == expected_service:
            rank = idx + 1  # Rank starts at 1
            target_score = row["correlation"]
            if pd.isna(target_score):
                equal_score_count = fault_rows_sorted["correlation"].isna().sum()
            else:
                equal_score_count = (fault_rows_sorted["correlation"] == target_score).sum()
            break

    results.append({
        "fault_name": fault_name,
        "expected_microservice": expected_service,
        "rank": rank,
        "num_with_same_score": equal_score_count
    })

# === SAVE TO CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print("Done! Results saved to", output_csv_path)