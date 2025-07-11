import json
import pandas as pd
import os

# === FILE PATHS ===
injected_faults_path = r"D:\HeMiRCA\double\InjectedFaults\injected_faults.json"
correlation_result_path = r"./correlation_result_spearman_mse-double.csv"
output_csv_path = r"./output_mse_double-dt1.csv"

# === LOAD DATA ===
with open(injected_faults_path, "r") as f:
    injected_faults = json.load(f)

correlation_df = pd.read_csv(correlation_result_path)

aggregated_df = (
    correlation_df.groupby(['fault_index', 'microservice'])['correlation']
    .max()
    .reset_index()
)

results = []

for fault_index, fault in enumerate(injected_faults['faults'], start=1):
    fault_name = fault['name']
    # Extract expected service name
    expected_service = fault_name.split("_")[0].split("-")
    services = expected_service[0].split("+")

    # Handle root causes
    rootcause1 = "ts-" + services[0] + "-service"
    rootcause2 = "ts-" + services[1] + "-service" if len(services) > 1 else None

    # Filter by fault index
    fault_rows = aggregated_df[aggregated_df["fault_index"] == fault_index]
    # Sort by correlation descending
    fault_rows_sorted = fault_rows.sort_values(by="correlation", ascending=False).reset_index(drop=True)

    # Get the rank of the expected service
    # Initialize output variables
    rank1 = rank2 = None
    equal_score_count1 = equal_score_count2 = 0

    for idx, row in fault_rows_sorted.iterrows():
        ms = row["microservice"]
        corr = row["correlation"]

        if ms == rootcause1 and rank1 is None:
            rank1 = idx + 1
            equal_score_count1 = fault_rows_sorted[fault_rows_sorted["correlation"] == corr].shape[0]

        elif ms == rootcause2 and rank2 is None:
            rank2 = idx + 1
            equal_score_count2 = fault_rows_sorted[fault_rows_sorted["correlation"] == corr].shape[0]

        if rank1 is not None and (rank2 is not None or rootcause2 is None):
            break

    results.append({
            "fault_name": fault_name,
            "expected_microservice1": rootcause1,
            "expected_microservice2": rootcause2,
            "rank1": rank1,
            "rank2": rank2,
            "num_with_same_score1": equal_score_count1,
            "num_with_same_score2": equal_score_count2
        })

# === SAVE TO CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print("Done! Results saved to", output_csv_path)