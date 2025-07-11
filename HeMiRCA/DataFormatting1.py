import os
import pickle
import pandas as pd
from collections import defaultdict

# Input and output directories
data_dir = r"D:\\HeMiRCA\Newlabaled-final-timestamp"
output_dir = r"D:\HeMiRCA\Metrics-generated"
os.makedirs(output_dir, exist_ok=True)

# Store per-source metrics
source_metrics = defaultdict(list)

# Loop over pickle files
for filename in os.listdir(data_dir):
    if filename.endswith(".pkl"):
        with open(os.path.join(data_dir, filename), "rb") as f:
            traces = pickle.load(f)

        for _, row in traces.iterrows():
            source = row.get("source", None)
            if not source:
                continue

            # timestamps = row.get("timestamp", [])
            timestamps = float(row.get("timestamp", [])) // 1e6
            if not isinstance(timestamps, list):
                timestamps = [timestamps]
            # Loop through each timestamp
            for ts in timestamps:
                record = {
                    "timestamp": ts,
                    "cpu_usage_system": row.get("cpu_use_system", None),
                    "cpu_usage_total": row.get("cpu_use", None),
                    "cpu_usage_user": row.get("cpu_use_user", None),
                    "memory_usage": row.get("mem_use_percent", None),
                    "memory_working_set": row.get("mem_use_amount", None),
                    "rx_bytes": row.get("net_receive_rate", None),
                    "tx_bytes": row.get("net_send_rate", None),
                    "latency": float(row["latency"])
                }



                source_metrics[source].append(record)

# Save per-source CSV
for source, records in source_metrics.items():
    df = pd.DataFrame(records)
    df.sort_values(by="timestamp", inplace=True)
    df.to_csv(os.path.join(output_dir, f"{source}.csv"), index=False)
    print(f"{source} file is generated.")

print(f"Done! Generated {len(source_metrics)} CSV files.")




