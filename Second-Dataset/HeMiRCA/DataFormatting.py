import os
import pickle
import pandas as pd
from collections import defaultdict

# Input and output directories
data_dir = r"D:\final_Newlabeled-alpha"
output_dir = r"D:\Metrics-generated"
os.makedirs(output_dir, exist_ok=True)

# Store per-source metrics
source_metrics = defaultdict(list)

# Loop over CSV files
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            source = row.get("source")
            if pd.isna(source):
                continue

            # Convert timestamp to UNIX float
            try:
                ts = pd.to_datetime(row["timestamp"]).timestamp()
            except Exception as e:
                continue  # skip malformed timestamps

            record = {
                "timestamp": ts,
                "latency": float(row["latency"])
            }

            source_metrics[source].append(record)
    print(filename)

# Save per-source CSV
for source, records in source_metrics.items():
    print(source)
    df = pd.DataFrame(records)
    df.sort_values(by="timestamp", inplace=True)
    df.to_csv(os.path.join(output_dir, f"{source}.csv"), index=False)
    print(f"{source} file is generated.")

print(f"Done! Generated {len(source_metrics)} CSV files.")

