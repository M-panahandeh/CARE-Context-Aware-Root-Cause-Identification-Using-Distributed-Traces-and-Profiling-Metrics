#########Generating fault_injection files
import os
import pickle
import json

data_dir = r"D:\final_Newlabeled-alpha"
output_directory = r"\InjectedFaults"
os.makedirs(output_directory, exist_ok=True)

result = {"faults": []}

# Optional: set dataset-wide start time (e.g., minimum of all files)
global_timestamps = []

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            continue

        if "timestamp" not in df.columns or df["timestamp"].isnull().all():
            print(f"Skipping {filename} — no valid timestamp data.")
            continue

        # Convert to UNIX timestamp (in seconds)
        try:
            timestamps = pd.to_datetime(df["timestamp"], errors='coerce').dropna().map(lambda x: int(x.timestamp()))
        except Exception as e:
            print(f"Error parsing timestamps in {filename}: {e}")
            continue

        if timestamps.empty:
            continue

        start_time = int(timestamps.min())
        end_time = int(timestamps.max())

        fault_entry = {
            "name": os.path.splitext(filename)[0],
            "start": start_time,
            "end": end_time
        }
        result["faults"].append(fault_entry)
        print(f"{filename}: start={start_time}, end={end_time}")

# Save the result
output_path = os.path.join(output_directory, "injected_faults.json")
with open(output_path, 'w') as json_file:
    json.dump(result, json_file, indent=2)

print("Fault injection summary saved to:", output_path)
