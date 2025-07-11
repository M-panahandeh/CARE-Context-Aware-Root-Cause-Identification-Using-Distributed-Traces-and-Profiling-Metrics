
##########Generating fault_injection files
import os
import pickle
import json

data_dir = r"D:\HeMiRCA\Newlabaled-final-timestamp"
output_directory = r"\HeMiRCA\InjectedFaults"
result = {"faults": []}

# Optional: set dataset-wide start time (e.g., minimum of all files)
global_timestamps = []

for filename in os.listdir(data_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            traces = pickle.load(f)

            print(f"\nLoaded {filename}")
            print("Type of traces:", type(traces))
            if isinstance(traces, list):
                print("First trace item type:", type(traces[0]))
                print("First trace item:", traces[0])


        # Assume each trace is a dictionary with a 'timestamp' field
        file_timestamps = traces['timestamp'].tolist()
        file_timestamps = [int(float(ts) // 1e6) for ts in traces['timestamp'].tolist()]


        if not file_timestamps:
            continue  # skip empty or malformed traces

        start_time = min(file_timestamps)
        end_time = max(file_timestamps)

        fault_entry = {
            "name": os.path.splitext(filename)[0],
            "start": start_time,
            "end": end_time
        }
        result["faults"].append(fault_entry)


# Save the result
output_path = os.path.join(output_directory, "injected_faults.json")
with open(output_path, 'w') as json_file:
    json.dump(result, json_file, indent=2)

print("Fault injection summary saved to:", output_path)