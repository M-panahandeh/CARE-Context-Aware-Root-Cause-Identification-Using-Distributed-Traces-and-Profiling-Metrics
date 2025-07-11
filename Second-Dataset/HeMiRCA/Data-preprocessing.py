import json
import os
import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import random

# Set your path here
data_dir = r"D:\final_Newlabeled-alpha"
trace_info_path= r"D:\traces_info"
trace_lat_path = r"D:\HeMiRCA\lat"
trace_vector_path= r"D:\HeMiRCA\vector"

uninjection=r"D:\HeMiRCA\uninjection"

def get_trace_info_func(data_dir, output_path, csv_files,service_dict,file_name):
    print("*** Extracting key features from your .csv abnormla trace data...")

    trace_dict = {}

    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            trace_id = str(row["trace_id"])

            # Basic fields
            timestamp = pd.to_datetime(row["timestamp"]).timestamp()
            duration = float(row["latency"])
            caller_name = row["source"]
            callee_name = row["target"]

            # Skip rows with missing source/target
            if pd.isna(caller_name) or pd.isna(callee_name):
                continue

            span_id = f"{trace_id}_{idx}"  # unique span ID

            span_data = {
                "spanID": span_id,
                "sendingTime": int(timestamp),
                "duration": duration,
                "parentSpanID": "",  # no hierarchy info in CSV
                "serviceName": callee_name,
                "callee": service_dict.get(callee_name, -1),
                "caller": service_dict.get(caller_name, -1)
            }

            if trace_id not in trace_dict:
                trace_dict[trace_id] = []

            trace_dict[trace_id].append(span_data)
        print(file+"Done!")

        # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, file_name)

    with open(file_path, "w") as f:
        json.dump(trace_dict, f, indent=2)

    print(f"trace_info saved at: {file_path}")
    return trace_dict


def get_trace_lat_func(trace_dict, output_dir, filename):
    print("*** trace latency computation...")

    trace_lat_dic = {}

    for key in trace_dict.keys():  # for each trace_id
        for span in trace_dict[key]:
            timestamp = span["sendingTime"]
            callee = str(span["callee"])
            duration = span["duration"]

            if timestamp not in trace_lat_dic:
                trace_lat_dic[timestamp] = {}

            if callee not in trace_lat_dic[timestamp]:
                trace_lat_dic[timestamp][callee] = []

            trace_lat_dic[timestamp][callee].append(duration)

    # Clean up empty entries (just in case)
    trace_lat_dic = {k: v for k, v in trace_lat_dic.items() if v}

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'w') as json_file:
        json.dump(trace_lat_dic, json_file, indent=2)

    print(f"trace_lat saved at: {file_path}")
    return trace_lat_dic

def get_trace_vector(input_filepath, output_dir, service_dict, filename):
    print("*** Trace vectorization...")

    with open(input_filepath, "r") as f:
        traces = json.load(f)

    num_services = len(service_dict)
    service_ids = list(range(1, num_services + 1))

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as fo:
        for timestamp in sorted(traces.keys()):
            timestamp = str(timestamp)
            vector = np.zeros(num_services)

            for callee in traces[timestamp].keys():
                callee_id = int(callee)
                latencies = traces[timestamp][callee]
                avg_latency = sum(latencies) / len(latencies)
                vector[callee_id - 1] = avg_latency

            # Convert vector to string
            vector_line = timestamp + ":" + ",".join(
                [str(int(x)) if x == 0 else str(round(x, 5)) for x in vector]
            )
            fo.write(vector_line + "\n")

    print(f"Trace vector written to: {file_path}")

def process_chunk_worker(args):
    file_path, chunk_start, chunk_size, service_dict, chunk_index, file_id, temp_dir = args
    result = defaultdict(list)

    try:
        # Read only the chunk rows needed
        skiprows = lambda x: x > 0 and (x < chunk_start or x >= chunk_start + chunk_size)
        chunk = pd.read_csv(file_path, skiprows=skiprows, nrows=chunk_size, dtype={
            "trace_id": str,
            "timestamp": str,
            "latency": float,
            "succ": str,
            "source": str,
            "target": str
        }, low_memory=True)

        chunk.dropna(subset=["trace_id", "timestamp", "latency", "source", "target"], inplace=True)

        # sample
        chunk = chunk.sample(frac=1, random_state=chunk_index)

        chunk["sendingTime"] = pd.to_datetime(chunk["timestamp"], errors='coerce')
        chunk.dropna(subset=["sendingTime"], inplace=True)
        chunk["sendingTime"] = (chunk["sendingTime"].view("int64") // 10**9)

        for row in chunk.itertuples(index=False):
            trace_id = row.trace_id
            span_id = f"{trace_id}_{file_id}_{chunk_index}"

            span_data = {
                "spanID": span_id,
                "sendingTime": int(row.sendingTime),
                "duration": float(row.latency),
                "parentSpanID": "",
                "serviceName": row.target,
                "callee": service_dict.get(row.target, -1),
                "caller": service_dict.get(row.source, -1)
            }
            result[trace_id].append(span_data)

        # Save this chunk to disk
        partial_path = os.path.join(temp_dir, f"partial_{file_id}_{chunk_index}.json")
        with open(partial_path, "w") as f:
            json.dump(result, f)

        print(f"Finished chunk {chunk_index} of file {file_id}")
        return partial_path

    except Exception as e:
        print(f"Error in chunk {chunk_index} of file {file_id}: {e}")
        return None


def get_trace_info_parallel(uninjection, output_path, csv_files_uninjection, service_dict, filename):
    import tempfile
    import gc
    temp_dir = tempfile.mkdtemp()

    chunk_size = 100_000
    num_workers = min(cpu_count(), 6)

    all_partial_files = []

    for file_id, file in enumerate(csv_files_uninjection):
        file_path = os.path.join(uninjection, file)
        print(f"Processing file: {file}")

        # Count total rows (fast estimate)
        with open(file_path, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # exclude header

        # Generate chunk offsets
        chunk_offsets = [(i, chunk_size) for i in range(1, total_rows + 1, chunk_size)]  # start row for each chunk (+1 for header skip)

        tasks = [(file_path, start, size, service_dict, idx, file_id, temp_dir)
                 for idx, (start, size) in enumerate(chunk_offsets)]

        with Pool(processes=num_workers) as pool:
            partial_files = pool.map(process_chunk_worker, tasks)
            all_partial_files.extend([pf for pf in partial_files if pf])

    print("Merging partial results...")
    # Merge partial JSON files
    trace_dict = defaultdict(list)
    for pf in all_partial_files:
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                partial_result = json.load(f)
                for trace_id, spans in partial_result.items():
                    trace_dict[trace_id].extend(spans)
        except MemoryError:
            print(f" MemoryError while loading: {pf}. Skipping this file.")
        except Exception as e:
            print(f" Error reading {pf}: {e}")
        finally:
            try:
                os.remove(pf)
            except:
                pass
            gc.collect()

    os.rmdir(temp_dir)

    # Save final output
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, filename)
    with open(output_file, "w") as f:
        json.dump(trace_dict, f, indent=2)

    print(f"trace_info saved at: {output_file}")
    return trace_dict


if "__main__" == __name__:


    # Collect all .pkl files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    csv_files_uninjection = [f for f in os.listdir(uninjection) if
                             f.endswith(".csv") and not f.startswith("~$") and not f.startswith("._")]
    # Set to collect unique service names
    unique_services = set()

    # Read each large file in chunks
    for file in csv_files_uninjection:
        file_path = os.path.join(uninjection, file)
        print(f"Processing: {file_path}")

        # Use chunks to avoid full memory load
        chunk_size = 100_000  # Adjust as needed
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Skip missing source/target early
                chunk = chunk.dropna(subset=["source", "target"])

                # Add to set
                unique_services.update(chunk["source"].astype(str).unique())
                unique_services.update(chunk["target"].astype(str).unique())
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Filter out any remaining non-string values just in case
    unique_services = {s for s in unique_services if isinstance(s, str)}

    # Sort and assign unique integer IDs
    sorted_services = sorted(unique_services)
    service_dict = {service: idx + 1 for idx, service in enumerate(sorted_services)}

    # Output
    print("Generated service_dict:")
    for k, v in service_dict.items():
        print(f"'{k}': {v},")



    #for abnormal Microservices
    trace_info_dict= get_trace_info_func(data_dir, trace_info_path, csv_files, service_dict,"trace_info_abnormal.json")
    # get trace_lat.json
    trace_lat_dic = get_trace_lat_func(trace_info_dict, trace_lat_path,"trace_lat_abnormal.json")
    # # get vector for VAE
    get_trace_vector(input_filepath=os.path.join(trace_lat_path, "trace_lat_abnormal.json"),
                     output_dir=trace_vector_path,
                     service_dict=service_dict, filename="trace_vector_abnormal.txt"
                     )



    #for normal data
    trace_info_dict = get_trace_info_parallel(uninjection, trace_info_path, csv_files_uninjection, service_dict,
                                          "trace_info_normal.json")
    # get trace_lat.json
    trace_lat_dic = get_trace_lat_func(trace_info_dict, trace_lat_path,"trace_lat_normal.json")
    # # get vector for VAE
    get_trace_vector(input_filepath=os.path.join(trace_lat_path, "trace_lat_normal.json"),
                     output_dir=trace_vector_path,
                     service_dict=service_dict, filename="trace_vector_normal.txt"
                     )









