import json
import os
import pickle
import numpy as np
import pandas as pd
import psutil
import time

# Set your path here
data_dir = r"D:\HeMiRCA\Newlabaled-final-timestamp"
trace_info_path= r"D:\HeMiRCA\traces_info"
trace_lat_path = r"D:\HeMiRCA\lat"
trace_vector_path= r"D:\HeMiRCA\vector"

uninjection=r"D:\uninjection"

pkl_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
pkl_files_uninjection = [file for file in os.listdir(uninjection) if file.endswith(".pkl")]

unique_services = set()

# Traverse each file
for file in pkl_files_uninjection:
    with open(os.path.join(uninjection, file), 'rb') as f:
        traces = pickle.load(f)

    for trace in traces:
        if "s_t" in trace:
            for src, tgt in trace["s_t"]:
                unique_services.add(src)
                unique_services.add(tgt)

# Sort and assign unique integer IDs (starting from 1)
sorted_services = sorted(unique_services)
service_dict = {service: idx + 1 for idx, service in enumerate(sorted_services)}


# Print or save the dictionary
print("Generated service_dict:")
for k, v in service_dict.items():
    print(f"'{k}': {v},")


def get_trace_info_func(data_dir, output_path, pkl_files,service_dict,file_name):
    print("*** Extracting key features from your .pkl trace data...")

    trace_dict = {}

    for file in pkl_files:
        with open(os.path.join(data_dir, file), 'rb') as f:
            traces = pd.read_pickle(f)

        for _, row in traces.iterrows():
            trace_id = row["trace_id"]
            timestamp = int(float(row["timestamp"]) // 1e6)  # microseconds → seconds
            duration = float(row["latency"]) / 1000.0  # microseconds → seconds
            caller_name = row["source"]
            callee_name = row["target"]

            span_data = {
                "spanID": trace_id,
                "sendingTime": timestamp,
                "duration": duration,
                "parentSpanID": "",  # no hierarchy in your data
                "serviceName": callee_name,
                "callee": service_dict.get(callee_name, -1),
                "caller": service_dict.get(caller_name, -1)
            }

            if trace_id not in trace_dict:
                trace_dict[trace_id] = []

            trace_dict[trace_id].append(span_data)

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

def get_trace_info_func_normal(uninjection, output_path, pkl_files_uninjection, service_dict,filename):
    print("*** Extracting key features from your .pkl trace data...")

    trace_dict = {}

    for file in pkl_files_uninjection:
        with open(os.path.join(uninjection, file), 'rb') as f:
            traces = pickle.load(f)

        for row in traces:
            trace_id = row["trace_id"]
            num_spans = len(row["s_t"])
            for i in range(num_spans):
                timestamp = int(float(row["timestamp"][i]))  # already in seconds
                duration = float(row["latency"][i])  # already in seconds
                caller_name, callee_name = row["s_t"][i]

                span_data = {
                    "spanID": trace_id + f"_{i}",  # optional: unique spanID per span
                    "sendingTime": timestamp,
                    "duration": duration,
                    "parentSpanID": "",  # no hierarchy
                    "serviceName": callee_name,
                    "callee": service_dict.get(callee_name, -1),
                    "caller": service_dict.get(caller_name, -1)
                }

                if trace_id not in trace_dict:
                    trace_dict[trace_id] = []

                trace_dict[trace_id].append(span_data)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)

    with open(file_path, "w") as f:
        json.dump(trace_dict, f, indent=2)

    print(f"trace_info saved at: {file_path}")
    return trace_dict



if "__main__" == __name__:

    #for abnormal Microservices
    trace_info_dict= get_trace_info_func(data_dir, trace_info_path, pkl_files, service_dict,"trace_info_abnormal.json")
    # get trace_lat.json
    trace_lat_dic = get_trace_lat_func(trace_info_dict, trace_lat_path,"trace_lat_abnormal.json")
    # # get vector for VAE
    get_trace_vector(input_filepath=os.path.join(trace_lat_path, "trace_lat_abnormal.json"),
                     output_dir=trace_vector_path,
                     service_dict=service_dict, filename="trace_vector_abnormal.txt"
                     )


    #for normal data
    trace_info_dict = get_trace_info_func_normal(uninjection, trace_info_path, pkl_files_uninjection, service_dict,
                                          "trace_info_normal.json")
    # get trace_lat.json
    trace_lat_dic = get_trace_lat_func(trace_info_dict, trace_lat_path,"trace_lat_normal.json")
    # # get vector for VAE
    get_trace_vector(input_filepath=os.path.join(trace_lat_path, "trace_lat_normal.json"),
                     output_dir=trace_vector_path,
                     service_dict=service_dict, filename="trace_vector_normal.txt"
                     )









