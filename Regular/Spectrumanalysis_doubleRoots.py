import json
import math
import os
import time

import numpy as np
import pandas as pd
from colorama import Back, Fore, Style, init
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pickle
from concurrent.futures import ProcessPoolExecutor
import openpyxl
import networkx as nx
# Set the interactive backend, e.g., 'TkAgg' or 'Qt5Agg'
import matplotlib
import matplotlib.pyplot as plt  # Import matplotlib.pyplot separately
#matplotlib.use('TkAgg')
import community

def SpectrumAnalysis (file):

    ################################################Read a file######################################################
    # Read the contents of each pickle file
    with open(file, 'rb') as file:
        data = pickle.load(file)
    tic = time.time()
    # Create a data structure to store the extracted information
    abnormal_traces = []
    normal_traces = []

    all_abnormal_trace_ids = data.loc[data['predict'] == 1.0, 'trace_id'].unique()
    # data['http_status'] = data['http_status'].astype(int)
    # all_abnormal_trace_ids = data.loc[data['http_status'].between(400, 599), 'trace_id'].unique()
    # Filter data where trace_id is in all_abnormal_trace_ids
    abnormal_traces_df = data[data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    abnormal_traces = abnormal_traces_df.to_dict(orient='records')
    # Filter data where trace_id is not in unique_trace_ids
    normal_traces_df = data[~data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    normal_traces = normal_traces_df.to_dict(orient='records')
    #################################################Step 1: Create call graphs nt and at####################################################
    # Create a dictionary to store the count for each service in  abnormal_traces
    O_ef = {}
    O_nf = {}

    # Initialize a set to keep track of unique trace_ids for the current node
    services = set()

    # Iterate over each data entry in abnormal_traces and normal_traces
    for data_entry in abnormal_traces + normal_traces:
        source = data_entry['source']
        target = data_entry['target']

        # Update the service counts for source and target
        if source not in services:
            services.add(source)
        if target not in services:
            services.add(target)
    for service in services:
        # Initialize a set to keep track of unique trace_ids for the current data_entry
        seen_trace_ids = set()
        trace_seen = {}
        # Iterate over each data entry in abnormal_traces
        for entry in abnormal_traces:
            source = entry['source']
            target = entry['target']
            trace_id = entry['trace_id']
            # Check if the source or target node matches the current service
            if (source == service or target == service) and trace_id not in seen_trace_ids:
                # Increment the count for the current node
                O_ef[service] = O_ef.get(service, 0) + 1
                # Add the trace_id to the set to avoid counting it multiple times for the same node
                seen_trace_ids.add(trace_id)
                trace_seen[trace_id] = True
            # for O_nf
            if ((source != service or target != service) and trace_id not in seen_trace_ids):
                trace_seen[trace_id] = False

        # Count the number of traces without the node for the current node
        O_nf[service] = sum(1 for trace_id, seen in trace_seen.items() if not seen)

    # Create a dictionary to store the count for service in normal_traces
    O_ep = {}
    O_np = {}
    for service in services:
        # Initialize a set to keep track of unique trace_ids for the current node
        seen_trace_ids = set()
        trace_seen = {}
        # Iterate over each data entry in abnormal_traces
        for entry in normal_traces:
            source = entry['source']
            target = entry['target']
            trace_id = entry['trace_id']

            # Check if the source or target node matches the current node in nt_graph
            if (source == service or target == service) and trace_id not in seen_trace_ids:
                # Increment the count for the current node
                O_ep[service] = O_ep.get(service, 0) + 1
                # Add the trace_id to the set to avoid counting it multiple times for the same node
                seen_trace_ids.add(trace_id)
                trace_seen[trace_id] = True
            # for O_np
            if ((source != service or target != service) and trace_id not in seen_trace_ids):
                trace_seen[trace_id] = False
        # Count the number of traces without the node for the current node
        O_np[service] = sum(1 for trace_id, seen in trace_seen.items() if not seen)

    ############################################ Step 2: Output ranked list############################################

    list_of_candiates = {}
    for service in services:

       try:
           # ochiai
           list_of_candiates[service] = (O_ef.get(service, 0)) / math.sqrt(
               (O_ef.get(service, 0) + O_ep.get(service, 0)) * (
                       O_ef.get(service, 0) + O_nf.get(service, 0)))

           # Dstar2
           # list_of_candiates[service] = (O_ef.get(service, 0) ** 2.0) / (
           #             O_ep.get(service, 0) + O_nf.get(service, 0))


       # RussellRao
       #        list_of_candiates[service] =O_ef.get(service, 0)/(O_ef.get(service, 0)+O_nf.get(service, 0)+O_ep.get(service, 0)+O_np.get(service, 0))

       # M2
       #           list_of_candiates[service] = O_ef.get(service, 0)/(O_ef.get(service, 0)+O_np.get(service, 0)+(2*O_ep.get(service, 0))+(2*O_nf.get(service, 0)))

       # list_of_candiates
       # list_of_candiates[service] = (O_ef.get(service, 0)) / math.sqrt(
       #     (O_ef.get(service, 0) + O_ep.get(service, 0)) * (
       #             O_ef.get(service, 0) + O_nf.get(service, 0)))
       except ZeroDivisionError:
           list_of_candiates[service] = 0


    # Sort services based on their Ochiai scores (high to low)
    sorted_list_of_candiates = sorted(list_of_candiates.items(), key=lambda x: x[1], reverse=True)
    # Iterate through the sorted_ochiai and assign ranks
    rank = 0
    prev_score = None
    ranked_list_of_candiates = {}

    for service, score in sorted_list_of_candiates:
        # If the score is same as the previous score, keep the same rank
        if prev_score == score:
            ranked_list_of_candiates[service] = (score, rank)
        else:
            rank += 1
            ranked_list_of_candiates[service] = (score, rank)
        prev_score = score

    init(autoreset=True)
    print(Back.GREEN + "Ranked list:\n")
    for service, (score, rank) in ranked_list_of_candiates.items():
        print(f"{rank}\t{service}\t{score}")

    toc = time.time()

    print("algo:", "ours", "time:", toc - tic)
    return ranked_list_of_candiates

def process_file(file):
        ranked_ochiai = SpectrumAnalysis(file)

        # define rootcause
        filename = os.path.basename(file)
        parts = filename.split('+')
        rootcause1 = "ts-" + parts[0] + "-service"
        rootcause2= "ts-"+ parts[1].split('_')[0] + "-service"

        # Create a DataFrame
        df = pd.DataFrame(columns=['Filename', 'Rank1', 'Score1', 'Number of Services with Equal Ranks1','Rank2', 'Score2', 'Number of Services with Equal Ranks2'])

        # get rank of root cause
        score1, rank1 = ranked_ochiai.get(rootcause1, (None, None))
        score2, rank2 = ranked_ochiai.get(rootcause2, (None, None))
        # services with the same rank
        services_with_same_rank1 = [service for service, (_, r) in ranked_ochiai.items() if
                                   r == rank1 and service != rootcause1]
        services_with_same_rank2 = [service for service, (_, r) in ranked_ochiai.items() if
                                    r == rank2 and service != rootcause2]

        return pd.DataFrame({
            'Filename': [filename],
            'Rank1': [rank1],
            'Score1':[score1],
            'Number of Services with Equal Ranks1': [len(services_with_same_rank1)],
            'Rank2': [rank2],
            'Score2': [score2],
            'Number of Services with Equal Ranks2': [len(services_with_same_rank2)]
        })


if __name__ == '__main__':
    directory = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\double_root-NewLabeld"
    # List all files in the directory
    all_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pkl')]
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Filename', 'Rank1', 'Score1', 'Number of Services with Equal Ranks1', 'Rank2', 'Score2',
                               'Number of Services with Equal Ranks2'])

    #  processing
    results = [process_file(file) for file in all_files]
    df = pd.concat(results, ignore_index=True)  # Combine the individual DataFrames
    df.to_excel('doubleroot-cause.xlsx', index=False)
