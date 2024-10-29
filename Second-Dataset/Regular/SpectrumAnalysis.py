import json
import math
import os
import time
import re
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
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def SpectrumAnalysis (file):

    ################################################Read a file######################################################
    # Read the contents of each pickle file
    data = pd.read_csv(file)
    tic = time.time()
    # Create a data structure to store the extracted information
    abnormal_traces = []
    normal_traces = []
    # Find unique trace_id where predict is 1.0
    predict_trace_ids = data.loc[data['predict'] == 1.0, 'trace_id'].unique()
    # error_data = data.loc[data['succ'] == False, 'trace_id'].unique()
    # Find unique trace_id where label is False
    # label_trace_ids = data.loc[data['succ'] == 0, 'trace_id'].unique()
    # Merge unique trace_ids from both conditions
    # all_abnormal_trace_ids = np.unique(np.concatenate((predict_trace_ids, error_data)))
    all_abnormal_trace_ids=predict_trace_ids
    # Filter data where trace_id is in all_abnormal_trace_ids
    abnormal_traces_df = data[data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    abnormal_traces = abnormal_traces_df.to_dict(orient='records')

    # Filter data where trace_id is not in unique_trace_ids
    normal_traces_df = data[~data['trace_id'].isin(all_abnormal_trace_ids)]
    # convert to a list of dictionaries:
    normal_traces = normal_traces_df.to_dict(orient='records')

    #print(abnormal_traces,normal_traces)
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

    Ochiai = {}
    for service in services:
       try:
           #RussellRao
            # Ochiai[service] =O_ef.get(service, 0)/(O_ef.get(service, 0)+O_nf.get(service, 0)+O_ep.get(service, 0)+O_np.get(service, 0))
           #Dstar2
           # Ochiai[service] =(O_ef.get(service, 0)^2)/(O_ep.get(service, 0)+O_nf.get(service, 0))

           #M2
            # Ochiai[service] = O_ef.get(service, 0)/(O_ef.get(service, 0)+O_np.get(service, 0)+(2*O_ep.get(service, 0))+(2*O_nf.get(service, 0)))

           #Ochiai
            Ochiai[service] = (O_ef.get(service, 0)) / math.sqrt(
                (O_ef.get(service, 0) + O_ep.get(service, 0)) * (
                        O_ef.get(service, 0) + O_nf.get(service, 0)))
       except ZeroDivisionError:
           Ochiai[service] = 0

    # Sort services based on their Ochiai scores (high to low)
    sorted_ochiai = sorted(Ochiai.items(), key=lambda x: x[1], reverse=True)
    # Iterate through the sorted_ochiai and assign ranks
    rank = 0
    prev_score = None
    ranked_ochiai = {}

    for service, score in sorted_ochiai:
        # If the score is same as the previous score, keep the same rank
        if prev_score == score:
            ranked_ochiai[service] = (score, rank)
        else:
            rank += 1
            ranked_ochiai[service] = (score, rank)
        prev_score = score

    init(autoreset=True)
    print(Back.GREEN + "Ranked list:\n")
    for service, (score, rank) in ranked_ochiai.items():
        print(f"{rank}\t{service}\t{score}")
    toc = time.time()

    print("algo:", "ours", "time:", toc - tic)
    return ranked_ochiai

def process_file(file):
        ranked_ochiai = SpectrumAnalysis(file)
        # define rootcause
        filename = os.path.basename(file)
        filename=  os.path.splitext(filename)[0]
        match = re.match(r'\d+(.*)', filename)
        rootcause = match.group(1)

        # Create a DataFrame
        df = pd.DataFrame(columns=['Filename', 'Rank','Score', 'Number of Services with Equal Ranks'])

        # get rank of root cause
        score, rank = ranked_ochiai.get(rootcause, (None, None))
        # services with the same rank
        services_with_same_rank = [service for service, (_, r) in ranked_ochiai.items() if
                                   r == rank and service != rootcause]

        return pd.DataFrame({
            'Filename': [filename],
            'Rank': [rank],
            'Score':[score],
            'Number of Services with Equal Ranks': [len(services_with_same_rank)]
        })


if __name__ == '__main__':

    directory = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\B\PreparedData\final_Newlabeled-alpha"
    # List all files in the directory
    all_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.csv') and os.path.isfile(os.path.join(directory, filename))]
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Filename', 'Rank','Score', 'Number of Services with Equal Ranks'])

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, all_files))

    for result in results:
        df = pd.concat([df, result], ignore_index=True)

    # analysis of the results in excel output file
    top1 = (df['Rank'] == 1).sum()
    top3 = (df['Rank'] <= 3).sum()
    top5 = (df['Rank'] <= 5).sum()
    exam_score = df['Rank'].mean()
    same_pos = df['Number of Services with Equal Ranks'].mean()
    # Prepare results
    results = {
            "Metric": ["Top 1", "Top 3", "Top 5", "Exam Score", "same-Position"],
            "root1": [top1/25, top3/25, top5/25, exam_score, same_pos],
        }
    results_df = pd.DataFrame(results)
    print(results_df)
    df.to_excel('noise-30.xlsx', index=False)







