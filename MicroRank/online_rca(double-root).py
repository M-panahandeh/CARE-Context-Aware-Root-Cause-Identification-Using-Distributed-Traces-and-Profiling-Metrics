import numpy as np
import math
import json
import os
import sys
import datetime

import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor

from process_data import get_operation_duration_data
#from process_data import get_span
from process_data import get_operation_slo
from process_data import get_service_operation_list
from process_data import get_pagerank_graph
from pagerank import trace_pagerank

import time
import datetime
from dateutil.parser import parse
import json
import csv
import codecs

def calculate_spectrum_without_delay_list(anomaly_result, normal_result, anomaly_list_len, normal_list_len,
                                          top_max, normal_num_list, anomaly_num_list, spectrum_method):
    spectrum = {}

    for node in anomaly_result:

        spectrum[node] = {}
        # spectrum[node]['ef'] = anomaly_result[node] * anomaly_list_len
        # spectrum[node]['nf'] = anomaly_list_len - anomaly_result[node] * anomaly_list_len
        spectrum[node]['ef'] = anomaly_result[node] * anomaly_num_list[node]
        spectrum[node]['nf'] = anomaly_result[node] * \
            (anomaly_list_len - anomaly_num_list[node])
        if node in normal_result:
            #spectrum[node]['ep'] = normal_result[node] * normal_list_len
            #spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = normal_result[node] * normal_num_list[node]
            spectrum[node]['np'] = normal_result[node] * \
                (normal_list_len - normal_num_list[node])
        else:
            spectrum[node]['ep'] = 0.0000001
            spectrum[node]['np'] = 0.0000001

    for node in normal_result:

        if node not in spectrum:
            spectrum[node] = {}
            #spectrum[node]['ep'] = normal_result[node] * normal_list_len
            #spectrum[node]['np'] = normal_list_len - normal_result[node] * normal_list_len
            spectrum[node]['ep'] = (
                1 + normal_result[node]) * normal_num_list[node]
            spectrum[node]['np'] = normal_list_len - normal_num_list[node]
            if node not in anomaly_result:
                spectrum[node]['ef'] = 0.0000001
                spectrum[node]['nf'] = 0.0000001

    # print('\n Micro Rank Spectrum raw:')
    # print(json.dumps(spectrum))
    result = {}

    for node in spectrum:
        # Dstar2
        if spectrum_method == "dstar2":
            result[node] = spectrum[node]['ef'] * spectrum[node]['ef'] / \
                (spectrum[node]['ep'] + spectrum[node]['nf'])
        # Ochiai
        elif spectrum_method == "ochiai":
            result[node] = spectrum[node]['ef'] / \
                math.sqrt((spectrum[node]['ep'] + spectrum[node]['ef']) * (
                    spectrum[node]['ef'] + spectrum[node]['nf']))

        elif spectrum_method == "jaccard":
            result[node] = spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['ep']
                                                   + spectrum[node]['nf'])

        elif spectrum_method == "sorensendice":
            result[node] = 2 * spectrum[node]['ef'] / \
                (2 * spectrum[node]['ef'] + spectrum[node]
                 ['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "m1":
            result[node] = (spectrum[node]['ef'] + spectrum[node]
                            ['np']) / (spectrum[node]['ep'] + spectrum[node]['nf'])

        elif spectrum_method == "m2":
            result[node] = spectrum[node]['ef'] / (2 * spectrum[node]['ep'] + 2 * spectrum[node]['nf'] +
                                                   spectrum[node]['ef'] + spectrum[node]['np'])
        elif spectrum_method == "goodman":
            result[node] = (2 * spectrum[node]['ef'] - spectrum[node]['nf'] - spectrum[node]['ep']) / \
                (2 * spectrum[node]['ef'] + spectrum[node]
                 ['nf'] + spectrum[node]['ep'])
        # Tarantula
        elif spectrum_method == "tarantula":
            result[node] = spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf']) / \
                (spectrum[node]['ef'] / (spectrum[node]['ef'] + spectrum[node]['nf']) +
                 spectrum[node]['ep'] / (spectrum[node]['ep'] + spectrum[node]['np']))
        # RussellRao
        elif spectrum_method == "russellrao":
            result[node] = spectrum[node]['ef'] / \
                (spectrum[node]['ef'] + spectrum[node]['nf'] +
                 spectrum[node]['ep'] + spectrum[node]['np'])

        # Hamann
        elif spectrum_method == "hamann":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np'] - spectrum[node]['ep'] - spectrum[node]
                            ['nf']) / (spectrum[node]['ef'] + spectrum[node]['nf'] + spectrum[node]['ep'] + spectrum[node]['np'])

        # Dice
        elif spectrum_method == "dice":
            result[node] = 2 * spectrum[node]['ef'] / \
                (spectrum[node]['ef'] + spectrum[node]
                 ['nf'] + spectrum[node]['ep'])

        # SimpleMatching
        elif spectrum_method == "simplematcing":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (spectrum[node]
                                                                            ['ef'] + spectrum[node]['np'] + spectrum[node]['nf'] + spectrum[node]['ep'])

        # RogersTanimoto
        elif spectrum_method == "rogers":
            result[node] = (spectrum[node]['ef'] + spectrum[node]['np']) / (spectrum[node]['ef'] +
                                                                            spectrum[node]['np'] + 2 * spectrum[node]['nf'] + 2 * spectrum[node]['ep'])

    # Top-n节点列表
    top_list = []
    score_list = []
    for index, score in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=True)):
        if index < top_max + 6:
            top_list.append(score[0])
            score_list.append(score[1])
            #print('%-50s: %.8f' % (score[0], score[1]))

    return top_list, score_list, result


def online_anomaly_detect_RCA(file_path):
    # file_path=
    # Read the contents of the pickle file
    file_name = os.path.basename(file_path)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    # #root_cause = f"ts-{file_name.split('_')[0]}-service"
    # root_cause = f"{file_name.split('_')[0]}" #for API
    parts = file_name.split('+')
    rootcause1 = "ts-" + parts[0] + "-service"
    rootcause2 = "ts-" + parts[1].split('_')[0] + "-service"
    abnormal_traces = []
    normal_traces = []

    # Find unique trace_id where predict is 1.0
    predict_trace_ids = data.loc[data['predict'] == 1.0, 'trace_id'].unique()
    # Merge unique trace_ids from both conditions
    data['http_status'] = data['http_status'].astype(int)
    error_trace_id = data.loc[data['http_status'].between(400, 599), 'trace_id'].unique()
    # Filter data where trace_id is in all_abnormal_trace_ids
    all_abnormal_trace_ids = np.unique(np.concatenate((predict_trace_ids, error_trace_id)))
    abnormal_traces_df = data[data['trace_id'].isin(all_abnormal_trace_ids)]
    normal_traces_df = data[~data['trace_id'].isin(all_abnormal_trace_ids)]

    anomaly_list=get_list(abnormal_traces_df)
    normal_list=get_list(normal_traces_df)
    #MicroRank
    operation_operation, operation_trace, trace_operation, pr_trace= get_pagerank_graph(normal_traces_df)

    # print (operation_operation, operation_trace, trace_operation, pr_trace)
    # Calculate operation weights and trace_num_list: A dictionary mapping each operation to the number of traces it is involved in
    normal_trace_result, normal_num_list = trace_pagerank(operation_operation, operation_trace, trace_operation,
                                                                  pr_trace, False)

    a_operation_operation, a_operation_trace, a_trace_operation, a_pr_trace = get_pagerank_graph(abnormal_traces_df)
    anomaly_trace_result, anomaly_num_list = trace_pagerank(a_operation_operation, a_operation_trace,
                                                                    a_trace_operation, a_pr_trace,
                                                                    True)
    top_list, score_list ,result= calculate_spectrum_without_delay_list(anomaly_result=anomaly_trace_result,
                                                                         normal_result=normal_trace_result,
                                                                         anomaly_list_len=len(
                                                                             anomaly_list),
                                                                         normal_list_len=len(
                                                                             normal_list),
                                                                         top_max=5,
                                                                         anomaly_num_list=anomaly_num_list,
                                                                         normal_num_list=normal_num_list,
                                                                         spectrum_method="m2")
    print(top_list, score_list)
    # sort result in our way
    # Iterate through the sorted_ochiai and assign ranks
    rank = 0
    prev_score = None
    ranked_ochiai = {}

    # Iterate over the sorted results
    for service, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
        # If the score is the same as the previous score, keep the same rank
        if prev_score == score:
            ranked_ochiai[service] = (score, rank)
        else:
            rank += 1
            ranked_ochiai[service] = (score, rank)
        prev_score = score

    # Assuming root_cause and file_name are defined or passed as parameters to the function
    score1, rank1 = ranked_ochiai.get(rootcause1, (None, None))
    services_with_same_rank1 = [service for service, (_, r) in ranked_ochiai.items() if
                               r == rank1 and service != rootcause1]
    score2, rank2 = ranked_ochiai.get(rootcause2, (None, None))
    services_with_same_rank2 = [service for service, (_, r) in ranked_ochiai.items() if
                                r == rank2 and service != rootcause2]

    df = pd.DataFrame({
        'Filename': [file_name],
        'Rank1': [rank1],
        'Score1': [score1],
        'Number of Services with Equal Ranks1': [len(services_with_same_rank1)],
        'Rank2': [rank2],
        'Score2': [score2],
        'Number of Services with Equal Ranks2': [len(services_with_same_rank2)]
    })
    return df

    # sleep 5min after a fault
    # time.sleep(240)
    # time.sleep(60)

def get_list(df):
    #get list of traces
    trace_list = df['trace_id'].unique().tolist()
    return trace_list
if __name__ == '__main__':

    directory = r"D:\double_root_NewLabeledForLatency"
    # List all files in the directory
    all_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pkl')]
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Filename', 'Rank', 'Score', 'Number of Services with Equal Ranks'])
    # Parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(online_anomaly_detect_RCA, all_files))
    for result in results:
        df = pd.concat([df, result], ignore_index=True)

    df.to_excel('double-root-russellrao.xlsx', index=False)

