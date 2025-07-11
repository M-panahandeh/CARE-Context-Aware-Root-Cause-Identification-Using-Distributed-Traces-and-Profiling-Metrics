import itertools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# import click
import numpy as np
import pandas as pd
# from sklearn.ensemble import IsolationForest
# from loguru import logger


from trainticket_config import FEATURE_NAMES
# from diskcache import Cache

DEBUG = True

threshold = 1.0


def anomaly_detection_isolation_forest(df, result_column, history, cache):
    indices = np.unique(df.index.values)
    for source, target in indices:
        empirical = df.loc[(source, target), FEATURE_NAMES].values
        # reference = history.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        if token not in cache:
            df.loc[(source, target), result_column] = 0
            continue
        model = cache[token]
        predict = model.predict(empirical)
        df.loc[(source, target), result_column] = predict
    return df


def anomaly_detection_3sigma_without_useful_features(df, result_column, history, cache):
    indices = np.unique(df.index.values)
    useful_feature = {key: FEATURE_NAMES for key in indices}
    return anomaly_detection_3sigma(df, result_column, None, useful_feature, cache=cache)


def anomaly_detection_3sigma(df, result_column, history, useful_feature, cache):
    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), result_column] = 0
            continue
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values
        mean, std = [], []
        for idx, feature in enumerate(features):
            token = f"reference-{source}-{target}-{feature}-mean-variance"
            if token in cache:
                mean.append(cache[token]['mean'])
                std.append(cache[token]['std'])
            else:
                mean.append(np.mean(empirical,axis=0)[idx])
                std.append(np.maximum(np.std(empirical,axis=0)[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, feature in enumerate(features):
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict


    print(df.loc[df['Ours-predict'] == 1, 'label'])
    return df




def invo_anomaly_detection_main(df, output_anomaly_file, history, useful_feature, cache_file, main_threshold):
    global threshold
    threshold = main_threshold

    history = None
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))
    #print ((useful_feature))
    # logger.debug(f"useful features: {useful_feature}")
    cache=""
    #df in the current (emprical) time window
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()

    tic = time.time()
    df = anomaly_detection_3sigma(df, 'Ours-predict', None, useful_feature, cache=cache)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, 'invos:', len(df))

    # df = anomaly_detection_3sigma_without_useful_features(df, 'NoSelection-predict', None, cache=cache)

    # # tic = time.time()
    # df = anomaly_detection_isolation_forest(df, 'IF-predict', None, cache=cache)
    # # toc = time.time()
    # # print("algo:", "IF", "time:", toc - tic, 'invos:', len(df))

    df['predict'] = df['Ours-predict']

    with open(output_anomaly_file, 'wb+') as f:
        pickle.dump(df, f)

def process_file(input_file):
    print(f"Processing {input_file} started.")  # Debugging information

    # Read the pickle file
    with open(input_file, 'rb') as file:
        data = pickle.load(file)
    # Extract the elements and construct the DataFrame
    rows = []
    for item in data:
        trace_id = item.get('trace_id', '')
        label = item.get('label', None)
        s_t = item.get('s_t', [('', '')])
        timestamps = item.get('timestamp', [])  # List of timestamps

        # Create rows for DataFrame
        # Group the features and source-target pairs by the same index
        for feature_values, st_values, timestamp in zip(itertools.zip_longest(
                *(item.get(key, []) for key in FEATURE_NAMES), fillvalue=None),
                s_t, timestamps):
            # Combine feature values into a row dictionary
            row = {key: value for key, value in zip(FEATURE_NAMES, feature_values)}

            # Add trace-related and timestamp information
            row['trace_id'] = trace_id
            row['label'] = label
            row['timestamp'] = timestamp  # Assign the corresponding timestamp

            # Assign the 'source' and 'target' from the st_values tuple
            row['source'], row['target'] = st_values

        rows.append(row)
    df = pd.DataFrame(rows)

    # Define the filename for your output file based on the input file's name
    filename = Path(input_file).name
    output_anomaly_file = os.path.join(r"D:\Newlabaled-final-timestamp",
                                        filename)

    #read useful features
    directory_features = r"D:\API_features"
    # Create the full path with the _features suffix and .txt extension
    useful_feature = os.path.join(directory_features, filename + "_features.txt")

    # do not need history for Anomal detection
    history = None

    # cash file
    cache_file = ""

    main_threshold = 1

    invo_anomaly_detection_main(df, output_anomaly_file, history, useful_feature, cache_file, main_threshold)

    print(f"Processing {input_file} completed.")  # Debugging information

if __name__ == '__main__':

    # read files
    directory = r"D:\test"
    files = [os.path.join(directory, filename) for filename in os.listdir(directory)]


    # process_file(files[0], history, fisher_threshold)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))


