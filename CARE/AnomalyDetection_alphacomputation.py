import itertools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import click
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from loguru import logger


from trainticket_config import FEATURE_NAMES
from diskcache import Cache

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


def invo_anomaly_detection_main(df, history, useful_feature, cache_file, main_threshold):
    global threshold
    threshold = main_threshold

    history = None
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))
    # logger.debug(f"useful features: {useful_feature}")
    cache=""
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()
    tic = time.time()
    df = anomaly_detection_3sigma(df, 'Ours-predict', None, useful_feature, cache=cache)
    toc = time.time()
    print("algo:", "ours", "time:", toc - tic, 'invos:', len(df))

    df['predict'] = df['Ours-predict']
    return df


def anomaly_severity_assignment(df,useful_feature):
    #we cmpute alpha here
    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))

    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), 'alpha_product'] = 1
            continue
        features = useful_feature[(source, target)]

        # Extract the values using .loc
        empirical = df.loc[(source, target), features].values

        # Compute mean and standard deviation
        mean = np.mean(empirical, axis=0)
        std = np.maximum(np.std(empirical, axis=0), 1e-10)#handle zero std values by max

        # Compute the anomaly score
        alpha = np.abs(empirical - mean) / std
        # alpha = np.atleast_1d(alpha) #for value assignment
        alpha[empirical == mean] = 0  # Handle cases where empirical is equal to mean
        # Update values in alpha array that are less than or equal to 1
        alpha[alpha <= 1] = 1

        # Assign the values back to the dataframe
        product = np.prod(alpha, axis=1)
        df.loc[(source, target), 'alpha_product'] = product

    return df

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
        # Create rows for DataFrame
        for values, st_values in itertools.zip_longest(
                itertools.zip_longest(*(item.get(key, []) for key in FEATURE_NAMES), fillvalue=None),
                s_t, fillvalue=('', '')):
            row = {key: value for key, value in zip(FEATURE_NAMES, values)}
            row['trace_id'] = trace_id
            row['label'] = label
            # Assigning the 'source' and 'target' from the st_values tuple
            row['source'], row['target'] = st_values
            rows.append(row)
    df = pd.DataFrame(rows)

    # Define the filename for your output file based on the input file's name
    filename = Path(input_file).name
    output_anomaly_file = os.path.join(r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\f2_NewLabeld_alpha",
                                        filename)

    #read useful features
    directory_features = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\features2"
    # Create the full path with the _features suffix and .txt extension
    useful_feature = os.path.join(directory_features, filename + "_features.txt")

    # do not need history for Anomal detection
    history = None

    # cash file
    cache_file = ""

    main_threshold = 1

    df=invo_anomaly_detection_main(df, history, useful_feature, cache_file, main_threshold)
    df=anomaly_severity_assignment(df,useful_feature)
    #write in destination
    with open(output_anomaly_file, 'wb+') as f:
        pickle.dump(df, f)

    print(f"Processing {input_file} completed.")  # Debugging information

if __name__ == '__main__':

    # read files
    directory = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\test"
    files = [os.path.join(directory, filename) for filename in os.listdir(directory)]


    # process_file(files[0], history, fisher_threshold)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, files))


