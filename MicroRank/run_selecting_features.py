import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os
import itertools

import click
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm
from pprint import pprint
# from trainticket_config import FEATURE_NAMES
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

DEBUG = False  # very slow
FEATURE_NAMES = ['latency']

def distribution_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def fisher_criteria(empirical, reference, side='two-sided'):
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


def selecting_feature_main(input_file,input_df, output_features_file, history, fisher_threshold):

    output_file = Path(output_features_file)
    # logger.debug(f'{input_file}')
    df = input_df.set_index(keys=['source', 'target'], drop=True).sort_index()
    history = history.set_index(keys=['source', 'target'], drop=True).sort_index()
    indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    useful_features_dict = defaultdict(list)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    for (source, target), feature in tqdm(product(indices, FEATURE_NAMES)):
        try:
            empirical = np.sort(df.loc[(source, target), feature].values)
            reference = np.sort(history.loc[(source, target), feature].values)
            # p_value = ks_2samp(
            #     empirical, reference, alternative=ALTERNATIVE[feature]
            # )[1]
            p_value = -1
            fisher = stderr_criteria(empirical, reference, fisher_threshold)
            # fisher = distribution_criteria(empirical, reference,fisher_threshold)
            # if target == 'ts-station-service':
            #    print(source,feature,fisher)
            # fisher = fisher_criteria(empirical, reference, side=ALTERNATIVE[feature])
            # if target == 'ts-food-service':
            #     logger.debug(f"{source} {target} {feature} {fisher} "
            #                  f"{np.mean(empirical)} {np.mean(reference)} {np.std(reference)}")
            if fisher:
                useful_features_dict[(source, target)].append(feature)
            try:
                if DEBUG:
                    import matplotlib.pyplot as plt
                    from matplotlib.figure import Figure
                    fig = Figure(figsize=(4, 3))
                    # x = np.sort(np.concatenate([empirical, reference]))
                    # print('DEBUG:')
                    # print(empirical,reference)
                    sns.distplot(empirical, label='Empirical')
                    sns.distplot(reference, label='Reference')
                    plt.xlabel(feature)
                    plt.ylabel('PDF')
                    plt.legend()
                    plt.title(f"{source}->{target}, ks={p_value:.2f}, fisher={fisher:.2f}")
                    plt.savefig(
                        plot_dir / f"{input_file.name.split('.')[0]}_{source}_{target}_{feature}.pdf",
                        bbox_inches='tight', pad_inches=0
                    )
            except:
                pass
            # logger.debug(f"{input_file.name} {source} {target} {feature} {fisher}")
            # useful_features_dict[(source, target)].append(feature)
        except Exception as e:
            # Handle any exceptions that may occur during processing
            pass

    # logger.debug(f"{input_file.name} {dict(useful_features_dict)}")
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)
        print("look at output")


def process_file(input_file,history,fisher_threshold):
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
    output_features_filename = Path(input_file).name + "_features.txt"
    output_features_file = os.path.join(r"D:\features_latency", output_features_filename)


    # call this fucntion to find useful fatures
    selecting_feature_main(input_file,df, output_features_file, history, fisher_threshold)
    print(f"Processing {input_file} completed.")  # Debugging information


if __name__ == '__main__':
    # Open a folder dialog for the user to select a folder
    folder_path = r'D:\uninjection'
    # Initialize an empty list to store DataFrames
    history_ls = []
    # Iterate through all files in the selected folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Open the pickle file in binary read mode
            with open(file_path, 'rb') as f:
                # Load the data from the file using pickle
                data = pickle.load(f)
                history_ls.extend(data)
        except Exception as e:
            print(f"An error occurred while processing {filename}: {str(e)}")
    # Initialize an empty list to store the rows of the DataFrame
    rows = []
    # Process the consolidated data
    for item in history_ls:
        trace_id = item.get('trace_id', '')
        label = item.get('label', None)
        s_t = item.get('s_t', [('', '')])
        for values, st_values in itertools.zip_longest(
                itertools.zip_longest(*(item.get(key, []) for key in FEATURE_NAMES), fillvalue=None),
                s_t,
                fillvalue=('', '')
        ):
            row = {key: value for key, value in zip(FEATURE_NAMES, values)}
            row['trace_id'] = trace_id
            row['label'] = label
            row['source'], row['target'] = st_values
            rows.append(row)
    # Create a DataFrame from the list of rows
    history = pd.DataFrame(rows)

    fisher_threshold = 0.1

    #read files
    directory = r"D:\test"
    files = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    # Convert file paths to tasks (each task is a tuple)
    tasks = [(file, history, fisher_threshold) for file in files]

    #process_file(files[0], history, fisher_threshold)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: process_file(*p), tasks)








