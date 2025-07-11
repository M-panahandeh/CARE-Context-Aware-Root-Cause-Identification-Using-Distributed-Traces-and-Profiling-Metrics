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

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

DEBUG = False  # very slow


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
    FEATURE_NAMES = ['latency']
    output_file = Path(output_features_file)
    # logger.debug(f'{input_file}')
    df = input_df.set_index(keys=['source', 'target'], drop=True).sort_index()
    # history = history.set_index(keys=['source', 'target'], drop=True).sort_index()

    try:
        # Ensure that indices are consistently formatted and handle NaN values
        df.index = df.index.map(lambda x: tuple(str(item) for item in x))
        df = df.sort_index()
        # Drop NaN values if present
        df.dropna(inplace=True)
        indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    except Exception as e:
        print (e)
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
        try:
            print(dict(useful_features_dict), file=f)
            print("look at output")
        except Exception as e:
            print(e)


def process_file(input_file,history,fisher_threshold):
    print(f"Processing {input_file} started.")  # Debugging information
    # Read the pickle file
    df = pd.read_csv(input_file)
    # Extract the elements and construct the DataFrame


    # Define the filename for your output file based on the input file's name
    output_features_filename = Path(input_file).name + "_features.txt"
    output_features_file = os.path.join(r"D:\features", output_features_filename)


    # call this fucntion to find useful fatures
    selecting_feature_main(input_file,df, output_features_file, history, fisher_threshold)
    print(f"Processing {input_file} completed.")  # Debugging information


if __name__ == '__main__':

    file_path = r'D:\historical_data.csv'
    # read historical data
    history = pd.read_csv(file_path)
    history = history.set_index(keys=['source', 'target'], drop=True).sort_index()
    history.index = history.index.map(lambda x: tuple(str(item) for item in x))
    history = history.sort_index()
    history.dropna(inplace=True)

    fisher_threshold = 0.1

    #read files

    directory = r"D:\PreparedData"
    files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename != 'historical_data.csv']

    # Convert file paths to tasks (each task is a tuple)
    tasks = [(file, history, fisher_threshold) for file in files]

    #process_file(files[0], history, fisher_threshold)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda p: process_file(*p), tasks)








