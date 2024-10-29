import csv
import os
from concurrent.futures import ThreadPoolExecutor
import pickle

import pandas as pd


def find_rank_in_pickle(file_path, root_causes):
    search_string1, search_string2 = root_causes
    rank1=0
    rank2=0
    # Load the contents of the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # Check if the data is a dictionary
        if isinstance(data, dict):
            # Iterate over the dictionary items
            for key, value in data.items():
                # Check if the search string is in the list
                if search_string1 in value:
                    rank1= value.index(search_string1) + 1,   # Adding 1 to make it 1-based index
                if search_string1 in value:
                    rank2 = value.index(search_string2) + 1,  # Adding 1 to make it 1-based index

        else:
            print("Invalid pickle file format. Expected a dictionary.")

    # If search string not found or data format is invalid
    return file_path,rank1, rank2

def extract_root_causes(filename):
    filename = os.path.basename(filename)
    parts = filename.split('+')

    rootcause1 = "ts-" + parts[0] + "-service"
    rootcause2 = "ts-" + parts[1].split('_')[0] + "-service"
    return rootcause1,rootcause2


if __name__ == "__main__":
    directory = r"D:\MYDESK\MyPhd\--Thesis--\paper3\data\A\microservice\TraceRCA\final\doubleroot"
    # List comprehension to generate tuples containing both file paths and formatted strings (root causes)

    inputs = [(os.path.join(directory, filename), extract_root_causes(filename)) for filename in os.listdir(directory) if
              filename != 'doubleroot' and os.path.isfile(os.path.join(directory, filename))]

    # Using ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: find_rank_in_pickle(x[0], x[1]), inputs))

    #print on csv
    # Write the results to the CSV file
    # Convert the list of results to a DataFrame
    # Create a list of tuples with individual rank numbers
    processed_results = [(file_path, rank1[0] if rank1 else None, rank2[0] if rank2 else None) for
                         file_path, rank1, rank2 in results]

    # Create a DataFrame from the processed results
    df = pd.DataFrame(processed_results, columns=['File Path', 'rank1', 'rank2'])
    df.to_excel('doubleroot.xlsx', index=False)



