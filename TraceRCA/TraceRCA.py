import csv
import os
from concurrent.futures import ThreadPoolExecutor
import pickle

import pandas as pd


def find_rank_in_pickle(file_path, search_string):
    # Load the contents of the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # Check if the data is a dictionary
        if isinstance(data, dict):
            # Iterate over the dictionary items
            for key, value in data.items():
                # Check if the search string is in the list
                if search_string in value:
                    # Return the index of the search string in the list
                    return file_path,value.index(search_string) + 1  # Adding 1 to make it 1-based index
        else:
            print("Invalid pickle file format. Expected a dictionary.")

    # If search string not found or data format is invalid
    return None

def extract_root_cause(filename):
    file_name_parts = filename.split('_')

    #for API and container
    # return file_name_parts[0]
    #for service
    return 'ts-' + file_name_parts[0] + '-service' if len(file_name_parts) >= 2 else None

if __name__ == "__main__":
    directory = r"D:\MS"
    # List comprehension to generate tuples containing both file paths and formatted strings (root causes)

    inputs = [(os.path.join(directory, filename), extract_root_cause(filename)) for filename in os.listdir(directory) if
              filename != 'doubleroot' and os.path.isfile(os.path.join(directory, filename))]

    # Using ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: find_rank_in_pickle(x[0], x[1]), inputs))

    #print on csv
    # Write the results to the CSV file
    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results, columns=['File Path', 'rank_of_root'])
    df.to_excel('ms.xlsx', index=False)

