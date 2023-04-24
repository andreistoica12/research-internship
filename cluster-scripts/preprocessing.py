import os
import pandas as pd
from datetime import datetime
import multiprocessing
from multiprocessing import Pool
import argparse


def df_to_dict_with_date_keys(csv_path):
    """Function to filter a dataframe by date, given the path to the csv file. 
    Returns a dictionary with the dates as keys and the values being the rows 
    where the value of the 'created_at' column corresponds to the key.

    Args:
        path (str): path to the csv file (e.g. /your/path/to/tweet_ids--2021-03-01.csv)

    Returns:
        dict: dictionary with the dates as keys and the values being the rows 
              where the value of the 'created_at' column corresponds to the key
    """    
    # print(f'Processing file {os.path.basename(path)} ... \n ')
    # Read the CSV file into a pandas dataframe
    df_from_file = pd.read_csv(csv_path, index_col= False)
        
    # Convert the "created_at" column to a pandas datetime object
    df_from_file['created_at'] = pd.to_datetime(df_from_file['created_at'])

    # Get all unique timestamp values from the "created_at" column
    unique_dates = df_from_file['created_at'].dt.date.unique()

    # Create a dictionary where the keys are the unique timestamp values
    # and the values are dataframes that correspond to each unique timestamp value
    days = {}
    for date in unique_dates:
        # Extract the rows that have the current timestamp value
        mask = df_from_file['created_at'].dt.date == date
        filtered_df = df_from_file[mask]
        # Store the resulting subset of rows as a dataframe in the dictionary
        days[date] = filtered_df
    
    # print(f'Done processing file {os.path.basename(path)}.')
    return days




def create_all_days_per_file_sequential(raw_data_path):
    """Function to create a dictionary where the keys are integer numbers nominating the processed files
    and the values are dictionaries that correspond to each processed file. Each dictionary is structured as follows:
    - the keys are unique timestamps
    - the values are dataframes that correspond to each unique timestamp

    Args:
        raw_data_path (str): path to the raw data directory (e.g. /your/path/to/covaxxy-csv)

    Returns:
        dict: dictionary with keys as numerical indexes and values as dictionaries
    """    
    # In order to read the data from the files, 
    # I need the paths of the files to be passed on to the read_csv() function. 
    file_paths = [ os.path.join(raw_data_path, file) for file in os.listdir(raw_data_path) ]

    # I decided to store the datasets per file in a dictionary, because it is faster to work with than a list.
    all_days_per_file = {}
    for index, file_path in enumerate(file_paths):
        # print(f'Processing file {index + 1}/{len(file_paths)} ... \n ')
        days_for_one_file = df_to_dict_with_date_keys(file_path)
        all_days_per_file[index + 1] = days_for_one_file
        # print(f'Added data from file {index + 1}/{len(file_paths)} to results. \n')


    return all_days_per_file




def create_days_sequential(raw_data_path):
    """Function to create the final dictionary, with merged and sorted chronologically dataframes for each key.
    Keys are unique timestamps and values are dataframes that correspond to each unique timestamp, 
    with data collected from all files.

    Args:
        raw_data_path (str): path to the raw data directory (e.g. /your/path/to/covaxxy-csv)

    Returns:
        dict: dictionary with keys as formatted timestamps and values as dataframes with
              tweets posted at the date specified by the key
    """    
    all_days_per_file = create_all_days_per_file_sequential(raw_data_path)
    
    print(f'Imported data from all files. Now merging and sorting chronologically...\n')
    days = dict()
    # Loop that merges all separate days dictionaries obtained after running the parallel computation
    # into one final dictionary, associated with all available data.
    for key, days_for_one_file in all_days_per_file.items():
        days = {k: pd.concat([days.get(k, pd.DataFrame()), days_for_one_file.get(k, pd.DataFrame())]) 
                for k in set(days) | set(days_for_one_file)}

    # Dictionary comprehension to format datetime object keys to strings
    #  - useful for ease of accessing further down the line.
    days = {datetime_key.strftime('%d-%m-%Y'): df for datetime_key, df in days.items()}

    # Iterate over all the keys in the dictionary
    for key in days.keys():
        days[key].sort_values('created_at', inplace=True)
        # Drop the "id" column from the dataframe corresponding to each key
        days[key].drop('id', axis=1, inplace=True)


    return days




def create_days_parallel(raw_data_path):
    """Function to create the merged days dictionary, performed using parallel computation.

    Args:
        raw_data_path (str): path to the folder where the .csv files are stored

    Returns:
        dict: dictionary containing the merged/concatenated days dictionary, based on all available files
    """    
    # In order to read the data from the files, 
    # I need the paths of the files to be passed on to the read_csv() function. 
    file_paths = [ os.path.join(raw_data_path, file) for file in os.listdir(raw_data_path) ]

    # Set the number of processes to run in parallel
    num_processes = multiprocessing.cpu_count() * 2
    # Create a pool of workers to execute the filter_df_by_date function
    with Pool(processes=num_processes) as pool:
        # Use the pool to execute the filter_df_by_date function on each file in parallel
        results = pool.map(df_to_dict_with_date_keys, file_paths)

    print(f'Obtained all results from parallel executions, now merging...')
    days = dict()
    # Loop that merges all separate days dictionaries obtained after running the parallel computation
    # into one final dictionary, associated with all available data.
    for result in results:
        days = {k: pd.concat([days.get(k, pd.DataFrame()), result.get(k, pd.DataFrame())]) 
                for k in set(days) | set(result)}

    # Dictionary comprehension to format datetime object keys to strings 
    # - useful for ease of accessing further down the line.
    days = {datetime_key.strftime('%d-%m-%Y'): df for datetime_key, df in days.items()}

    # Iterate over all the keys in the dictionary
    for key in days.keys():
        days[key].sort_values('created_at', inplace=True)
        # Drop the "id" column from the dataframe corresponding to each key
        days[key].drop('id', axis=1, inplace=True)


    return days




def create_merged_days(days):
    """Function to merge all available days in the days dictionary into a single pandas Dataframe.
    The resulting DataFrame will be sorted by date (each value in the input days dictionary has sorted values and
    I make sure I sort the keys in ascending order as well, when iterating through the dictionary and creating the
    dataframe).

    Args:
        days (dict): dictionary where the keys are the date strings and the values are dataframes containing the
                     rows of each day in the whole dataset

    Returns:
        pandas.core.frame.DataFrame: dataframe containing the merged days, sorted in ascending order by date
    """    
    # Here, I merged all data (from all available days) into a single dataframe (they have the same structure).
    # I did that because some replies to a tweet posted today can come some days after, so we need to take care
    # of the dataset as a whole.

    
    # Convert string keys to datetime objects and sort them
    sorted_keys = sorted([datetime.strptime(k, '%d-%m-%Y') for k in days.keys()])

    # Convert datetime objects back to string keys with format '%d-%m-%Y'
    sorted_key_strings = [k.strftime('%d-%m-%Y') for k in sorted_keys]

    # concatenate the dataframes and reset the index
    merged_days = pd.concat([days[key] for key in sorted_key_strings], ignore_index=True)


    def string_to_int(reference_id):
        try:
            return int(reference_id)
        except ValueError:
            return reference_id

    # change the data type of the 'reference_id' column from string to int - 
    # where possible (the '#' values remain the same)
    merged_days['reference_id'] = merged_days['reference_id'].apply(string_to_int)

    return merged_days




def count_tweets(days):
    """Function to count the total number of tweets in the days dictionary.

    Args:
        days (dict): dictionary where the keys are the date strings and the values are dataframes
                     containing the rows of each day

    Returns:
        int: number of tweets in the days dictionary
    """    
    total_rows = 0
    for key in days:
        total_rows += len(days[key])
    return total_rows





def main():
    # The root directory of the project should be 1 level above the preprocessing Python script
    # in the directory tree, so parent level.

    # absolute path of the directory containing the (current) preprocessing Python script
    # (path to the cluster-scripts directory)
    cluster_scripts_path = os.path.dirname(os.path.abspath(__file__))
    # parent level directory - root directory of the project
    rootdir_path = os.path.dirname(cluster_scripts_path)

    files_path = os.path.join(rootdir_path, 'files')


    # NOTE: If this script is located in a different relative location to the root directory of the project,
    # you need to set the root directory path accordingly.

    # In order to offer flexibility when it comed to where each user wishes to store the data,
    # I decided to add some command line arguments when running this script: --input, --output .
    # This way, if the user wishes to run the script in a terminal window, he/she can specify these
    # arguments themselves. The steps to parse the command line arguments are the following:
    # 1. Create an argument parser
    parser = argparse.ArgumentParser()

    # 2. Add arguments for: input folder woth all the .csv files, output (data) folder
    parser.add_argument('--input', type=str, help='Input folder path, storing all the .csv files')
    parser.add_argument('--output', type=str, help='Root data folder path')

    # 3. Parse the command-line arguments
    args = parser.parse_args()

    raw_data_path = args.input
    data_path = args.output

    # If you do not wish to specify command line arguments when running the Python script or 
    # you do not run the script in a terminal window, you can set the paths to the
    # raw data (all the .csv files) and the root folder of the data manually, as follows:

    # raw_data_path = rootdir_path + f'{path_separator}data{path_separator}covaxxy-csv'
    # data_path = rootdir_path + f'{path_separator}data'

    print('Importing data from files...')
    days = create_days_parallel(raw_data_path)
    # # In case the parallel algorithm doesn't complete, you can try running the sequential version.
    # # Results will be the same, albeit the computation time will be longer.
    # days = create_days_sequential(raw_data_path)
    print(f'Total number of tweets is: {count_tweets(days)}')

    sorted_dates_datetimes = sorted([ datetime.strptime(date_string, '%d-%m-%Y') for date_string in days.keys() ])
    formatted_sorted_dates = [ date_object.strftime('%d-%m-%Y') for date_object in sorted_dates_datetimes ]
    number_of_days = f'{len(formatted_sorted_dates)}_days'
    with open(os.path.join(files_path, f'unique_dates_{number_of_days}.txt'), 'w') as f:
        for date in formatted_sorted_dates:
            f.write(date + f' - {len(days[date])} tweets\n')

    merged_days = create_merged_days(days)
    # merged_days = merged_days.head(500000)

    merged_days.to_csv(os.path.join(data_path, f'covaxxy_merged_{number_of_days}.csv'), index=False)
    # merged_days.to_csv(os.path.join(data_path, 'covaxxy_merged_test.csv'), index=False)
    print('Merged file with all data saved locally.')


if __name__ == "__main__":
    main()
