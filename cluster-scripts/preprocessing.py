import pandas as pd
import os
from datetime import datetime
import platform
import multiprocessing
from multiprocessing import Pool


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




def create_days_parallel(data_path):
    """Function to create the merged days dictionary, performed using parallel computation.

    Args:
        data_path (str): path to the folder where the .csv files are stored

    Returns:
        dict: dictionary containing the merged/concatenated days dictionary, based on all available files
    """    
    # In order to read the data from the files, 
    # I need the paths of the files to be passed on to the read_csv() function. 
    file_paths = [ os.path.join(data_path, file) for file in os.listdir(data_path) ]

    # Set the number of processes to run in parallel
    num_processes = multiprocessing.cpu_count() * 2
    # Create a pool of workers to execute the filter_df_by_date function
    with Pool(processes=num_processes) as pool:
        # Use the pool to execute the filter_df_by_date function on each file in parallel
        results = pool.map(df_to_dict_with_date_keys, file_paths)

    print(f'Obtained all results from parallel executions, now merging...\n')
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

    path_separator_Windows = "\\"
    path_separator = '/' if platform.system() == 'Linux' else path_separator_Windows

    # root directory of the Linux university cluster
    rootdir_path = '/home1/s4915372/research-internship'

    # # root directory of the project - if your local machine has enough resources to run the script 
    # rootdir_path = os.getcwd()

    raw_data_path = rootdir_path + f'{path_separator}data{path_separator}covaxxy-csv'
    files_path = rootdir_path + f'{path_separator}files'

    print('Importing data from files...')
    days = create_days_parallel(raw_data_path)
    # # In case the parallel algorithm doesn't complete, you can try running the sequential version.
    # # Results will be the same, albeit the computation time will be longer.
    # days = create_days_sequential(raw_data_path)
    print(f'Total number of tweets is: {count_tweets(days)}')

    sorted_dates_datetimes = sorted([ datetime.strptime(date_string, '%d-%m-%Y') for date_string in days.keys() ])
    formatted_sorted_dates = [ date_object.strftime('%d-%m-%Y') for date_object in sorted_dates_datetimes ]
    number_of_days = f'{len(formatted_sorted_dates)}_days'
    with open(files_path + f'{path_separator}unique_dates_{number_of_days}.txt', 'w') as f:
        for date in formatted_sorted_dates:
            f.write(date + f' - {len(days[date])} tweets\n')

    merged_days = create_merged_days(days)
    # merged_days = merged_days.head(500000)

    merged_days.to_csv(rootdir_path + 
                       f'{path_separator}data{path_separator}covaxxy_merged_{number_of_days}.csv', index=False)
    # merged_days.to_csv(rootdir_path + f'{path_separator}data{path_separator}covaxxy_merged_test.csv', index=False)
    print('Merged file with all data saved locally.')


if __name__ == "__main__":
    main()
