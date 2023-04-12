import pandas as pd
import os
from datetime import datetime
import platform


def filter_df_by_date(path):
    """Function to filter a dataframe by date, given the path to the csv file. 
    Returns a dictionary with the dates as keys and the values being the rows where the value of the 'created_at' column
    corresponds to the key.

    Args:
        path (str): path to the csv file (e.g. /your/path/to/tweet_ids--2021-03-01.csv)

    Returns:
        dict: dictionary with the dates as keys and the values being the rows where the value of the 'created_at' column corresponds
              to the key
    """    
    # print(f'Processing file {os.path.basename(path)} ... \n ')
    # Read the CSV file into a pandas dataframe
    df_from_file = pd.read_csv(path, index_col= False)
        
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



def create_days_per_file_sequential(data_path):
    # In order to read the data from the files, I need the paths of the files to be passed on to the read_csv() function. 
    file_paths = [ os.path.join(data_path, file) for file in os.listdir(data_path) ]

    days_per_file = {}
    for index, file_path in enumerate(file_paths):
        print(f'Processing file {index + 1}/{len(file_paths)} ... \n ')
        days = filter_df_by_date(file_path)
        days_per_file[index + 1] = days
        print(f'Added data from file {index + 1}/{len(file_paths)} to results. \n')


    return days_per_file



def create_days_sequential(data_path):
    days_per_file = create_days_per_file_sequential(data_path)
    
    print(f'We have the data from all files. Now merging and sorting chronologically...\n')
    days = dict()
    # Loop that merges all separate days dictionaries obtained after running the parallel computation
    # into one final dictionary, associated with all available data.
    for key, days_from_file in days_per_file.items():
        days = {k: pd.concat([days.get(k, pd.DataFrame()), days_from_file.get(k, pd.DataFrame())]) 
                for k in set(days) | set(days_from_file)}

    # Dictionary comprehension to format datetime object keys to strings - useful for ease of accessing further down the line.
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

    # change the data type of the 'reference_id' column from string to int - where possible (the '#' values remain the same)
    merged_days['reference_id'] = merged_days['reference_id'].apply(string_to_int)

    return merged_days



def count_tweets(days):
    total_rows = 0
    for key in days:
        total_rows += len(days[key])
    return total_rows




def main():

    path_separator_Windows = "\\"
    path_separator = '/' if platform.system() == 'Linux' else path_separator_Windows

    rootdir_path = os.getcwd()

    data_path = rootdir_path + f'{path_separator}data{path_separator}covaxxy-csv'

    days = create_days_sequential(data_path)
    print(f'Total number of tweets is: {count_tweets(days)}')

    merged_days = create_merged_days(days)
    merged_days.to_csv(rootdir_path + f'{path_separator}data/covaxxy_merged.csv', index=False)
    print('Merged file with all data saved locally.')


if __name__ == "__main__":
    main()
