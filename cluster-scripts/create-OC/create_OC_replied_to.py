import platform
import os
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool

from sentistrength import PySentiStr

import re
import contractions
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.data
nltk.download('punkt')
# Load the punkt tokenizer data from the local directory
# nltk.data.load(f'tokenizers{path_separator}punkt{path_separator}english.pickle')

import json


def custom_stop_words(path_to_stopwords):
    """Function to read a .txt file containing (custom) stop words and return a set of these stop words.

    Args:
        path_to_stopwords (str): path to the.txt file containing stop words (e.g. /your/path/to/files/stop_words.txt)

    Returns:
        set: set of stop words
    """    
    stop_words = set()
    with open(path_to_stopwords, 'r') as f:
        for line in f:
            word = line.strip()  # remove whitespace and newline characters
            stop_words.add(word)
    return stop_words



def remove_emoji(text):
    """Function that takes a text string as input and uses a regular expression pattern to match all Unicode characters
    that are classified as emojis. The regular expression includes different ranges of Unicode characters 
    that represent different types of emojis, such as emoticons, symbols, and flags.

    Args:
        text (str): text string to remove emokis from

    Returns:
        str: text string with all emojis removed
    """    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)



def remove_stopwords(text, stop_words):
    """Function that removes stop words from a given text.

    Args:
        text (str): text string
        stop_words (set): set of stop words

    Returns:
        str: text string without stop words
    """    
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove the stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # Join the filtered tokens back into a string
    filtered_text = ' '.join(filtered_tokens)

    return filtered_text



def clean_text(text, stop_words):
    """Function to clean the raw text, e.g. from a tweet. Performs the following steps:
    1. Lowercase all the words in the text
    2. Replace all new line characters with a white space
    3. Remove tags
    4. Remove URLs
    5. Remove punctuations
    6. Convert contractions to their full forms
    7. Remove emojis (emoticons, symbols, flags, etc.)
    8. Remove stopwords


    Args:
        text (str): text string to be cleaned before passing it to the sentiment analysis model
        stop_words (set): set of stop words to be removed from the text

    Returns:
        str: cleaned text string
    """        
    # 1. Lowercase all words in the text
    text = text.lower()

    # 2. Replace the new line character with empty string
    text = text.replace("\n", "")
    
    # 3. Remove words starting with '@' - tags (most common noise in replies)
    text = re.sub(r'@\w+', '', text, flags=re.MULTILINE)

    # 4. Remove words starting with 'http' - hyperlinks
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)

    # 5. Remove punctuation from the text using regular expressions
    text = re.sub(r'[^\w\s]', '', text)

    # 6. Remove contractions, such as you're => you are
    contractions.fix(text)

    # 7. Remove emojis
    text = remove_emoji(text)

    # 8. Remove stopwords in English
    text = remove_stopwords(text, stop_words)

    return text



def string_to_int(reference_id):
    try:
        return int(reference_id)
    except ValueError:
        return reference_id



def create_path_to_opinion_changes(reaction_types, opinion_changes_path, path_separator):
    """Function to create the path to the opinion changes JSON file, based on the reaction types we took into consideration.

    Args:
        reaction_types (list): list of reaction types

    Returns:
        str: path to the opinion changes file
    """    
    type = "_".join(reaction_types)
    path = opinion_changes_path + f"{path_separator}{type}_OC.json"

    return path



def group_reactions(merged_days, reaction_types):
    """Function to group reactions based on the reaction types list given as an input parameter, by the
    'author_id' and 'reference_id' columns. This means that each group of reactions contains a (set of) reaction(s)
    posted by the user identified by the 'author_id' and the source tweet identified by the 'reference_id'.

    Args:
        merged_days (pandas.core.frame.DataFrame): dataframe with all the data
        reaction_types (list): list of reaction types we want to consider

    Returns:
        dict: dictionary where the key is a tuple of the form (author_id, reference_id)
              and the value is a dataframe with all reactions corresponding to that combination
    """    
    reactions = merged_days[merged_days['reference_type'].isin(reaction_types)]
    multiple_reactions = reactions[reactions.duplicated(subset=['author_id', 'reference_id'], keep=False)]

    # group the rows by the two columns
    grouped_df = multiple_reactions.groupby(['author_id', 'reference_id'])
    groups_of_reactions = grouped_df.groups

    return groups_of_reactions



def compute_sentiments(rows_indexes, dataset, stop_words, senti):
    """Function to compute the sentiment list for a set of rows in the dataset (given by dataset), taking
    into account the given stop words.

    Args:
        rows_indexes (pandas.core.indexes.numeric.Int64Index): indexes of rows in the dataset that we want to compute the sentiment for
        dataset (pandas.core.frame.DataFrame): dataframe containing the dataset
        stop_words (set): set of stop words

    Returns:
        list: list of sentiment scores for each row identified by rows_indexes
    """    
    texts = [ clean_text(dataset.loc[index, 'text'], stop_words) 
             if dataset.loc[index, 'reference_type'] != 'retweeted' else 'extremely fabulous'
             for index in rows_indexes ]
    
    sentiments = senti.getSentiment(texts, score='scale')

    return sentiments



def opinion_change(rows_indexes, dataset, stop_words, senti):
    """Function to detect whether an opinion change occured within a group of reactions (replies/quotes/retweets).

    Args:
        rows_indexes (pandas.core.indexes.numeric.Int64Index): list of indexes in the original dataframe (dataset)
                                                               where we aim to detect an opinion change
                                                               (e.g. Int64Index([1848965, 1850146, 1850687], dtype='int64'))
        dataset (pandas.core.frame.DataFrame): dataframe containing the opinion changes
        stop_words (list): list of stopwords

    Returns:
        bool: boolean value which confirms or denies the existence of an opinion change between the rows analyzed
    """ 
    sentiments = compute_sentiments(rows_indexes, dataset, stop_words, senti)
    sentiments = np.array(sentiments)

    positive = np.any(sentiments > 0)
    negative = np.any(sentiments < 0)

    return positive and negative



def process_sentiments_for_group(rows_indexes, merged_days, stop_words, senti):
    """Function to compute the sentiments of the tweets provided by the row indexes within the merged_days dataframe.
    Returns a list of sentiments corresponding to each of the tweets or an empty list if there was no opinion change 
    within that group.

    Args:
        rows_indexes (pandas.core.indexes.numeric.Int64Index): list of indexes in the original dataframe (dataset)
                                                               where we aim to detect an opinion change
                                                               (e.g. Int64Index([1848965, 1850146, 1850687], dtype='int64'))
    Returns:
        list: list of sentiments for the rows or empty list if there was no opinion change within that group.
    """    

    processed_values = []
    if opinion_change(rows_indexes, merged_days, stop_words, senti):
        processed_values = compute_sentiments(rows_indexes, merged_days, stop_words, senti)

    return processed_values




def process_dict_chunk(input_dict, merged_days, stop_words, senti):
    """Function to create an opinion_changes dictionary only for a chunk of data.

    Args:
        input_dict (dict): chunk of data

    Returns:
        dict: opinion_changes dictionary for a chunk of data
    """    
    # Process a chunk of the input dictionary
    processed_dict = {}
    counter = 0
    progress = 0.0001
    
    for group, rows_indexes in input_dict.items():
        processed_values = process_sentiments_for_group(rows_indexes, merged_days, stop_words, senti)
        if processed_values:  # only add non-empty lists to the dictionary
            processed_dict[group] = processed_values

        counter += 1
        if ((counter / len(input_dict)) >= progress):
            print(f"{counter} / {len(input_dict)} entries processed...\n")
            progress += 0.0001
        if counter == len(input_dict):
            print(f"Thread has finished processing all {len(input_dict)} entries.")


    return processed_dict




def process_dict_in_parallel(input_dict, merged_days, stop_words, senti, num_processes=None):
    """Function to process the input dictionary of reactions in parallel and merge the atomic results together into a single dictionary,
    which will be the final opinion_changes dictionary.

    Args:
        input_dict (dict): dictionary of reactions grouped by some columns
                           (we expect the columns to be 'author_id' and 'reference_id')
        num_processes (int): number of parallel(worker) threads/processes. Defaults to None.

    Returns:
        dict: the final opinion_changes dictionary, which contains all the pairs of 'author_id' and 'reference_id'
              (and their respective rows in the original dataframe) in the whole dataset, where an opinion change occured
    """    
    # Default to using all available CPU cores
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    print(f'Chose number of processes: {num_processes}')

    # Split the input dictionary into smaller chunks for parallel processing
    chunk_size = len(input_dict) // num_processes if len(input_dict) // num_processes != 0 else 1
    input_chunks = [dict(list(input_dict.items())[i:i + chunk_size]) for i in range(0, len(input_dict), chunk_size)]
    print(f'Splitted input dictionary into {len(input_chunks)} chunks of size {chunk_size}')

    args = []
    for input_chunk in input_chunks:
        args.append(tuple([input_chunk, merged_days, stop_words, senti]))
    # Process the input chunks in parallel using a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_dicts = pool.starmap(process_dict_chunk, args)

    # Merge the processed dictionaries from each input chunk
    processed_dict = {}
    for d in processed_dicts:
        processed_dict.update(d)

    return processed_dict




def save_opinion_changes_to_JSON(opinion_changes, reaction_types, opinion_changes_path, path_separator):
    """Function to save the dictionary of opinion changes to a JSON file.

    Args:
        opinion_changes (dict): dictionary with opinion changes
        reaction_types (list): list of reaction types
    """    
    path = create_path_to_opinion_changes(reaction_types, opinion_changes_path, path_separator)

    # create a new dictionary with string keys
    opinion_changes_for_JSON_file = {str(key): value for key, value in opinion_changes.items() }
    with open(path, 'w') as file:
        json.dump(opinion_changes_for_JSON_file, file, indent=4)



def create_all_OC(merged_days, stop_words, senti, reaction_types_full_list, opinion_changes_path, path_separator):
    for reaction_types in reaction_types_full_list:
        groups_of_reactions = group_reactions(merged_days, reaction_types)
        opinion_changes_parallel = process_dict_in_parallel(groups_of_reactions, merged_days, stop_words, senti)
        save_opinion_changes_to_JSON(opinion_changes_parallel, reaction_types, opinion_changes_path, path_separator)




def main():
    path_separator_Windows = "\\"
    path_separator = '/' if platform.system() == 'Linux' else path_separator_Windows
    # root directory of the Linux university cluster
    rootdir_path = '/home1/s4915372/research-internship'

    # # root directory of the project - if your local machine has enough resources to run the script 
    # rootdir_path = os.getcwd()

    dataset_possibilities = ['15_days', '25_days']
    number_of_days = dataset_possibilities[1]

    files_path = rootdir_path + f'{path_separator}files'

    data_path = rootdir_path + f'{path_separator}data{path_separator}covaxxy_merged_{number_of_days}.csv'
    # data_path = rootdir_path + f'{path_separator}data{path_separator}covaxxy_merged_test.csv'

    opinion_changes_path = files_path + f'{path_separator}opinion-changes-{number_of_days}'
    # opinion_changes_path = files_path + f'{path_separator}opinion-changes-test'

    path_to_sentistrength = rootdir_path + f'{path_separator}SentiStrength'
    path_to_sentistrength_jar = path_to_sentistrength + f'{path_separator}SentiStrengthCom.jar'
    path_to_sentistrength_language_folder = path_to_sentistrength + f'{path_separator}LanguageFolder'

    path_to_stopwords = files_path + f"{path_separator}stopwords.txt"
    stop_words = custom_stop_words(path_to_stopwords)

    merged_days = pd.read_csv(data_path)
    merged_days['reference_id'] = merged_days['reference_id'].apply(string_to_int)

    reaction_types_full_list = [['quoted'], 
                            ['quoted', 'retweeted'], 
                            ['replied_to'], 
                            ['replied_to', 'quoted'], 
                            ['replied_to', 'quoted', 'retweeted'],
                            ['replied_to', 'retweeted']]
    
    senti = PySentiStr()
    senti.setSentiStrengthPath(path_to_sentistrength_jar)
    senti.setSentiStrengthLanguageFolderPath(path_to_sentistrength_language_folder)



    # # Create all opinion changes files, corresponding to all combinations of reaction types, all at once
    # create_all_OC(merged_days, stop_words, senti, reaction_types_full_list, opinion_changes_path, path_separator)


    # Instead of creating all files at once, I decided to create them one at a time, due to memory restrictions.
    # More specifically, the univeristy cluster's resource scheduler needs me to specify the memory is should allocate
    # beforehand. Because the script initially created all files at once, I needed to reserve a lot of memory,
    # so thre job would be scheduled too late. This was not necessary, because for each file, most operations
    # need to be performed again, there is no global state variable which can be reused. 
    # This is why I opted for the following option.
    reaction_types = reaction_types_full_list[2]

    groups_of_reactions = group_reactions(merged_days, reaction_types)
    opinion_changes_parallel = process_dict_in_parallel(groups_of_reactions, merged_days, stop_words, senti)
    save_opinion_changes_to_JSON(opinion_changes_parallel, reaction_types, opinion_changes_path, path_separator)



if __name__ == "__main__":
    main()
