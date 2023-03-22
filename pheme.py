import os
import shutil
import json
from dateutil import parser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


rootdir_path = '/home/andreistoica12/research-internship'

data_path = '/home/andreistoica12/research-internship/data/PhemeDataset'

events_path = data_path + "/threads/en"

files_path = os.path.join(rootdir_path, 'files')
if os.path.exists(files_path):
    shutil.rmtree(files_path, ignore_errors=False, onerror=None)
os.makedirs(files_path)

graphs_path = os.path.join(rootdir_path, 'graphs')
if os.path.exists(graphs_path):
    shutil.rmtree(graphs_path, ignore_errors=False, onerror=None)
os.makedirs(graphs_path)

pheme_graphs_path = os.path.join(graphs_path, 'pheme')
if os.path.exists(pheme_graphs_path):
    shutil.rmtree(pheme_graphs_path, ignore_errors=False, onerror=None)
os.makedirs(pheme_graphs_path)

pheme_longitudinal_analysis_graphs = os.path.join(pheme_graphs_path, 'longitudinal-analysis')
if os.path.exists(pheme_longitudinal_analysis_graphs):
    shutil.rmtree(pheme_longitudinal_analysis_graphs, ignore_errors=False, onerror=None)
os.makedirs(pheme_longitudinal_analysis_graphs)

pheme_reaction_times_graphs = os.path.join(pheme_graphs_path, 'reaction-times')
if os.path.exists(pheme_reaction_times_graphs):
    shutil.rmtree(pheme_reaction_times_graphs, ignore_errors=False, onerror=None)
os.makedirs(pheme_reaction_times_graphs)

pheme_reaction_counts_graphs = os.path.join(pheme_graphs_path, 'reaction-counts')
if os.path.exists(pheme_reaction_counts_graphs):
    shutil.rmtree(pheme_reaction_counts_graphs, ignore_errors=False, onerror=None)
os.makedirs(pheme_reaction_counts_graphs)

pheme_influencers_graphs = os.path.join(pheme_graphs_path, 'influencers')
if os.path.exists(pheme_influencers_graphs):
    shutil.rmtree(pheme_influencers_graphs, ignore_errors=False, onerror=None)
os.makedirs(pheme_influencers_graphs)

pheme_avg_followers_counts_graphs = os.path.join(pheme_graphs_path, 'avg-followers-counts')
if os.path.exists(pheme_avg_followers_counts_graphs):
    shutil.rmtree(pheme_avg_followers_counts_graphs, ignore_errors=False, onerror=None)
os.makedirs(pheme_avg_followers_counts_graphs)



clustering_algorithms = ['k-means', 'DBSCAN']

clustering_algorithm = clustering_algorithms[1]



def tweet_hour(tweet_path):
    """Function that parses a JSON file associated with a tweet in the PhemeDataset and returns the posting hour.

    Args:
        tweet_path (str): path to the JSON file associated with a tweet

    Returns:
        int: posting hour of a tweet
    """    
    with open(tweet_path) as f:
        tweet = json.load(f)
    date = parser.parse(tweet['created_at'])
    
    return date.hour


def source_tweet_path(story_path):
    """Function that, given the path to a story, gets the path to the JSON file corresponding to the source tweet.

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)

    Returns:
        str: path to the source tweet JSON file
    """    
    source_dir_path = story_path + "/source-tweets"
    source_path = source_dir_path + "/" + os.listdir(source_dir_path)[0]
    
    return source_path


def reaction_tweets_paths(story_path):
    """Function that generates a list of reactions (replies) to a tweet within a story.

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)

    Returns:
        list: list of paths(strings) for the reactions to the source tweet of a story
    """    
    reactions_paths_list = []
    reactions_dir_path = story_path + "/reactions"
    for reaction_name in os.listdir(reactions_dir_path):
        reaction_path = reactions_dir_path + "/" + reaction_name
        reactions_paths_list.append(reaction_path)
        
    return reactions_paths_list


def validateJSON(JSON_path):
    """Function that checks whether a JSON file is valid or invalid.

    Args:
        JSON_path (str): path to a JSON file

    Returns:
        bool: True for a valid JSON file, False otherwise
    """    
    try:
        with open(JSON_path, 'r') as file:
            data = json.load(file)
    except ValueError as err:
        return False
    return True


def format_retweets_json(retweets_path):
    """Function that transforms an invalid JSON file which contains multiple objects, not correctly separated through a comma and
    not having square brackets at the beginning and end of the file, into a valid JSON file.

    Args:
        retweets_path (str): path to the invalid JSON file
    """    
    if not validateJSON(retweets_path):
        with open(retweets_path, 'r') as invalid_json:
            data = invalid_json.read()
        data = "[\n" + data.replace("}\n{", "},\n{") + "]"
        with open(retweets_path,'w') as valid_json:
            valid_json.write(data)


def hours_list_retweets(story_path):
    """Function that generates a list of the posting hours of all retweets.

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)

    Returns:
        list: list of the posting hours for all retweet occurences.
    """    
    retweets_path = story_path + "/retweets.json"
    hours = []
    if os.path.exists(retweets_path):
        format_retweets_json(retweets_path)
        with open(retweets_path, 'r') as file:
            retweets_list = json.load(file)
        if type(retweets_list) == list:
            hours = [ parser.parse(retweet['created_at']).hour for retweet in retweets_list ]
        else:   # we have this case when the JSON file contains one object, but we need to pass a list forward, so we'll have a 1-length list
            hours = [parser.parse(retweets_list['created_at']).hour]

    return hours


def hours_list_story(story_path):
    """Function that generates a list of the posting hours of all tweets (source tweet, replies, retweets) within a story.

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)

    Returns:
        list: list of the posting hours for all tweets (source tweet, replies, retweets) within a story.
    """    
    # I create a list with all occurences of dates corresponding to the source tweet, reactions (replies) and retweets.
    hours = []

    # source hour
    source_path = source_tweet_path(story_path)
    hour = tweet_hour(source_path)
    hours.append(hour)

    # reactions hours
    reactions_paths_list = reaction_tweets_paths(story_path)
    for reaction_path in reactions_paths_list:
        hour = tweet_hour(reaction_path)
        hours.append(hour)
    
    # retweets hours
    hours.extend(hours_list_retweets(story_path))
    
    return hours


def time_distribution_event(event_path):
    """Function that creates a distribution of the posting hours of the tweets (source tweets, replies, retweets) related to an event.

    Args:
        event_path (str): path to the root folder of a event (e.g. /your/path/to/charliehebdo)

    Returns:
        pandas.core.series.Series: distribution of the posting hours of all tweets related to an event
    """    
    hours = []
    for story_id in os.listdir(event_path):
        story_path = event_path + "/" + story_id
        hours.extend(hours_list_story(story_path))
    hours.sort()
    hours_series = pd.Series(hours)
    distribution = hours_series.value_counts()[hours_series.unique()]
    
    return distribution


def plot_time_distribution_event(event_name, distribution):
    """Function that saves the bar chart plot with the distribution of the posting hours of all tweets related to an event, locally.

    Args:
        event_name (str): name of an event (e.g. charliehebdo)
        distribution (pandas.core.series.Series): distribution of the posting hours of all tweets related to an event
    """  
    global pheme_longitudinal_analysis_graphs  
    axes = distribution.plot(kind='bar')
    figure_path = f"{pheme_longitudinal_analysis_graphs}/{event_name}_distribution.png"
    axes.figure.savefig(figure_path)
    plt.close()


def plot_time_distributions(events_path):
    """Function that saves a bar chart plot with the distribution of the posting hours of all tweets related to each event, locally.

    Args:
        events_path (str): path to the root folder of the events (e.g. /your/path/to/threads/en)
    """    
    for event_name in os.listdir(events_path):
        event_path = events_path + "/" + event_name
        distribution = time_distribution_event(event_path)
        plot_time_distribution_event(event_name, distribution)


def deltas_story(story_path, deltas_type):
    """Function that generates a list of time differences between the moment a reaction (reply or retweet) has been posted
    and the moment the source tweet of a story has been posted. Depending on the time unit of such time difference 
    (e.g. 'hours' or 'minutes'), the function outputs a list with numbers expressed in the respective type.

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)
        deltas_type (str): time unit for time differences

    Returns:
        list: list of time differences expressed in deltas_type time units
    """    
    if deltas_type not in ['minutes', 'hours']:
        print("Deltas type doesn't have a valid value - it should be either 'hours' or 'minutes' !")
        return []
    
    factor = 60 * 60
    factor = 60 if deltas_type == 'minutes' else factor

    deltas = []

    # Step 1: get t0 datetime object from the source timestamp
    source_path = source_tweet_path(story_path)
    with open(source_path) as file:
        source = json.load(file)
    t0 = parser.parse(source['created_at'])

    # Step 2: for all reactions, get the difference in minutes/hours 
    # from the time the source was posted and the time each reaction was posted
    reactions_paths_list = reaction_tweets_paths(story_path)
    for reaction_path in reactions_paths_list:
        with open(reaction_path) as file:
            reaction = json.load(file)
        deltas.append((parser.parse(reaction['created_at']) - t0).total_seconds() / factor )

    # Step 3: for all retweets, get the same time difference in miuntes as above
    retweets_path = story_path + "/retweets.json"
    if os.path.exists(retweets_path):
        format_retweets_json(retweets_path)
        with open(retweets_path, 'r') as file:
            retweets_list = json.load(file)
        if type(retweets_list) == list:
            deltas.extend([ (parser.parse(retweet['created_at']) - t0).total_seconds() / factor for retweet in retweets_list ])
        else:   # here, the JSON file contains one object, but we need to pass a list forward, so we'll have a 1-length list
            deltas.extend([ (parser.parse(retweets_list['created_at']) - t0).total_seconds() / factor ])

    return deltas


def deltas_event(event_path, deltas_type):
    """Function that generates a list of time differences between the source tweets of all stories of an event
    and their respective reactions (replies and retweets).

    Args:
        event_path (str): path to the root folder of an event (e.g. /your/path/to/charliehebdo)
        deltas_type (str): time unit for time differences (e.g. 'hours', 'minutes')

    Returns:
        list: list of time differences expressed in the time unit specified as an input parameter (deltas_type)
    """
    deltas = []
    for story_name in os.listdir(event_path):
            story_path = event_path + "/" + story_name
            deltas.extend(deltas_story(story_path, deltas_type))

    return deltas


def deltas_all_events(events_path, deltas_type):
    """Function that generates a dictionary containing all time differences between the source tweets and their respective reactions
    (replies and retweets) for all events.

    Args:
        events_path (str): path to the root folder of the events (e.g. /your/path/to/threads/en)
        deltas_type (str): time unit for time differences (e.g. 'hours', 'minutes')

    Returns:
        dict: dictionary where keys are the event names and the values are lists of time differences
              expressed in the time unit specified as an input parameter (deltas_type)
    """    
    deltas_all_events = {}
    for event_name in os.listdir(events_path):
        event_path = events_path + "/" + event_name
        deltas_all_events[event_name] = deltas_event(event_path, deltas_type)
    
    return deltas_all_events


def elbow_method_k_means_clusters(deltas, max_nr_clusters):
    """Function that calculates the optimal number of clusters that the K-Means clustering algorithm should create out of a list
    of one-dimensional elements. The function implements the Elbow Method of choosing the optimal number of clusters, meaning it fits 
    multiple K-Means models with different numbers of clusters (from 1 to max_nr_clusters) and compute the within-cluster sum of squares
    (wcss) for each model. The next step is to plot the wcss values against the number of clusters and look for the "elbow" point,
    where the rate of decrease in wcss begins to level off. This is a good indication of the optimal number of clusters. This step is
    performed with the help of the KneeLocator function within the kneed module.

    Args:
        deltas (list): one-dimensionaal list (e.g. time differences between the posting hour of a source tweet
                       and the posting hours of the reactions - replies and retweets)
        max_nr_clusters (int): maximum number of clusters that we take into account as potential optimal number to be returned

    Returns:
        int: optimal number of clusters for a K-Means clusterization
    """    
    wcss = []   # within-cluster sum of squares
    deltas = np.array(deltas)
    deltas = deltas.reshape(-1, 1)
    for k in range(1, max_nr_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(deltas)
        wcss.append(kmeans.inertia_)

    kn = KneeLocator(range(1, max_nr_clusters+1), wcss, curve='convex', direction='decreasing')

    # # Plot the within-cluster sum of squares against the number of clusters
    # # If you want to visualize the knee point of the graph, you can plot the graph using the following function, 
    # # specifically designed to highlight the knee point
    # kn.plot_knee()

    return kn.knee


def k_means_clustering(k, deltas):
    """Function to generate the clusters after performing the K-Means clustering algoithm to a one-dimensional array.

    Args:
        k (int): number of clusters
        deltas (list): one-dimensional list (e.g. of time differences between the posting hour of a source tweet
                       and the posting hours of the reactions - replies and retweets)

    Returns:
        dict: dictionary with keys as auto-generated labels of the clusters and values as lists of elements in each cluster
    """    
    deltas = np.array(deltas)
    deltas = deltas.reshape(-1, 1)

    # Create a KMeans object with the specified number of clusters
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    kmeans.fit(deltas)

    # Get the labels assigned to each data point
    labels = kmeans.labels_

    # Get the centroids of each cluster
    centroids = kmeans.cluster_centers_

    # # Print an overview of the resulting clusters
    # for i in range(k):
    #     cluster_data = deltas[labels == i]
    #     print(f"Cluster {i+1} has {len(cluster_data)} data points and a centroid of {centroids[i][0]}")

    # Create a dictionary to store the clustered data
    clusters_dict = {}
    labels_list = []
    for label in labels:
        labels_list.append(label)
    
    for index, label in enumerate(labels_list):
        if label not in clusters_dict:
            clusters_dict[label] = [deltas[index][0]]
        else:
            clusters_dict[label].append(deltas[index][0])
    
    return clusters_dict


def epsilon_DBSCAN(deltas):
    """Function to compute the optimal value of the epsilon parameter - the most important input parameter - of the 
    DBSCAN clustering algorithm. The function first computes the distances between each point and its n-th (in our case, 5th)
    nearest neighbor and chooses the optimal value for the distance using the Elbow Method (it creates a graph of the distances
    against each data point and chooses the point where the curve starts to rise steeply).

    Args:
        deltas (list): one-dimensional list (e.g. of time differences between the posting hour of a source tweet
                       and the posting hours of the reactions - replies and retweets)

    Returns:
        float: optimal epsilon value for DBSCAN
    """    
    deltas = np.array(deltas)
    deltas = deltas.reshape(-1, 1)

    neigh = NearestNeighbors(n_neighbors=5) # usually, the kth nearest neighbor is chosen somewhere between 3 and 10
    nbrs = neigh.fit(deltas)
    distances, indices = nbrs.kneighbors(deltas)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)

    kn = KneeLocator(range(1, len(distances)+1), distances, curve='convex', direction='increasing')

    return float(distances[kn.knee])


def DBSCAN_clustering(deltas):
    """Function to cluster a list of one-dimensional elements with the help of the DBSCAN clustering algorithm.

    Args:
        deltas (list): list of one-dimensional elements (e.g. of time differences between the posting hour of a source tweet
                       and the posting hours of the reactions - replies and retweets)

    Returns:
        dict: dictionary with keys as auto-generated labels of the clusters and values as lists of elements in each cluster
    """    
    deltas = np.array(deltas)
    deltas = deltas.reshape(-1, 1)

    # # Compute the value for epsilon using the function defined previously
    # epsilon = epsilon_DBSCAN(deltas)
    epsilon = 0.3

    # Create a DBSCAN object with epsilon as the computed value and minimum samples=5
    dbscan = DBSCAN(eps=epsilon, min_samples=5)

    # Fit the DBSCAN object to the data
    dbscan.fit(deltas)

    # Get the labels assigned to each data point
    labels = dbscan.labels_

    # # Print the number of clusters and the labels assigned to each data point
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # print("Number of clusters:", n_clusters)
    # print("Labels:", labels)

    # Create a dictionary to store the clusters
    clusters_dict = {}
    for i, label in enumerate(labels):
        if label in clusters_dict:
            clusters_dict[label].append(deltas[i][0])
        else:
            clusters_dict[label] = [deltas[i][0]]
    
    return clusters_dict


def clusters_all_events(events_path, deltas_type, clustering_algorithm):
    """Function that generates clusters for all events, using the clustering algorithm and the time unit
    for differences between the posting times of source tweets and their respective reactions' (replies and retweets) posting times,
    both given as input parameters.

    Args:
        events_path (str): path to the root folder of the events (e.g. /your/path/to/threads/en)
        deltas_type (str): time unit for time differences (e.g. 'hours', 'minutes')
        clustering_algorithm (str): clustering algorithm used (e.g. 'k-means', 'DBSCAN')

    Returns:
        dict: dictionary with keys as event names and values as dictionaries with clusters for each event
    """    
    all_events_deltas = deltas_all_events(events_path, deltas_type)
    all_events_clusters = {}
    for event_name, event_deltas in all_events_deltas.items():
        if clustering_algorithm == 'k-means':
            # loop over maximum 10 clusters, as this is a range where you usually find the optimum number of clusters
            k = elbow_method_k_means_clusters(event_deltas, 10)
            # print(f"\nEvent {event_name}:")
            # print(f"NOTE: Numbers represent {deltas_type}\n")
            all_events_clusters[event_name] = k_means_clustering(k, event_deltas)
        elif clustering_algorithm == 'DBSCAN':
            all_events_clusters[event_name] = DBSCAN_clustering(event_deltas)
        
    return all_events_clusters


def clusters_for_plot(clusters, number_of_clusters):
    """Function that creates the most populated number_of_clusters (or less, if after completion, the number os clusters is smaller)
    clusters or subclusters obtained after repeated clusterization. More specifically, the original list of data points is clustered
    with either DBSCAN or K-Means. Then, we analyze the most populated cluster. If it contains more than 70% of all data points,
    the function performs another clusterization on the biggest cluster. This operation is repeated until the most populated
    cluster or subcluster gather no more than 70% of the data points.

    Args:
        clusters (dict): clusters of the original data points, obtained through either DBSCAN or K-Means
        number_of_clusters (int): number of the biggest clusters or subclusters we want to save

    Returns:
        dict: dictionary with keys as integers ranging from 0 to number_of_clusters and values as the biggest number_of_clusters clusters,
              starting with the biggest and decreasing
    """    
    total_length = 0
    for value in clusters.values():
            total_length += len(value)
    subclusters_for_plot = {}
    
    def subclusters(clusters):
            if(len(clusters)):
                key_for_biggest_cluster = max(clusters, key=lambda k: len(clusters[k]))
                if len(clusters[key_for_biggest_cluster]) / total_length < 0.7:
                    print("Reached base case.")
                    return clusters
                else:
                    # print("We'll go back into subclusters function.")
                    # print(f"Length of current clusters: {len(clusters)}")
                    # print("Current clusters look like this:")
                    # for key, value in clusters.items():
                    #      print(f"{key}: {math.floor(min(value))} - {math.ceil(max(value))}")

                    biggest_cluster = clusters[key_for_biggest_cluster]
                    # print("Biggest cluster:")
                    # print(biggest_cluster[:3])
                    
                    del clusters[key_for_biggest_cluster]
                    # print("Clusters after removing biggest_cluster:")
                    # for key, value in clusters.items():
                    #      print(f"{key}: {math.floor(min(value))} - {math.ceil(max(value))}")
                    
                    # K-MEANS CLUSTERING
                    k = elbow_method_k_means_clusters(biggest_cluster, 10)
                    # print(f"k = {k}")
                    
                    subcl = k_means_clustering(k, biggest_cluster)
                    # print("subcl - the subclusters dictionary obtained from the biggest_cluster looks like this:")
                    # for key, value in subcl.items():
                    #      print(f"{key}: {math.floor(min(value))} - {math.ceil(max(value))}")
                    
                    for key, value in subcl.items():
                        clusters[max(clusters.keys())+1] = value
                    
                    # print(f"Length of clusters after modifications: {len(clusters)}")
                    # print("Clusters look like this:")
                    # for key, value in clusters.items():
                    #      print(f"{key}: {math.floor(min(value))} - {math.ceil(max(value))}")
                    
                    subclusters(clusters)
            else:
                print("Clusters dictionary is empty.")
                return clusters
            

    subclusters(clusters)

    if len(clusters) >= number_of_clusters:
        for i in range(number_of_clusters):
            key_for_biggest_cluster = max(clusters, key=lambda k: len(clusters[k]))
            subclusters_for_plot[i] = clusters[key_for_biggest_cluster]
            del clusters[key_for_biggest_cluster]
    else:
        subclusters_for_plot = clusters
    
    return subclusters_for_plot


def plot_reaction_times(events_path, deltas_type):
    """Function that saves a plot of the most populated clusters or subclusters for each event, locally, with labels in increasing order,
    based on the posting hour of the earliest tweet in the cluster.

    Args:
        events_path (str): path to the root folder of the events (e.g. /your/path/to/threads/en)
        deltas_type (str): time unit for time differences (e.g. 'hours', 'minutes')
    """    
    global clustering_algorithm
    all_clusters = clusters_all_events(events_path, deltas_type, clustering_algorithm)

    for event_name, clusters in all_clusters.items():
        print(f"Event: {event_name}")
        clusters_to_be_plotted = clusters_for_plot(clusters, 5)

        final_clusters_for_plot = {}
        for key, value in clusters_to_be_plotted.items():
            final_clusters_for_plot[f"{math.floor(min(value))} - {math.ceil(max(value))}"] = len(value)

        intervals_unsorted = list(final_clusters_for_plot.keys())
        values_unsorted = list(final_clusters_for_plot.values())
        df_plot = pd.DataFrame(
            dict(
                Interval=intervals_unsorted,
                Value=values_unsorted
            )
        )
        
        df_plot['Start of interval'] = df_plot['Interval'].str.split(' - ').str[0]
        df_plot['Start of interval'] = df_plot['Start of interval'].apply(pd.to_numeric) 
        df_plot_sorted = df_plot.sort_values('Start of interval')
        intervals = list(df_plot_sorted['Interval'])
        values = list(df_plot_sorted['Value'])


        plt.bar(range(len(final_clusters_for_plot)), values, tick_label=intervals)
        # Rotate the x-axis labels by 45 degrees
        plt.xticks(rotation=45)
        plt.title(f'Distribution of reaction times in {deltas_type}')
        plt.xlabel(f'Reaction times (between x and y {deltas_type})')
        plt.ylabel('Number of reactions')
        plt.savefig(pheme_reaction_times_graphs + f"/{event_name}_reaction_times.png")
        plt.close()


def tweet_followers_count(tweet_path):
    """Function that gets the followers count of the author of a tweet

    Args:
        tweet_path (str): path to a JSON file associated with a tweet 
        (e.g. /your/path/to/charliehebdo/552783667052167168/source-tweets/552783667052167168)

    Returns:
        int: followers count of the user who posted the tweet
    """    
    with open(tweet_path) as f:
        tweet = json.load(f)
    
    return tweet['user']['followers_count']


def reactions_count_and_followers_count_for_source_tweet(story_path):
    """Function that generates the number of reactions (replies or retweets) of a single source tweet 
    (a story contains one source tweet).

    Args:
        story_path (str): path to the root folder of a story (e.g. /your/path/to/charliehebdo/552783667052167168)

    Returns:
        int: number of reactions for a source tweet, 
        int: followers counts of the user who posted the source tweet
    """  
    source_path = source_tweet_path(story_path)
    followers_count = tweet_followers_count(source_path)
    
    reactions_paths_list = reaction_tweets_paths(story_path)
    reactions_count = len(reactions_paths_list)

    retweets_path = story_path + "/retweets.json"
    if os.path.exists(retweets_path):
        format_retweets_json(retweets_path)
        with open(retweets_path, 'r') as file:
            retweets_list = json.load(file)
        reactions_count += len(retweets_list)
    
    return reactions_count, followers_count


def reaction_counts_influencers_avg_followers_count_per_tweet_event(event_path):
    """Function that generates the dictionary of reaction counts per tweet per hour.
     Based on this dictionary, a plot can be made.

    Args:
        event_path (str): path to the root folder of an event (e.g. /your/path/to/charliehebdo)

    Returns:
        dict: dictionary with keys as posting hours of source tweets and values as the number of reactions per tweet per key (hour),
        dict: dictionary with keys as posting hours of source tweets and values as the number of followers of the most followed user who
              posted at each hour (key),
        dict: dictionary with keys as posting hours of source tweets and values as the average number of followers of the users who
              posted at each hour (key)
    """    
    number_of_source_tweets_per_hour = {}
    reactions_count_per_hour = {}
    max_followers_count_per_hour = {}
    sum_followers_count_per_hour = {}
    for story_id in os.listdir(event_path):
        story_path = event_path + "/" + story_id
        source_path = source_tweet_path(story_path)
        source_hour = tweet_hour(source_path)

        if source_hour in reactions_count_per_hour:
            number_of_source_tweets_per_hour[source_hour] += 1
            reactions_count, followers_count = reactions_count_and_followers_count_for_source_tweet(story_path)
            reactions_count_per_hour[source_hour] += reactions_count
            max_followers_count_per_hour[source_hour] = max(max_followers_count_per_hour[source_hour], followers_count)
            sum_followers_count_per_hour[source_hour] += followers_count
        else:
            number_of_source_tweets_per_hour[source_hour] = 1
            reactions_count, followers_count = reactions_count_and_followers_count_for_source_tweet(story_path)
            reactions_count_per_hour[source_hour] = reactions_count
            max_followers_count_per_hour[source_hour] = followers_count
            sum_followers_count_per_hour[source_hour] = followers_count

    reactions_per_tweet = {}
    avg_followers_count_per_hour = {}
    for source_hour in number_of_source_tweets_per_hour:
        reactions_per_tweet[source_hour] = round(reactions_count_per_hour[source_hour] / number_of_source_tweets_per_hour[source_hour])
        avg_followers_count_per_hour[source_hour] = round(sum_followers_count_per_hour[source_hour] / number_of_source_tweets_per_hour[source_hour])

    # print(f"Event: {os.path.basename(event_path)}")
    # import pprint
    # pprint.pprint(number_of_source_tweets_per_hour)

    return reactions_per_tweet, max_followers_count_per_hour, avg_followers_count_per_hour

    # COMMENT THE FIRST RETURN STATEMENT AND UNCOMMENT THE SECOND TO RETURN A LIST WITH 2 DICTIONARIES:
    # - the reaction counts per tweet
    # - the absolute reaction counts
    # return [reactions_per_tweet, reactions_count_per_hour]


def add_labels_y_value(x,y):
    """Function that takes the x and y-axis to be passed onto a plot function and generates labels,
    such that on top of each y value, it is displayed centrally.

    Args:
        x (list): list of labels for x-axis of a plot
        y (list): list of values for y-axis of a plot
    """    
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')


def add_labels_percentage(x, y):
    """Function that takes the x and y-axis to be passed onto a plot function and generates labels,
    such that on top of each y value, the percentage of the y value out of the sum of all y-s is displayed as a red bounding box.
    Useful when y-s represent the counts of occurences for some values.

    Args:
        x (list): list of labels for x-axis of a plot
        y (list): list of values for y-axis of a plot
    """    
    for i in range(len(x)):
        percentage = y[i] / sum(y) * 100
        plt.text(i, y[i], f"{round(percentage, 1)}%", ha = 'center',
                 bbox = dict(facecolor = 'red', alpha =.7, pad=2))


def sorted_plot_labels_from_dict(dictionary):
        """Function that returns the sorted labels for a plot from a dictionary

        Args:
            dictionary (dict): dictionary ready to be plotted in th shape of a bar chart, 
            where the keys represent the x-axis and the values represent the y-axis

        Returns:
            tuple: the sorted lists of x and y-values for the plot
        """        
        # sort the dictionary by keys
        sorted_dict = sorted(dictionary.items())

        # extract the sorted keys and values
        sorted_x = [k for k, v in sorted_dict]
        sorted_y = [v for k, v in sorted_dict]

        return sorted_x, sorted_y


def plot_reaction_counts_influencers_avg_followers_count_per_tweet_event(event_path):
    """Function that creates and saves locally multiple plots for an event in the dataset. 
    The plots show:
     - the number of reaction counts (replies or retweets) per tweet posted at an hour indicated by the x-axis labels.
     - the followers count of the most followed user who posted a source tweet at an hour indicated by the x-axis labels.
     - the average followers count for source tweets posted at an hour indicated by the x-axis labels.

    Args:
        event_path (str): path to the root folder of an event (e.g. /your/path/to/charliehebdo)
    """    
    (reactions_per_tweet_dict, max_followers_count_per_hour_dict, avg_followers_count_per_hour_dict) = (
        reaction_counts_influencers_avg_followers_count_per_tweet_event(event_path)
    )

    # PROCESS REACTION COUNTS
    sorted_posting_hours, sorted_counts = sorted_plot_labels_from_dict(reactions_per_tweet_dict)

    plt.bar(range(len(reactions_per_tweet_dict)), sorted_counts, tick_label=sorted_posting_hours)
    # Rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    # calling the function to add value labels
    add_labels_y_value(sorted_posting_hours, sorted_counts)
    graph_type = 'per tweet'
    plt.title(f"Reaction counts {graph_type} - {os.path.basename(event_path)} event")
    plt.xlabel("Posting hour of source tweet")
    plt.ylabel(f"Number of reactions {graph_type}")
    plt.savefig(pheme_reaction_counts_graphs + f"/{os.path.basename(event_path)}_reaction_counts_{graph_type}.png")
    plt.close()

    # PROCESS INFLUENCERS
    sorted_posting_hours, sorted_follower_counts = sorted_plot_labels_from_dict(max_followers_count_per_hour_dict)

    plt.bar(range(len(max_followers_count_per_hour_dict)), sorted_follower_counts, tick_label=sorted_posting_hours)
    # Rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    plt.title(f"Most influential source tweets authors - {os.path.basename(event_path)} event")
    plt.xlabel("Posting hour of source tweet")
    plt.ylabel(f"Number of followers for most followed user")
    plt.savefig(pheme_influencers_graphs + f"/{os.path.basename(event_path)}_influencers.png")
    plt.close()

    # PROCESS AVERAGE FOLLOWERS COUNTS
    sorted_posting_hours, sorted_avg_follower_counts = sorted_plot_labels_from_dict(avg_followers_count_per_hour_dict)

    plt.bar(range(len(avg_followers_count_per_hour_dict)), sorted_avg_follower_counts, tick_label=sorted_posting_hours)
    # Rotate the x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    plt.title(f"Average followers counts - {os.path.basename(event_path)} event")
    plt.xlabel("Posting hour of source tweet")
    plt.ylabel(f"Average number of followers")
    plt.savefig(pheme_avg_followers_counts_graphs + f"/{os.path.basename(event_path)}_avg_followers_count.png")
    plt.close()


    # COMMENT THE FIRST PART OF THE FUNCTION AND UNCOMMENT THIS PART TO CREATE 2 GRAPHS:
    # - reaction counts per tweet per hour
    # - absolute reaction counts per hour
    # NOTE: you should also return a list in the reaction_counts_per_tweet_event() function 
    # (comment out the current return statement and uncomment the other)
    
    # [reactions_per_tweet_dict, reactions_count_per_hour_dict] = reaction_counts_per_tweet_event(event_path)
    # results = [reactions_per_tweet_dict, reactions_count_per_hour_dict]
    # for i in range(len(results)):
    #     dictionary = results[i]
    #     # sort the dictionary by keys
    #     sorted_reactions_per_tweet_dict = sorted(dictionary.items())

    #     # extract the sorted keys and values
    #     sorted_posting_hours = [k for k, v in sorted_reactions_per_tweet_dict]
    #     sorted_counts = [v for k, v in sorted_reactions_per_tweet_dict]

    #     plt.bar(range(len(reactions_per_tweet_dict)), sorted_counts, tick_label=sorted_posting_hours)
    #     # Rotate the x-axis labels by 45 degrees
    #     plt.xticks(rotation=45)
    #     # calling the function to add value labels
    #     add_labels_y_value(sorted_posting_hours, sorted_counts)
    #     graph_type = 'absolute'
    #     graph_type = 'per tweet' if i == 0 else graph_type
    #     plt.title(f"Reaction counts {graph_type} - {os.path.basename(event_path)} event")
    #     plt.xlabel("Posting hour of source tweet")
    #     plt.ylabel(f"Number of reactions {graph_type}")
    #     plt.savefig(pheme_reaction_counts_graphs + f"/{os.path.basename(event_path)}_reaction_counts_{graph_type}.png")
    #     plt.close()


def plot_reaction_counts_influencers_avg_followers_counts_per_tweet(events_path):
    """Function that saves all plots, corresponding to all events, locally.

    Args:
        events_path (str): path to the root folder of all events in the dataset(e.g. /your/path/to/threads/en)
    """    
    for event_name in os.listdir(events_path):
        event_path = events_path + "/" + event_name
        plot_reaction_counts_influencers_avg_followers_count_per_tweet_event(event_path)



def main():

    plot_time_distributions(events_path)

    deltas_types = ['hours', 'minutes']

    deltas_type = deltas_types[0]

    plot_reaction_times(events_path, deltas_type)

    plot_reaction_counts_influencers_avg_followers_counts_per_tweet(events_path)


if __name__ == "__main__":
    main()

