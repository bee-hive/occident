import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import groupby

def split(a):
    """
    Split an array into groups of consecutive equal elements.

    Parameters
    ----------
    a : array-like
        The array to split.

    Returns
    -------
    A list of arrays, each containing consecutive equal elements from the input array.
    """
    group = (n-i for i,n in enumerate(a))
    b     = [g for _,(*g,) in groupby(a,lambda _:next(group))]  
    return b
    
def compute_T_cell_density_vs_cancer_divisions(div_df, tcenters, cncenters, len_scale=np.sqrt(900), max_time=350):
    """
    Given a well, returns the average local T cell density for each cancer cell track and if the cell divided.
    
    Parameters:
    -----------
    dif_df (pandas.DataFrame): The division dataframe for a given well (*_div.pkl).
    tcenters (numpy.ndarray): The T cell centroids for a given well.
    cncenters (numpy.ndarray): The cancer cell nuclei centroids in a given well.
    len_scale (float): The length scale for the RBF kernel. Default is sqrt(900).
    max_time (int): The maximum time frame of the centroid arrays to use. Default is 350.
    
    Returns:
    -------
    x (list): The average local T cell density for each cancer cell track.
    y (list): If the cell divided.
    """

    # compute the distances between T cell centroids and cancer cell centroids
    full_distances = np.zeros((max_time, len(cncenters), len(tcenters)))
    for i in range(max_time):
        full_distances[i] = cdist(cncenters[:,i], tcenters[:,i])
        
    y = []
    x = []

    # compute the RBF kernel scores for each time point
    # the RBF kernel is defined as exp(-||x-y||^2 / len_scale^2)
    rbf_scores = np.zeros((max_time, len(cncenters)))
    for i in range(max_time):
        rbf_scores[i] = np.nansum(np.exp(-np.square(full_distances[i]/len_scale)), axis = 1)

    # for each cancer cell, find the time points where it exists and split them into consecutive time points
    # then compute the average RBF score for those time points
    # and check if the nucleus divided at that time point
    for i in range(len(cncenters)):
        exists = np.where(~np.isnan(cncenters[i,:,0]))[0]
        consecutives = split(exists)
    
        parent_df = div_df[div_df['parent'] == i + 1]
        for tlist in consecutives:
            x.append(np.nanmean(rbf_scores[tlist, i]))
            if len(parent_df) > 0:
                has_divide = parent_df['frame'].values[0] - 1 in tlist
            else:
                has_divide = 0
            y.append(has_divide)

    return x, y