import numpy as np

def centroid_calculation(dcl_ob, type='cancer', lower_lim=0, upper_lim=350):
    """
    Calculate the centroids of cells in a deepcell object.

    Args:
        dcl_ob (dict): A deepcell object containing the segmentation data.
            It should have the following keys:
            - 'X': Input image as a numpy array
            - 'y': Output segmentation as a numpy array
            - 'divisions': List of division dictionaries
            - 'cells': List of cell dictionaries
        type (str): The type of centroid array to calculate. Must be either 'cancer', 'T' or 'nuclei'. Default is 'cancer'.
        lower_lim (int): The lower limit of the time frames to consider. Default is 0.
        upper_lim (int): The upper limit of the time frames to consider. Default is 350.
    Returns:
        centroids (numpy.ndarray): A 3D array of shape (n_cells, n_time_fames, 2) containing the x and y coordinates of the centroids.
    """
    assert type in ['cancer', 'T', 'nuclei'], "Type must be either 'cancer', 'T' or 'nuclei'."
    if type == 'cancer':
        ts = dcl_ob['y'][1][lower_lim:upper_lim,:,0,:]
    elif type == 'T':
        ts = dcl_ob['y'][0][lower_lim:upper_lim,:,0,:]
    elif type == 'nuclei':
        ts = dcl_ob['y'][lower_lim:upper_lim,:,:,0]
    cells = np.unique(ts)[1:]

    # create an empty array to hold the centroids
    # the first dimension is the cell id, the second dimension is the time frame,
    # and the third dimension is the x and y coordinates
    centroids = np.empty((len(cells), upper_lim - lower_lim, 2))
    centroids[:] = np.nan
    # Loop through each cell and calculate the centroid at each time frame
    for i in range(len(cells)):
        if i % 1000 == 0:
            print(i)
        cell_t, cell_x, cell_y = np.where(ts == cells[i])
        for t in np.unique(cell_t):
            mask = np.where(cell_t == t)
            x_mean = np.mean(cell_x[mask])
            y_mean = np.mean(cell_y[mask])
            centroids[i, t, 0] = x_mean
            centroids[i, t, 1] = y_mean
    
    return centroids