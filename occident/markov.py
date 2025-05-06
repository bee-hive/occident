import numpy as np

def get_box_wise_counts(tcenters, ccenters, boxsize=50, max_time=350, image_height=600):
    """
    Get the number of T cells and cancer cells in each box at each time point.
    Args:
        tcenters (numpy.ndarray): Array of T cell centroids across all time points. Corresponds to just one well.
        ccenters (numpy.ndarray): Array of cancer cell centroids across all time points. Corresponds to just one well.
        boxsize (int): The size of the boxes to divide the image into.
        max_time (int): The maximum time of the centroid arrays to use.
        image_height (int): The pixel height of the original image that we're dividing into boxes.
    Returns:
        t_box_array (numpy.ndarray): A 3D array of shape (max_time, n_boxes, 3) containing the counts of T cells.
        c_box_array (numpy.ndarray): A 3D array of shape (max_time, n_boxes, 3) containing the counts of cancer cells.
    """
    # compute the number of boxes in the image
    # the number of rows in the image is the number of pixels in the image divided by the box size
    row_width = np.floor(image_height/boxsize).astype(int)
    # the number of boxes is the number of rows squared (since the image is square)
    n_boxes = row_width ** 2

    # compute box ids for each row and column, creating separate arrays for T cells and cancer (C) cells
    trow_col = np.floor(tcenters/boxsize)
    tbox_ids = trow_col[:,:,0] * row_width + trow_col[:,:,1]
    crow_col = np.floor(ccenters/boxsize)
    cbox_ids = crow_col[:,:,0] * row_width + crow_col[:,:,1]

    # create a 3D array to hold the counts of T cells and cancer cells in each box at each time point
    # the first dimension is time, the second dimension is the box id, and the third dimension is type of T cell
    # 0 = already there, 1 = new T cell, 2 = from another box
    t_box_array = np.zeros((max_time, n_boxes, 3))
    # loop through all time points and boxes to count the number of T cells in each box
    for i in range(1, max_time):
        for box in range(n_boxes):
            ts_in_box_i = np.where(tbox_ids[:,i] == box)[0]
            if len(ts_in_box_i) == 0:
                continue
            for tcell in ts_in_box_i:
                if np.isnan(tbox_ids[tcell, i -1]):
                    t_box_array[i, box, 1] += 1
                elif tbox_ids[tcell, i-1] == box:
                    t_box_array[i, box, 0] += 1
                else:
                    t_box_array[i, box, 2] += 1
    for box in range(n_boxes):
        ts_in_box_0 = np.where(tbox_ids[:,0] == box)[0]
        for tcell in ts_in_box_0:
            t_box_array[0, box, 0] += 1

    # repeat the process for cancer cells
    c_box_array = np.zeros((max_time, n_boxes, 3))
    for i in range(1, max_time):
        for box in range(n_boxes):
            cs_in_box_i = np.where(cbox_ids[:,i] == box)[0]
            if len(cs_in_box_i) == 0:
                continue
            for ccell in cs_in_box_i:
                if np.isnan(cbox_ids[ccell, i -1]):
                    c_box_array[i, box, 1] += 1
                elif cbox_ids[ccell, i-1] == box:
                    c_box_array[i, box, 0] += 1
                else:
                    c_box_array[i, box, 2] += 1
    for box in range(n_boxes):
        cs_in_box_0 = np.where(cbox_ids[:,0] == box)[0]
        for ccell in cs_in_box_0:
            c_box_array[0, box, 0] += 1
    
    return t_box_array, c_box_array


def get_markov(t_boxes, c_boxes, tmax, cmax, t0=1, c0=1, max_time=350):
    """
    Get the Markov transition matrix and the number of T and cancer cells tracked in each box. By default, this function only computes
    the Markov transition matrix from the starting point of 1 T cell and 1 cancer cell. If you wish to compute the Markov transition matrix
    from a different starting point, you can specify the t0 and c0 parameters.

    Args:
        t_boxes (numpy.ndarray): A 3D array of shape (max_time, n_boxes, 3) containing the counts of T cells.
        c_boxes (numpy.ndarray): A 3D array of shape (max_time, n_boxes, 3) containing the counts of cancer cells.
        tmax (int): The maximum number of T cells in a box.
        cmax (int): The maximum number of cancer cells in a box.
        t0 (int): The initial number of T cells in a box.
        c0 (int): The initial number of cancer cells in a box.
        max_time (int): The final time point to use.
    Returns:
        tuple: A tuple containing the Markov transition matrix and the number of T and cancer cells tracked in each box.
    """
    # create a 2D array to hold the sum of T cells and cancer cells in each box at each time point
    # the first dimension is the cell type (T = 0, cancer = 1), the second dimension is time,
    # and the third dimension is the box id
    sum_box = np.zeros((2, t_boxes.shape[0], t_boxes.shape[1]))
    sum_box[0] = np.sum(t_boxes, axis = 2)
    sum_box[1] = np.sum(c_boxes, axis = 2)

    # create a markov transition matrix where the rows represent the number of T cells and the columns represent the number of cancer cells
    markov = np.zeros((tmax + 1, cmax + 1))
    is_tracked = np.zeros((tmax + 1, cmax + 1))
    # subset the sum_box array to only include the boxes with t0 T cells and c0 cancer cells
    ixes = np.where(np.logical_and(sum_box[0] == t0, sum_box[1] == c0))
    # loop through the boxes and time points to fill in the markov transition matrix
    for i in range(len(ixes[0])):
        # get the time frame and box id that this index corresponds to
        time = ixes[0][i]
        box = ixes[1][i]
        # if this is the last time point, skip it as we don't observe the next time piont (time + 1)
        if time == max_time - 1:
            continue
        # get the number of T cells and cancer cells in the next time point
        t1 = int(sum_box[0][time + 1, box])
        c1 = int(sum_box[1][time + 1, box])
        # add 1 to the markov transition matrix at the row,column corresponding to the number of T cells and cancer cells at the next time point
        markov[t1, c1] += 1
        is_tracked[t1, c1] += t_boxes[time + 1, box, 2]

    return markov, is_tracked

def norm_markov(markov):
    """
    Normalize the Markov transition matrix by dividing each element by the sum of the matrix.
    Args:
        markov (numpy.ndarray): The Markov transition matrix to normalize.
    Returns:
        numpy.ndarray: The normalized Markov transition matrix.
    """
    return markov / np.sum(markov)