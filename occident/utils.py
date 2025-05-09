import json
import numpy as np
from io import BytesIO
import json
import skimage as sk
import zipfile
import scipy.stats

def load_deepcell_object(filepath, expand_dim=3):
    """
    Loads data from a deepcell object zip file containing:
    - X.ome.tiff: Input image
    - y.ome.tiff: Output segmentation
    - cells.json: List of cell dictionaries
    - divisions.json: List of division dictionaries
    
    Returns a dictionary containing:
    - X: Input image as a numpy array
    - y: Output segmentation as a numpy array
    - divisions: List of division dictionaries
    - cells: List of cell dictionaries
    """
    f = zipfile.ZipFile(filepath, 'r')
    # load the files from the zip file
    file_bytes = f.read("cells.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        cells = json.load(b)
    file_bytes = f.read("divisions.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        divisions = json.load(b)
    file_bytes = f.read("X.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        X = sk.io.imread(b, plugin="tifffile")
    file_bytes = f.read("y.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        y = sk.io.imread(b, plugin="tifffile")
    # if expand_dim is not None, expand the dimensions of X and y
    if expand_dim is not None:
        X = np.expand_dims(X, expand_dim)
        y = np.expand_dims(y, expand_dim)
     # return the deepcell object as a dictionary
    dcl_ob = {
        'X': X,
        'y': y,
        'divisions':divisions,
        'cells': cells}
    return dcl_ob

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate mean and confidence interval.

    Parameters
    ----------
    data : array-like
        The data to calculate the mean and confidence interval from.
    confidence : float, optional
        The confidence interval desired. Defaults to 0.95.

    Returns
    -------
    mean : float
        The mean of the data.
    lower : float
        The lower bound of the confidence interval.
    upper : float
        The upper bound of the confidence interval.
    """
    a = 1.0 * np.array(data)
    n = np.count_nonzero(~np.isnan(a))  # Count non-NaN entries
    if n == 0:
        return np.nan, np.nan, np.nan  # Return NaN if all are NaNs
    mean = np.nanmean(a)
    se = scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

def estimate_se(ci_lower, ci_upper):
    """
    Estimate the standard error (SE) from a given confidence interval.

    Parameters
    ----------
    ci_lower : float
        The lower bound of the confidence interval.
    ci_upper : float
        The upper bound of the confidence interval.

    Returns
    -------
    se : float
        The estimated standard error.

    Notes
    -----
    The calculation is based on the formula for the standard error of the mean:

        SE = (upper - lower) / (2 * 1.96)

    where upper and lower are the upper and lower bounds of the confidence interval.
    """
    return (ci_upper - ci_lower) / (2 * 1.96)