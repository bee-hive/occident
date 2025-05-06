import json
import numpy as np
from io import BytesIO
import json
import skimage as sk
import zipfile

def load_deepcell_object(filepath):
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
    # return the deepcell object as a dictionary
    dcl_ob = {
        'X': np.expand_dims(X,3),
        'y': np.expand_dims(y,3),
        'divisions':divisions,
        'cells': cells}
    return dcl_ob