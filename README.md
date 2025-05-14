# occident

Occident is a tool for analyzing live-cell imaging data of cancer-vs-T cell in vitro screens. 

### Installing dependencies

To reproduce this code, we recommend intalling all dependencies into a miniforge conda environment using the instructions as follows.

```
conda create --name occident python=3.11.9
conda activate occident
pip install -r requirements.txt
```

Then install the `occident` package in editable mode

```
pip install -e .
```
This command installs `occident` as a development (editable) package, meaning that any changes you make to the code in the occident folder will be immediately reflected without needing to reinstall the package.

After installing these dependencies, you should be able to lauch jupyter from the `occident` conda environment and run the corresponding notebooks in this repository.


### General usage

All core functions are found within the `occident` package. This package is split into subpackages for different categories of analysis. Individual functions from these subpackages can be imported as long as the `occident` conda environment is activated, regardless of whether the python file is located in the `occident` folder or not. Here are some ways you can import occident:

```
from occident.utils import *
from occident.markov import get_box_wise_counts
import occident as oc
```

### Expected data format

Occident currently expects deepcell segmentation + tracking data as a zip file with the following items:

- `X.ome.tiff`: Input images
    - Raw intensities (generally in the range [0, 255]) of shape (channels, frames, height, width)
- `y.ome.tiff`: Segmentation mask
    - Cell mask of shape (cell_type, frames, height, width). The values in the arrays correspond to the cell_id, which are linked across frames. For T cell vs cancer cell co-cultures, it is standard that the 0 in the first dimension corresponds to T-cells and the 1 in the first dimension corresponds to cancer cells.
- `cells.json`: List of cell dictionaries (not used by occident)
- `divisions.json`: List of division dictionaries (not used by occident)

Cell division events are provided to occident functions through a separate `div.pkl` file for each well. The `div.pkl` file is a pandas dataframe with columns:
- parent: Parent cell id
- daugher_1: First daughter cell id
- daughter_2: Second daughter cell id
- frame: Frame of the cell division event
Each row corresponds to a unique cell division event. The cell ids and frame numbers correspond to those specified in the `y.ome.tiff` cell mask.

### Citing occident

If you use occident, please cite the following paper: https://doi.org/10.1101/2024.11.19.624390. Code used to generate this paper is available at https://github.com/bee-hive/Occident-Paper and https://github.com/vanvalenlab/Caliban-2024_Schwartz_et_al.