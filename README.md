# occident2

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

All core functions are found within the `occident` package. This package is split into subpackages for different categories of analysis. Individual functions from these subpackages can be imported as long as the `occident` conda environment is activated, regardless of whether the python file is located in the `occident` folder or not.

```
from occident.utils import *
from occident.markov import get_box_wise_counts
```