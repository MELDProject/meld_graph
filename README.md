# meld_classifier_GDL
MELD classifier using Geometric Deep Learning

In progress

## Installation
- create the meld_graph environment from the `environment.yml`: `conda env create -f environment.yml`
- install pytorch-geometric using pip wheels as described [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### On a mac (CPU only)
This assumes you have torch 10.1.2 installed. Test this running: `python -c "import torch; print(torch.__version__)"`
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
pip install torch-geometric
```

Test if the torch-geometric installation worked by importing some packages: `from torch_geometric.data import Data`