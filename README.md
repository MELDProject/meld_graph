# meld_classifier_GDL
MELD classifier using Geometric Deep Learning

In progress

## Installation on CPU, see below for Cambridge HPC
dependencies: pytorch, pytorch_geometric, meld_classifier

Below installation on HPC(linux) and on MacOS environments are described.

### On the HPC (GPU)
On Ampere GPUs cuda 11.4 is enabled by default, for which no pytorch version is available. Installation here is done with cuda 11.1. and pytorch 1.10.0 (important!). After installation, running code also seems to work without loading the cuda/11.1 module.

- create an interactive GPU session
`sintr -A CORE-WCHN-MELD-SL2-GPU -N1 -n1 -t 0:59:0 -p ampere --qos=INTR --gres=gpu:1`
- make sure that cuda 11.1 is loaded: `module load cuda/11.1`
- activate conda is activated `module load miniconda/3`
- install the hpc environment
`conda env create -f environment_hpc.yml`
- install pytorch-geometric using pip:
    ```
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
    pip install torch-geometric
    ```
- install meld classifier and meld graph with `pip install -e .` in the respective directories.
- test if pytorch works with `python -c "import torch; torch.cuda.get_device_name(0)` -> should return a GPU name, and no other warnings/errors
- test if everything works in scripts by running `python test_model.py`


Original steps for installing pytorch and pytorch geometric from Hannah were: (with cuda 11.1 loaded)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install pytorch=1.10.0 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
```


### CPU environment on HPC:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pyg -c pyg -c conda-forge
```

### On a mac (CPU only)
- create the meld_graph environment from the `environment.yml`: `conda env create -f environment.yml`
- install pytorch-geometric using pip wheels as described [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): 
    This assumes you have torch 1.10.2 installed. Test this running: `python -c "import torch; print(torch.__version__)"`
    ```
    pip install torch-scatter -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-sparse -f https://data.pyg.org/whl/torch-10.1.2+cpu.html
    pip install torch-geometric
    ```
    Test if the torch-geometric installation worked by importing some packages: `from torch_geometric.data import Data`
