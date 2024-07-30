#!/bin/bash

# create the meld graph environment with all the dependencies 
conda env create -f environment-mac.yml
# activate the environment
eval "$(conda shell.bash hook)"
conda activate meld_graph
# install meld_graph with pip (with `-e`, the development mode, to allow changes in the code to be immediately visible in the installation)
pip install -e .
