#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate meld_graph
python scripts/new_patient_pipeline/$1.py ${@:2}