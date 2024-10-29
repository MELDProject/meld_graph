#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate meld_graph

if [ $1 = 'pytest' ]; then
  pytest ${@:2}
else
  python scripts/new_patient_pipeline/$1 ${@:2}
fi
