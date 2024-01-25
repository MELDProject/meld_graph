#!/bin/bash

# Activate the Conda environment
source activate meld_graph

# Run your application
# exec gosu myuser python "scripts/new_patient_pipeline/$@"
# python scripts/new_patient_pipeline/$@
$@