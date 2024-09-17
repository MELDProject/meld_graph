#!/bin/bash

#source FreeSurfer 
source $FREESURFER_HOME/FreeSurferEnv.sh
# Run your application
# exec gosu myuser python "scripts/new_patient_pipeline/$@"
# python scripts/new_patient_pipeline/$@
$@
