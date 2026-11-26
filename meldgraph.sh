#!/bin/bash
# ensure conda command is available
source /Users/cll/opt/anaconda3/etc/profile.d/conda.sh #modified
conda activate meld_graph
export PYTHONPATH=/Users/cll/meld_graph:$PYTHONPATH #modified

if [ "$1" = "pytest" ]; then
  pytest ${@:2}
 else
  python scripts/new_patient_pipeline/"$1" "${@:2}"
fi

#!/bin/bash

# eval "$(conda shell.bash hook)"
# conda activate meld_graph

# if [ $1 = 'pytest' ]; then
#  pytest ${@:2}
# else
#  python scripts/new_patient_pipeline/$1 ${@:2}
# fi