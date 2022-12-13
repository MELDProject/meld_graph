#! /bin/bash
# ARGS: path to folder containing configs
echo "starting experiments in" $1
# find fold folders containing stuff to start sequentially
for f in $(find $1 -name 'fold_*'); do
    # go over configs in order
    config_files=$(ls $f/*.py | sort -n -t _ -k 2)
    i=0
    for config_file in $config_files; do
        # start config and keep SID as dependency
        if [ $i -eq 0 ]; then
            RES=$(sbatch train.sh $config_file)
            SID=${RES##* }
            echo "started training for config" $config_file "with" $SID
        else
            RES=$(sbatch --dependency afterany:$SID train.sh $config_file)
            SID=${RES##* }
            echo "started training for config" $config_file "with" $SID "depending on previously started job"
        fi
        i=$((i+1))
    done
done