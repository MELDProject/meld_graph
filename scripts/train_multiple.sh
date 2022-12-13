#! /bin/bash
# ARGS: path to folder containing configs
echo "starting experiments in" $1
# find fold folders containing stuff to start sequentially
for f in $(find $1 -name 'fold_*'); do
    # go over configs in order
    config_files=$(echo $(find $f -name 's_*.py') | sort)
    i=0
    for config_file in $config_files; do
        # start config and keep SID as dependency
        if [ $i -eq 0 ]; then
            RES=$(sbatch train.sh $config_file --parsable)
            SID=${RES##* }
            echo "started training for config" $config_file "with" $SID
        else
            RES=$(sbatch train.sh $config_file --parsable --dependency afterany:$SID)
            SID=${RES##* }
            echo "started training for config" $config_file "with" $SID "depending on previously started job"
        fi
        i=$((i+1))
    done
done