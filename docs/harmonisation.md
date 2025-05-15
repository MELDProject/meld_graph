# Compute the harmonisation parameters for a new scanner

## Information about the harmonisation
Each MRI scanner / sequence  / FreeSurfer version will introduce small non-biological differences in the features that are calculated and used to predict where the FCD is. To help remove these biases, we advise harmonisation of your patient's features. This will make your patient's features "look like" the features we used to train the classifier. 

Harmonisation of your patient data is not mandatory but recommended, to remove any bias induced by the scanner and sequence used. For more details on the MELD FCD predictions performances with and without harmonisation please refer to our [paper](https://jamanetwork.com/journals/jamaneurology/fullarticle/2830410)

Here is the video tutorial detailing how to compute the harmonisation parameters - [Harmonisation tutorial](https://youtu.be/te_TR6sA5sQ).

## Compute the harmonisation paramaters 

The harmonisation parameters are computed using [Distributed Combat](https://doi.org/10.1016/j.neuroimage.2021.118822).
To get these parameters you will need a cohort of subjects acquired from the same scanner and under the same protocol (sequence, parameters, ...).
Subjects can be controls and/or patients, but we advise to use ***at least 20 subjects*** to enable an accurate harmonisation (see (paper)[https://jamanetwork.com/journals/jamaneurology/fullarticle/2830410]). 
Try to ensure the data are high quality (i.e no blurring, no artefacts, no cavities in the brain).
Demographic information (e.g age and sex) will be required for this process. 
WARNING: zero variance in the demographics information (e.g. having the same age for all subjects) will lead to Combat failures or errors. 

Once you have done the process once, you can follow the [general guidelines to predict on a new patient](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html) 

## Running

- Ensure you have installed the MELD pipeline with [docker container](https://meld-graph.readthedocs.io/en/latest/install_docker.html) or [native installation](https://meld-graph.readthedocs.io/en/latest/install_native.html). 
- **Chose a harmonisation** code for this scanner starting by 'H' (e.g H1, H2, ..). This harmonisation code will be needed to organise your data and run the code as detailled below. 
- Ensure you have [organised your MRI data](https://meld-graph.readthedocs.io/en/latest/prepare_data.html#prepare-the-mri-data-mandatory) and [provided demographic information](https://meld-graph.readthedocs.io/en/latest/prepare_data.html#prepare-the-demographic-information-required-only-to-compute-the-harmonisation-parameters) before running this pipeline. 


### Second step : Run the pipeline to get the harmonisation parameters


::::{tab-set}
:::{tab-item} Docker
:sync: docker
Open a terminal and `cd` to where you extracted the release zip.

```bash
DOCKER_USER="$(id -u):$(id -g)" docker compose run meld_graph python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
```
WINDOWS USER: 
- On windows, you do not need the `DOCKER_USER="$(id -u):$(id -g)"` part
- Make sure that the extensions '.txt' or '.csv' are in the files name even if they do not appear with the extension in your Windows folder 
For example:\
```bash
docker compose run meld_graph python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code H1 -ids list_subjects.txt -demos demographics_file.csv --harmo_only
```
:::

:::{tab-item} Native
:sync: native
Open a terminal and `cd` to the meld graph folder.

You will need to first activate FreeSurfer
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

Then run: 

```bash
./meldgraph.sh new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
```

:::
:::{tab-item} Singularity
:sync: singularity

If using Singularity or Apptainer, there are some paths that you need to export before running the pipeline. Find the paths to export in the [singularity installation](https://meld-graph.readthedocs.io/en/latest/install_singularity.html). Tip : You can add those paths to your `~/.bashrc` file to ensure they are always activated when opening a new terminal. 

And then run:
```bash
singularity exec meld_graph.sif /bin/bash -c "cd /app && source \$FREESURFER_HOME/FreeSurferEnv.sh && python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only"
```

:::
::::

This calls the MELD pipeline command. You can tune this command using the variables and flag describes further below. 

Note: This command will segment the brain using Freesurfer, extract the features and compute the harmonisation parameters, for the subjects provided in the subjects list. If you wish to also get the predictions on these subjects you can remove the flag '--harmo_only'. 

## Tune the command

You can tune this command using additional variables and flags as detailed bellow:

| **Mandatory variables**         |  Comment | 
|-------|---|
|```-harmo_code <harmo_code>```  |  The site code should start with H, e.g. H1. | 
|```-ids <subjects_list>``` |  A text file containing the list of subjects. An example 'subjects_list.txt' is provided in the <meld_data_folder>. | 
|```-demos <demographic_file>```| The name of the csv file containing the demographic information as detailled in the [guidelines](https://meld-graph.readthedocs.io/en/latest/prepare_data.html#prepare-the-demographic-information-required-only-to-compute-the-harmonisation-parameters). An example 'demographics_file.csv' is provided in the <meld_data_folder>.|
| **Optional variables** |
|```--parallelise``` | use this flag to speed up the segmentation by running Freesurfer/FastSurfer on multiple subjects in parallel. |
|```--fastsurfer``` | use this flag to use FastSurfer instead of Freesurfer. Requires FastSurfer installed. |
|```--harmo_only``` | Use this flag to do all the processes up to the harmonisation. Usefull if you want to harmonise on some subjects but do not wish to predict on them |


## What's next ? 
Once you have successfully computed the harmonisation parameters, they should be saved in your <meld_data_folder>. The file is called 'MELD_<site_code>_combat_parameters.hdf5' and is stored in 'output/preprocessed_surf_data/MELD_<site_code>/'.
You can now refer to the guidelines [to predict a new patient](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html) to predict lesion in patients from that same scanner.

## FAQs 
Please see our [FAQ page](https://meld-graph.readthedocs.io/en/latest/FAQs.html) for common questions about the harmonisation