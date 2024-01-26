# Compute the harmonisation parameters for a new scanner

## Information about the harmonisation
The MELD pipeline enables the harmonisation of your patient's features before prediction, if you are providing harmonisation parameters.

Harmonisation of your patient data is not mandatory but recommended, to remove any bias induced by the scanner and sequence used. For more details on the MELD FCD predictions performances with and without harmonisation please refers to our (paper)[]

## Compute the harmonisation paramaters 

The harmonisation parameters are computed using [Distributed Combat](https://doi.org/10.1016/j.neuroimage.2021.118822).
To get these parameters you will need a cohort of subjects acquired from the same scanner and under the same protocol (sequence, parameters, ...).
Subjects can be controls and/or patients, but we advise to use ***at least 20 subjects*** to enable an accurate harmonisation. 
Try to ensure the data are high quality (i.e no blurring, no artefacts, no cavities in the brain).
Demographic information (e.g age and sex) will be required for this process.

Once you have done the process once, you can follow the [general guidelines to predict on a new patient](https:/meld-graph.readthedocs.io/en/latest/docs/run_prediction_pipeline.md) 

## Before running

- Ensure you have installed the MELD pipeline with [docker container](https:/meld-graph.readthedocs.io/en/latest/docs/install_docker.md) or [native installation](https:/meld-graph.readthedocs.io/en/latest/docs/install_native.md). 
- **Chose a harmonisation** code for this scanner starting by 'H' (e.g H1, H2, ..). This harmonisation code will be needed to organise your data and run the code as detailled below. 
- Ensure you have [organised your MRI data](https:/meld-graph.readthedocs.io/en/latest/docs/prepare_data.md#prepare-the-mri-data-mandatory) and [provided demographic information](https:/meld-graph.readthedocs.io/en/latest/docs/prepare_data.md#prepare-the-demographic-information-required-only-to-compute-the-harmonisation-parameters) before running this pipeline. 


## Run with Docker 

When running with Docker container, you just need to run the following command

```bash
docker run -it \
    --rm --gpus all --user "$(id -u):$(id -g)" \
    -v <path_to_meld_data>:/data \
    -v <path_to_freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \
    meld_graph new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
```
With <path_to_meld_data> being the path to where your meld data folder is stored, and <path_to_freesurfer_license> the path to where you have stored the license.txt from Freesurfer. See [installation](https:/meld-graph.readthedocs.io/en/latest/docs/install_docker.md) for more details

The first 5th lines are arguments describing the docker. The last line is calling the MELD pipeline command. You can tune this command using the variables and flag describes further below. 

Note: This command will segment the brain using Freesurfer, extract the features and compute the harmonisation parameters, for the subjects provided in the subjects list. If you wish to also get the predictions on these subjects you can remove the flag '--harmo_only'. 

## Run with native installation

When running with native installation, you will need to first ensure the following:
- 1. You have activate the meld_graph environment : 
```bash
conda activate meld_graph
```
- 2. Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

NOTES: MELD pipeline has only been tested and validated on Freesurfer up to V7.2. Please do not use higher version than V7.2.0 \

**Main pipeline command**
```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -id <subject_id> 
```
You can tune this command using the variables and flag describes further bellow. 


### Second step : Run the pipeline to get the harmonisation parameters

You will need to make sure you are in the folder containing the MELD classifier scripts
```bash
  cd <path_to_meld_classifier_folder>
```

To compute the harmonisation parameters for your new site you can use the pipeline below:

```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
```

Note: This command will segment the brain using Freesurfer, extract the features and compute the harmonisation parameters, for the subjects provided in the subjects list. If you wish to also get the predictions on these subjects you can remove the flag '--harmo_only'. 

## Tune the command

You can tune this command using additional variables and flags as detailed bellow:

| **Mandatory variables**         |  Comment | 
|-------|---|
|```-harmo_code <harmo_code>```  |  The site code should start with H, e.g. H1. | 
|```-ids <subjects_list>``` |  A text file containing the list of subjects. An example 'subjects_list.txt' is provided in the <meld_data_folder>. | 
|```-demos <demographic_file>```| The name of the csv file containing the demographic information as detailled in the [guidelines](https:/meld-graph.readthedocs.io/en/latest/docs/prepare_data.md#prepare-the-demographic-information-required-only-to-compute-the-harmonisation-parameters). An example 'demographics_file.csv' is provided in the <meld_data_folder>.|
| **Optional variables** |
|```--parallelise``` | use this flag to speed up the segmentation by running Freesurfer/FastSurfer on multiple subjects in parallel. |
|```--fastsurfer``` | use this flag to use FastSurfer instead of Freesurfer. Requires FastSurfer installed. |
|```--harmo_only``` | Use this flag to do all the processes up to the harmonisation. Usefull if you want to harmonise on some subjects but do not wish to predict on them |


## What's next ? 
Once you have successfully computed the harmonisation parameters, they should be saved in your <meld_data_folder>. The file is called 'MELD_<site_code>_combat_parameters.hdf5' and is stored in 'output/preprocessed_surf_data/MELD_<site_code>/'.
You can now refer to the guidelines [to predict a new patient](https:/meld-graph.readthedocs.io/en/latest/docs/run_prediction_pipeline.md) to predict lesion in patients from that same scanner.
