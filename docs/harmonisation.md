# Compute the harmonisation parameters for a new scanner

## Information about the harmonisation
MRI data from different MRI scanners looks subtly different. This means that feature measurements, e.g. cortical thickness measurements, differ depending on which MRI scanner a patient was scanned on. We harmonise features (using NeuroCombat) from new scanners to the MELD dataset to adust for site based differences. 

We advise new users to harmonise data from their MRI scanner to the MELD graph dataset. This is because when we compared running the MELD Graph algorithm on new data with and without harmonisation, sensitivity remained stable, 72% with and 70% without harmonisation. Whereas, specificity dropped from 56% with to 39% without harmonisation. For more details on how performance varies with and without harmonisation please refer to our (paper)[].

The harmonisation step only needs to be run once, and requires data from at least 20 subjects acquired on the same scanner as well as the age and sex of the 20 subjects.

Harmonisation of your patient data is not mandatory but recommended. 

## Compute the harmonisation paramaters 

The harmonisation parameters are computed using [Distributed Combat](https://doi.org/10.1016/j.neuroimage.2021.118822).
To get these parameters you will need a cohort of subjects acquired from the same scanner and under the same protocol (sequence, parameters, ...).
Subjects can be controls and/or patients, but we advise to use ***at least 20 subjects***. 
Try to ensure the data is high quality (i.e no blurring, no artefacts, no cavities in the brain).
You will need to know the age at scan and sex of each of the subjects. 

Once you have done the harmonisation process once, you can follow the [general guidelines to predict on a new patient](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html) 

## Running

- Ensure you have installed the MELD pipeline with [docker container](https://meld-graph.readthedocs.io/en/latest/install_docker.html) or [native installation](https://meld-graph.readthedocs.io/en/latest/install_native.html). 
- **Choose a harmonisation** code for this scanner starting by 'H' (e.g H1, H2, ..). This harmonisation code will be needed to organise your data and run the code. 
- [Organise your MRI data](https://meld-graph.readthedocs.io/en/latest/prepare_data.html#prepare-the-mri-data-mandatory) and [provide demographic information](https://meld-graph.readthedocs.io/en/latest/prepare_data.html#prepare-the-demographic-information-required-only-to-compute-the-harmonisation-parameters) before running this pipeline. 


### Second step : Run the pipeline to get the harmonisation parameters


::::{tab-set}
:::{tab-item} Docker
:sync: docker
Open a terminal and `cd` to where you extracted the release zip.

```bash
docker compose run meld_graph python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
```
:::
:::{tab-item} Native
:sync: native
Open a terminal and `cd` to the meld graph folder.

```bash
./meldgraph.sh new_pt_pipeline -harmo_code <harmo_code> -ids <subjects_list> -demos <demographic_file> --harmo_only
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
You can now refer to the guidelines [to predict a new patient](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html) to predict FCD lesions in patients scanned on that same scanner.
