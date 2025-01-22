# Singularity container

**WARNING: Installation and use not yet tested. Please do let us know if you are succeeding / failing to use the singularity container on HPC**

The Singularity container has been created to be used on HPC supporting Linux as they do not work with Docker container. If you are not working on a HPC, we recommend to install the docker version of container. 

Notes: 
- The Singularity image is built from the Docker container. 
- You will need **~20GB of space** to install the container
- The image contains Miniconda 3, Freesurfer V7.2, Fastsurfer V1.1.2 and torch 1.10.0+cu111. The whole image is 13.5 GB.  

## Prerequisites

### Install Singularity
You will need to have Singularity installed. Most of the HPC will already have Singularity installed or Apptainer. You can check if Singularity/Apptainer is installed on your computer by running:
```bash
singularity --version
```
If this command displays the singularity or apptainer version already installed. If not, please follow the [guidelines](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) to install singularity on your machine.


## Freesurfer licence
You will need to download a Freesurfer license.txt to enable Freesurfer/Fastsurfer to perform the segmentation. Please follow the [guidelines](https://surfer.nmr.mgh.harvard.edu/fswiki/License) to download the file and keep a record of the path where you saved it. 

## Configuration
In order to run the singularity image, you'll need to build the singularity image from the meld_graph docker image. This will create a singularity image called meld_graph.sif where you ran the command. 

Make sure you have 20GB of storage space available for the docker

```bash
singularity build meld_graph.sif docker://meldproject/meld_graph:latest 
```

## Set up paths and download model
Before being able to use the classifier on your data, data paths need to be set up and the pretrained model needs to be downloaded. 

1. Make sure you have 2GB available for the meld data.
2. Create the **meld_data** folder, if it doesn't exist already. This folder is where where you would like to store MRI data to run the classifier. 
2. Run this command to set the paths needed:
-  <path_to_meld_data_folder> : Add the path to meld_data folder
- <path_to_FS_license>: path where the license.txt has been saved
```bash
export SINGULARITY_BINDPATH=/<path_to_meld_data_folder>:/data,<path_to_FS_license>/license.txt:/license.txt:ro
export SINGULARITYENV_FS_LICENSE=/license.txt
```
OR with Apptainer
```bash
export APPTAINER_BINDPATH=/<path_to_meld_data_folder>:/data,<path_to_FS_license>/license.txt:/license.txt:ro
export APPTAINERENV_FS_LICENSE=/license.txt
```

:::{admonition} Singularity
:class: tip
You can add those paths to your `~/.bashrc` file to ensure they are always activated when opening a new terminal. 
:::

3. Run this command to download the data folder 
```bash
singularity exec meld_graph.sif /bin/bash -c "cd /app && python scripts/new_patient_pipeline/prepare_classifier.py "
```
It will download the data in the meld_data folder you set up in step 2. 

## Verify installation
To verify that you have installed all packages, set up paths correctly, and downloaded all data, this verification script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15 minutes to run.

```bash
singularity exec meld_graph.sif /bin/bash -c "cd /app && pytest -s"
```

### Errors
If you run into errors at this stage and need help, you can re-run by changing the last line of the command by the command below to save the terminal outputs in a txt file. Please send `pytest_errors.log` to us so we can work with you to solve any problems. [How best to reach us.](#contact)

```bash
singularity exec meld_graph.sif /bin/bash -c "cd /app && pytest -s | tee pytest_errors.log"
```

You will find `pytest_errors.log` in the folder where you launched the command. 


## FAQs
Please see our [FAQ page](https://meld-graph.readthedocs.io/en/latest/FAQs.html) for common installation problems and questions

## Contact

If you encounter any errors, please contact the MELD team for support at `meld.study@gmail.com`
