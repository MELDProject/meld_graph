# Docker container

The Docker container has all the prerequisites embedded on it which makes it easier to install and compatible with most of the OS systems. 

Notes: 
- Currently only tested on Linux
- The docker image contains Miniconda 3, Freesurfer V7.2, Fastsurfer V1.1.2 and torch 1.10.0+cu111. The whole image is 13.5 GB.  
- You will need **~14GB of space** to install the container
- Docker does not work on HPC, a singularity container is coming for that. 

## Prerequisites

### Install Docker
You will need to have docker installed. You can check if docker is installed on your computer by running:
```bash
docker
```
If this command display the instruction for docker, then it is already installed. If not, please follow the [guidelines](https://docs.docker.com/engine/install/) to install docker on your machine.


## Enable GPUs
If your computer has GPUs and you wish to use them to run the pipeline, you will need to get the [*nvidia container toolkit*](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Enabling the GPUs accelerate the brain segmentation when using Fastsurfer and the predictions. 


### Freesurfer licence
You will need to download a Freesurfer license.txt to enable Freesurfer/Fastsurfer to perform the segmentation. Please follow the [guidelines](https://surfer.nmr.mgh.harvard.edu/fswiki/License) to download the file and keep a record of the path where you saved it. 

## Pull the docker image

Pull the container, it will take time and 15GB of storage space. This will need to be run only once

```bash
docker pull mathrip/meld_graph:latest
```

Run the pipeline command to print the help
 
```bash
docker run -it --rm \
  mathrip/meld_graph:latest \
  python scripts/new_patient_pipeline/new_pt_pipeline.py -h 
```

If you encounter any error, please contact the MELD team for support at `meld.study@gmail.com`

## Set up paths and download model
Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. 

First, create the <meld_data> folder where you want to download the meld data structure and save the outputs

Then run:

```bash
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v <meld_data>:/data \
    mathrip/meld_graph:latest \
    python scripts/new_patient_pipeline/prepare_classifier.py
```
With `<meld_data>` being the path to where your meld data folder is stored.

This script will ask you if you want to change the location for the MELD data folder, say **"N"** for no and wait until the downloading is finished.

Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the python call.

## Verify installation
We provide a test script to allow you to verify that you have installed all packages, set up paths correctly, and downloaded all data. This script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15minutes to run.

```bash
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v <meld_data>:/data \
    -v <freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \
    mathrip/meld_graph:latest \
    pytest
```
With `<meld_data>` being the path to where your meld data folder is stored and <freesurfer_license> for the path to where you have stored the license.txt from Freesurfer. See [installation](https://meld-graph.readthedocs.io/en/latest/install_docker.html) for more details

Note: If you run into errors at this stage and need help, you can re-run by changing the last line of the command by the command below to save the terminal outputs in a txt file, and send it to us. We can then work with you to solve any problems.
  ```bash
  pytest -s | tee pytest_errors.log
  ```
  You will find this pytest_errors.log in the folder where you launched the command. 

## Test GPU

You can test that the pipeline is working well with your GPU by running the same command and adding the flag `--gpus all`

Example:
```bash
docker run -it --rm \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    -v <meld_data>:/data \
    -v <freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \
    mathrip/meld_graph:latest \
    pytest
```

## FAQs
Please see our [FAQ](https:/meld-graph.readthedocs.io/en/latest/FAQs.html) for common installation problems.

