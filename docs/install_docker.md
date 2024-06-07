# Docker container

The Docker container has all the prerequisites embedded on it which makes it easier to install and compatible with most of the OS systems. 

Notes: 
- Currently only tested on **Linux and Windows** (HPC Singularity coming soon, Mac M chip computers have to do a {doc}`install_native`)
- You will need **~14GB of space** to install the container
- The docker image contains Miniconda 3, Freesurfer V7.2, Fastsurfer V1.1.2 and torch 1.10.0+cu111. The whole image is 13.5 GB.  

## Prerequisites

### Install Docker
You will need to have docker installed. You can check if docker is installed on your computer by running:
```bash
docker --version
```
If this command displays the docker version then it is already installed. If not, please follow the [guidelines](https://docs.docker.com/engine/install/) to install docker on your machine.

:::{admonition} Windows
:class: tip

On windows, Docker should be [using WSL2](https://docs.docker.com/desktop/wsl/).
:::


## Enable GPUs

Enabling your computer's GPUs for running the pipeline accelerates the brain segmentation when using Fastsurfer and the predictions. Follow instructions for your operating system to install.

::::{tab-set}

:::{tab-item} Linux
:sync: linux
Install the [*nvidia container toolkit*](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
:::

:::{tab-item} Windows
:sync: windows
Follow the instructions for [*enabling NVIDIA CUDA on WSL*](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl). If you have a recent NVIDIA driver CUDA should already be installed.
:::

::::


## Freesurfer licence
You will need to download a Freesurfer license.txt to enable Freesurfer/Fastsurfer to perform the segmentation. Please follow the [guidelines](https://surfer.nmr.mgh.harvard.edu/fswiki/License) to download the file and keep a record of the path where you saved it. 

## Configuration
In order to run the docker, you'll need to configure a couple of files

1. Download `meld_graph.zip` from the [latest github release](https://github.com/MELDProject/meld_graph/releases/latest) and extract it.
2. Copy the freesurfer `license.txt` into the extracted folder
3. (optional) Create the meld_data folder, if it doesn't exist already, and edit the compose.yml `volumes` line before the `:` to point to it. For example, if you wanted the folder to be on a mounted drive in Linux it might be:
```
    volumes:
      - /mnt/datadrive/meld-data:/data
```

:::{admonition} Windows
:class: tip

On windows, if you're using absolute paths, use forward slashes and quotes:
```
    volumes:
      - "C:/Users/John/Desktop/meld-data:/data"
```
:::

## Set up paths and download model
Before being able to use the classifier on your data, data paths need to be set up and the pretrained model needs to be downloaded. 

1. Make sure you have 15GB of storage space available for the docker, and 2GB available for the meld data.

2. Run this command to download the docker image and the training data

```bash
docker-compose run meld_graph python scripts/new_patient_pipeline/prepare_classifier.py
```

:::{note}
Append `--skip-download-data` to the python call to skip downloading the test data.
:::

3. This script will ask you if you want to change the location for the MELD data folder, say **"N"** for no and wait until the download is finished.


## Verify installation
To verify that you have installed all packages, set up paths correctly, and downloaded all data, this verification script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15 minutes to run.

```bash
docker-compose run meld_graph pytest
```

### Errors
If you run into errors at this stage and need help, you can re-run by changing the last line of the command by the command below to save the terminal outputs in a txt file. Please send `pytest_errors.log` to us so we can work with you to solve any problems. [How best to reach us.](#contact)

::::{tab-set}

:::{tab-item} Linux
:sync: linux
```bash
docker-compose run meld_graph pytest -s | tee pytest_errors.log
```
:::

:::{tab-item} Windows
:sync: windows
```bash
docker-compose run meld_graph pytest -s | tee -filepath ./pytest_errors.log
```
:::

::::

You will find `pytest_errors.log` in the folder where you launched the command. 

## Test GPU

You can test that the pipeline is working well with your GPU by changing `count` to `all` in the `compose.yml` file. The `deploy` section should look like this to enable gpus:

```
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
          count: all
```

To disable gpus, change it back to `0`.

## FAQs
Please see our {doc}`FAQs` for common installation problems.

## Contact

If you encounter any errors, please contact the MELD team for support at `meld.study@gmail.com`
