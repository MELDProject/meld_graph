# Docker container

The Docker container has all the prerequisites embedded on it which makes it easier to install and compatible with most of the OS systems. 

Notes: 
- Currently only tested on **Linux and Windows** (Mac and HPC Singularity coming soon)
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

## Pull the docker image

Make sure you have 15GB of storage space available and pull the docker image. This will need to be run only once.

```bash
docker pull mathrip/meld_graph:latest
```

:::{warning}
It can take a couple hours, so please leave it running.
:::

To test it's installed, run the pipeline command that prints the help
 
::::{tab-set}

:::{tab-item} Linux
:sync: linux
```bash
docker run -it --rm \
  mathrip/meld_graph:latest \
  python scripts/new_patient_pipeline/new_pt_pipeline.py -h 
```
:::

:::{tab-item} Windows
:sync: windows
```bash
docker run -it --rm `
  mathrip/meld_graph:latest `
  python scripts/new_patient_pipeline/new_pt_pipeline.py -h 
```
:::
::::

## Set up paths and download model
Before being able to use the classifier on your data, data paths need to be set up and the pretrained model needs to be downloaded. 

1. Create the `<meld_data>` folder where you want to download the meld data structure and save the outputs. It should typically take a couple GB of space.

2. Run:


::::{tab-set}

:::{tab-item} Linux
:sync: linux
```bash
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v <meld_data>:/data \
    mathrip/meld_graph:latest \
    python scripts/new_patient_pipeline/prepare_classifier.py
```
:::

:::{tab-item} Windows
:sync: windows
```bash
docker run -it --rm `
    -v <meld_data>:/data `
    mathrip/meld_graph:latest `
    python scripts/new_patient_pipeline/prepare_classifier.py
```

Windows docker uses unix style paths with named drives. E.g. `C:\Users\ada` becomes `/c/Users/ada`.
:::

::::

:::{note}
Append `--skip-download-data` to the python call to skip downloading the test data.
:::

In this command and the following ones, replace `<meld_data>` with the path to your meld data folder.

3. This script will ask you if you want to change the location for the MELD data folder, say **"N"** for no and wait until the downloading is finished.


## Verify installation
To verify that you have installed all packages, set up paths correctly, and downloaded all data, this verification script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15 minutes to run.

::::{tab-set}

:::{tab-item} Linux
:sync: linux
```bash
docker run -it --rm \
    --user "$(id -u):$(id -g)" \
    -v <meld_data>:/data \
    -v <freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \colon_fence
    mathrip/meld_graph:latest \
    pytest
```
:::

:::{tab-item} Windows
:sync: windows
```bash
docker run -it --rm `
    -v <meld_data>:/data `
    -v <freesurfer_license>:/license.txt:ro `
    -e FS_LICENSE='/license.txt' `
    mathrip/meld_graph:latest `
    pytest
```
:::
::::

In this command and the following ones, replace <freesurfer_license> with the path to where you have stored the license.txt from Freesurfer. See [the freesurfer licence section](#freesurfer-licence) for more details
colon_fence

### Errors
If you run into errors at this stage and need help, you can re-run by changing the last line of the command by the command below to save the terminal outputs in a txt file. Please send `pytest_errors.log` to us so we can work with you to solve any problems. [How best to reach us.](#contact)

::::{tab-set}

:::{tab-item} Linux
:sync: linux
```bash
pytest -s | tee pytest_errors.log
```
:::

:::{tab-item} Windows
:sync: windows
```bash
pytest -s | tee -filepath ./pytest_errors.log
```
:::

::::

You will find `pytest_errors.log` in the folder where you launched the command. 

## Test GPU

You can test that the pipeline is working well with your GPU by running the same command and adding the flag `--gpus all`

::::{tab-set}

:::{tab-item} Linux
:sync: linux
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
:::

:::{tab-item} Windows
:sync: windows
```bash
docker run -it --rm `
    --gpus all `
    -v <meld_data>:/data `
    -v <freesurfer_license>:/license.txt:ro `
    -e FS_LICENSE='/license.txt' `
    mathrip/meld_graph:latest `
    pytest
```
:::
::::

## FAQs
Please see our {doc}`FAQs` for common installation problems.

## Contact

If you encounter any errors, please contact the MELD team for support at `meld.study@gmail.com`
