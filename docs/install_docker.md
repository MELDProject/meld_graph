# Docker container

The Docker container has all the prerequisites embedded on it which makes it easier to install and compatible with most of the OS systems. 

Notes: 
- Currently only tested on Linux
- You will need ~14GB of space to install the container
- Docker does not work on HPC, a singularity container is coming for that. 

## Prerequisites
You will need Docker (ADD INSTALLATION GUIDE)

The docker image uses Miniconda 3, Freesurfer V7.2, Fastsurfer V1.1.2 and torch 1.10.0+cu111. The whole image is 13 GB. 

You will need to download a Freesurfer license.txt by following [here](https://surfer.nmr.mgh.harvard.edu/fswiki/License)

## Pull the docker image

```bash

```

## Set up paths and download model
Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. For this, run:

```bash
docker run -it \
    --rm --gpus all --user "$(id -u):$(id -g)" \
    -v <path_to_meld_data>:/data \
    -v <path_to_freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \
    meld_graph \
    python scripts/prepare_classifier.py
```

With <path_to_meld_data> being the path to where your meld data folder is stored, and <path_to_freesurfer_license> the path to where you have stored the license.txt from Freesurfer. See [installation](https:/meld-graph.readthedocs.io/en/latest/install_docker.html) for more details

NOTE: This script will ask you if you want to change the location for the MELD data folder, say "N" for no. 

Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the python call.

## Verify installation
We provide a test script to allow you to verify that you have installed all packages, set up paths correctly, and downloaded all data. This script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15minutes to run.

```bash
docker run -it \
    --rm --gpus all --user "$(id -u):$(id -g)" \
    -v <path_to_meld_data>:/data \
    -v <path_to_freesurfer_license>:/license.txt:ro \
    -e FS_LICENSE='/license.txt' \
    meld_graph \
    pytest
```

Note: If you run into errors at this stage and need help, you can re-run by changing the last line of the command by the command below to save the terminal outputs in a txt file, and send it to us. We can then work with you to solve any problems.
  ```bash
  pytest -s | tee pytest_errors.log
  ```
  You will find this pytest_errors.log in the folder where you launched the command. 

## FAQs
Please see our [FAQ](https:/meld-graph.readthedocs.io/en/latest/FAQs.html) for common installation problems.

