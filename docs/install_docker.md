# Docker container

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
python scripts/prepare_classifier.py
```

This script will ask you for the location of your **MELD data folder** and download the pretrained model and test data to a folder inside your MELD data folder. Please provide the path to where you would like to store MRI data to run the classifier on.

Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the python call.

## Verify installation
We provide a test script to allow you to verify that you have installed all packages, set up paths correctly, and downloaded all data. This script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15minutes to run.

Note: Do not forget to activate Fressurfer as describe above before to run the test.

```bash
cd <path_to_meld_graph>
conda activate meld_graph
pytest
```
Note: If you run into errors at this stage and need help, you can re-run the command below to save the terminal outputs in a txt file, and send it to us. We can then work with you to solve any problems.
  ```bash
  pytest -s | tee pytest_errors.log
  ```
  You will find this pytest_errors.log file in <path_to_meld_graph>. 

## FAQs
Please see our [FAQ](FAQs.md) for common installation problems.

