# Native installation 

The native installation of the MELD graph enables more freedom to change the code and run your the scripts and notebook to train / evaluate your own classifier. 

Note: This installation have been tested on Ubuntu 18.04.5 with CUDA Version: 11.4. As a [docker container](https://meld-graph.readthedocs.io/en/latest/docs/install_docker.html) is provided, the native installation won't be supported by the MELD team. 

## Prerequisites
For preprocessing, MELD classifier requires Freesurfer. It is trained on data from versions 6 & v5.3, but compatible with Freesurfer **version up to V7.2**. Please follow instructions on [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) to install FreeSurfer. WARNING: MELD pipeline has not been adapted for Freesurfer V7.3 and above. Please install Freesurfer V7.2 instead. \

MELD pipeline is also working with FastSurfer (quicker version of Freesurfer). If you wish to use FastSurfer instead please follow instructions for the [native install of Fastsurfer V1.1.2](https://github.com/Deep-MI/FastSurfer.git) by running the below:
```bash
git clone --branch v1.1.2 https://github.com/Deep-MI/FastSurfer.git
```
Note that Fastsurfer requires to install Freesurfer V7.2 to works 

You will need to ensure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

## Conda installation
We use [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to manage the environment and dependencies. Please follow instructions on [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to install Anaconda.

Install MELD classifier and python dependencies:
```bash
# checkout and install the github repo 
git clone https://github.com/MELDProject/meld_graph.git 

# enter the meld_graph directory
cd meld_graph
# create the meld graph environment with all the dependencies 
conda env create -f environment.yml
# activate the environment
conda activate meld_graph
# install meld_graph with pip (with `-e`, the development mode, to allow changes in the code to be immediately visible in the installation)
pip install -e .
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
Please see our [FAQ](https://meld-graph.readthedocs.io/en/latest/docs/FAQs.html) for common installation problems.


