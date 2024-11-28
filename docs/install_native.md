# Native installation 

The native installation of the MELD graph enables more freedom to change the code and run your the scripts and notebook to train / evaluate your own classifier.

::::{tab-set}
:::{tab-item} Mac
:sync: mac
Native installation is required on any M chip Apple computer (mid-2021 onward).
:::

:::{tab-item} Linux
:sync: linux
Note: This installation has been tested on Ubuntu 18.04.5 with CUDA Version: 11.4. As a [docker container](https://meld-graph.readthedocs.io/en/latest/install_docker.html) is provided, the linux native installation won't be supported by the MELD team.
:::

:::{tab-item} Windows
Note: Windows native installation has not been tested. The MELD team highly recommends installing the [docker container](https://meld-graph.readthedocs.io/en/latest/install_docker.html).
:::
::::

Here is the video tutorial demonstrating how to do the native installation - [Native Installation of MELD Graph Tutorial](https://youtu.be/jUCahJ-AebM).

## Prerequisites
For preprocessing, MELD classifier requires Freesurfer. It is trained on data from versions 6 & v5.3, but compatible with Freesurfer **version up to V7.2**. You must already have a freesurfer `license.txt` that was obtained by [following the instructions on their wiki](https://surfer.nmr.mgh.harvard.edu/fswiki/License).

:::{warning}
MELD will not work on Freesurfer v7.3 and above
:::

::::{tab-set}
:::{tab-item} Mac
:sync: mac
1. Download **the `.pkg`** Freesurfer 7.2.0  version from the [Freesurfer downloads page](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads#A7.2.0release), or [download directly](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.2.0/freesurfer-darwin-macOS-7.2.0.pkg).

2. Follow the [mac install instructions](https://surfer.nmr.mgh.harvard.edu/fswiki//FS7_mac#Performingtheinstall) making sure it is installed in the default directory (`/Applications/freesurfer/7.2.0`).
:::

:::{tab-item} Linux
:sync: linux
Please follow instructions on [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) to install FreeSurfer. 

MELD pipeline also works with FastSurfer (quicker version of Freesurfer). If you wish to use FastSurfer instead please follow instructions for the [native install of Fastsurfer V1.1.2](https://github.com/Deep-MI/FastSurfer.git) by running the below:
```bash
git clone --branch v1.1.2 https://github.com/Deep-MI/FastSurfer.git
```
:::
::::

Note that Fastsurfer requires to install Freesurfer V7.2 to work 

You will need to ensure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running: 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.


## Conda installation
We use [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to manage the environment and dependencies. Please follow instructions on [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) to install Anaconda.

## Download the MELD classifier:
::::{tab-set}

:::{tab-item} Download

1. Go to the [github releases page](https://github.com/MELDProject/meld_graph/releases) and download the latest source zip or tar.
2. Extract the file
3. Copy your freesurfer `license.txt` to the meld directory
4. Highlight the extracted directory and press Command/Ctrl+C
5. Open a terminal and type `cd ` then press Command/Ctrl+V

Take note of the path - this is the path that should be used wherever `<path_to_meld_graph>` appears in the rest of these docs.
:::

:::{tab-item} Git
```bash
# checkout and install the github repo 
git clone https://github.com/MELDProject/meld_graph.git 

# enter the meld_graph directory
cd meld_graph

# copy freesurfer license.txt into the meld directory
cp $FREESURFER_HOME/license.txt ./
```
:::
::::

Then activate your environment by running the following:

::::{tab-set}
:::{tab-item} Mac
:sync: mac
```
./meldsetup.sh
```
:::
:::{tab-item} Linux
:sync: linux
```
# create the meld graph environment with all the dependencies 
conda env create -f environment.yml
# activate the environment
conda activate meld_graph
# install meld_graph with pip (with `-e`, the development mode, to allow changes in the code to be immediately visible in the installation)
pip install -e .
```
:::
::::

## Set up paths and download model
Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. For this, run:
```bash
./meldgraph.sh prepare_classifier.py
```

This script will ask you if you want to change the path to the data folder, answer **'y'** for yes. \
Then, it will ask for the the location of your **MELD data folder**, where you would like to store MRI data to run the classifier. Create the **MELD data folder**, if it doesn't exist, and provide the path. It will download the pretrained model and test data to a folder inside your MELD data folder


Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the call.

## Verify installation
We provide a test script to allow you to verify that you have installed all packages, set up paths correctly, and downloaded all data. This script will run the pipeline to predict the lesion classifier on a new patient. It takes approximately 15minutes to run.

```bash
cd <path_to_meld_graph>
./meldgraph.sh pytest
```
:::{warning}
If you run into errors at this stage and need help, you can re-run the command below to save the terminal outputs in a txt file, and send it to us. We can then work with you to solve any problems.

```bash
./meldgraph.sh pytest -s | tee pytest_errors.log
```
You will find this pytest_errors.log file in <path_to_meld_graph>. 
:::

## FAQs 
Please see our [FAQ page](https://meld-graph.readthedocs.io/en/latest/FAQs.html) for common installation problems and questions


