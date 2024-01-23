# Run MELD classifier pipeline with native installation 

## Installation

### Prerequisites
For preprocessing, MELD classifier requires Freesurfer. It is trained on data from versions 6 & v5.3, but compatible with Freesurfer **version up to V7.2**. Please follow instructions on [Freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) to install FreeSurfer. \
MELD pipeline is now also working with FastSurfer (quicker version of Freesurfer). If you wish to use FastSurfer instead please follow instructions for the [native install of Fastsurfer](https://github.com/Deep-MI/FastSurfer.git). Note that Fastsurfer requires to install Freesurfer V7.2 to works \
WARNING: MELD pipeline has not been adapted for Freesurfer V7.3 and above. Please install Freesurfer V7.2 instead.

You will need to ensure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

### Conda installation
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

### Set up paths and download model
Before being able to use the classifier on your data, some paths need to be set up and the pretrained model needs to be downloaded. For this, run:
```bash
python scripts/prepare_classifier.py
```

This script will ask you for the location of your **MELD data folder** and download the pretrained model and test data to a folder inside your MELD data folder. Please provide the path to where you would like to store MRI data to run the classifier on.


Note: You can also skip the downloading of the test data. For this, append the option `--skip-download-data` to the python call.

### FAQs
Please see our [FAQ](FAQs.md) for common installation problems.

### Verify installation
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

## Organising your data

**NEED UPDATE**
You need to organise the MRI data for the patients you want to run the classifier on.

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient. 

The IDs should follow the same naming structure as before. i.e. MELD\_<harmo\_code>\_<scanner\_field>\_FCD\_000X

e.g.MELD\_H1\_3T\_FCD\_0001 

In each patient folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder. Please ensure 'T1' is in the file name.

Place the FLAIR nifti file into the FLAIR folder. Please ensure 'FLAIR' is in the file name.

![example](images/example_folder_structure.png)

## Use

### first step : checks
Before running the below pipeline, ensure that you have installed the MELD classifier and activate the meld_graph environment : 
```bash
conda activate meld_graph
```
Also you need to make sure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

NOTES: MELD pipeline has only been tested and validated on Freesurfer up to V7.2. Please do not use higher version than V7.2.0 \


### Second step
Go into the meld_classifier folder 
```bash
  cd <path_to_meld_classifier_folder>
```

### Overview new patient pipeline 

The pipeline is split into 3 main scripts as illustrated below and detailed in the next section. 
![pipeline_fig](images/tutorial_pipeline_fig.png)

The pipeline can be called using one unique command line. Example to run the whole pipeline on 1 subject:

```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code <harmo_code> -id <subject_id> 
```

You can tune this command using additional variables and flags as detailed bellow:

| **Mandatory variables**         |  Comment | 
|-------|---|
| ```-harmo_code <harmo_code>```  |  the harmonisation code code should start with H, e.g. H1. If you cannot remember your harmonisation code code - contact the MELD team. | 
|either ```-id <subject_id>```  |  if you want to run the pipeline on 1 single subject. Needs to be in MELD format MELD\_<harmo\_code>\_<scanner\_field>\_FCD\_000X |  
|or ```-ids <subjects_list>``` |  if you want to run the pipeline on more than 1 subject, you can pass the name of a text file containing the list of subjects. An example 'subjects_list.txt' is provided in the <meld_data_folder>. | 
| **Optional variables** |
|```--parallelise``` | use this flag to speed up the segmentation by running Freesurfer/FastSurfer on multiple subjects in parallel. |
|```--fastsurfer``` | use this flag to use FastSurfer instead of Freesurfer. Requires FastSurfer installed. |
|```--skip_segmentation``` | use this flag to skips the segmentation, features extraction and smoothing (processes from script1). Usefull if you already have these outputs and you just want to run the preprocessing and the predictions (e.g: after harmonisation) |
|```--harmo_only``` | use this flag to do all the processes up to the harmonisation. Useful if you want to harmonise on some subjects but do not wish to predict on them (see [Harmonisation_new_site.md](Harmonisation_new_site.md) guidelines) |
|**More advanced variables** | 
| ```--split``` | use this flag to split your list of subjects in smaller chunks to avoid data overload during prediction step. Useful if running more than 30 patients at a time. |
|```--no_nifti```| use this flag to run to all the processes up saving the predictions as surface vectors in the hdf5 file. Does not produce produce nifti and pdf outputs.|
|```--no_report``` | use this flag to do all the processes up to creating the prediction as a nifti file. Does not produce the pdf reports. |
|```--debug_mode``` | use this flag to print additional information to debug the code (e.g print the commands, print errors) |


NOTES: 
- you need to have set up your paths & organised your data before running this pipeline (see section **First step - Organising your data!**)
- We recommend using the same FreeSurfer/FastSurfer version that you used to process your patient's data that was used to train the classifier (existing site) / to get the harmonisation parameters (new site).
- Outputs of the pipeline (prediction back into the native nifti MRI and MELD reports) are stored in the folder ```output/predictions_reports/<subject_id>```. 

**Examples of use case**: 

To run the whole prediction pipeline on 1 subject using fastsurfer:
```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code H4 -id MELD_H4_3T_FCD_0001 --fastsurfer
```

To run the whole prediction pipeline on multiples subjects with parallelisation:
```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -harmo_code H4 -ids list_subjects.txt --parallelise
```

### Additional information about the 3 different scripts / steps

#### Script 1 - FreeSurfer reconstruction and smoothing

This script:
 1. Runs a FreeSurfer reconstruction on a participant
 2. Extracts surface-based features needed for the classifier:
    * Samples the features
    * Creates the registration to the template surface fsaverage_sym
    * Moves the features to the template surface
    * Write feature in hdf5
 3. Preprocess features: 
    * Smooth features and write in hdf5

To know more about the script and how to use it on its own:
```bash
python scripts/new_patient_pipeline/run_script_segmentation.py -h
```

#### Script 2 - Feature Preprocessing

This script : 
  1. Combat harmonise features and write into hdf5
  2. Normalise the smoothed features (intra-subject & inter-subject (by controls)) and write in hdf5
  3. Normalise the raw combat features (intra-subject, asymmetry and then inter-subject (by controls)) and write in hdf5

  Notes: 
  - Features need to have been extracted and smoothed using script 1. 
  - (optional): this script can also be called to harmonise your data for new harmo_code but will need to pass a file containing demographics information.

To know more about the script and how to use it on its own:
```bash
python scripts/new_patient_pipeline/run_script_preprocessing.py -h
```

#### Script 3 - Lesions prediction & MELD reports

This script : 
1. Run the MELD classifier and predict lesion on new subject
2. Register the prediction back into the native nifti MRI. Results are stored in output/predictions_reports/<subjec_id>/predictions.
3. Create MELD reports with predicted lesion location on inflated brain, on native MRI and associated saliencies. Reports are stored in output/predictions_reports/<subjec_id>/predictions/reports.

Notes: 
- Features need to have been processed using script 2 and Freesurfer outputs need to be available for each subject

To know more about the script and how to use it on its own:
```bash
python scripts/new_patient_pipeline/run_script_prediction.py -h
```

## Interpretation of results

Refer to our [guidelines](/documentation/Interpret_results.md) for details on how to read and interprete the MELD pipeline results