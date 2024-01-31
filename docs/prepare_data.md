# Prepare the data

The MELD pipeline relies on the MRI data to be organised in the MELD or BIDS format. 

If you are preparing the data for the harmonisation step, you will also need to prepare the demographic information. 

## Prepare the MRI data (Mandatory)

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient with the id of the subject. 

### MELD format

In each subject folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder.

Place the FLAIR nifti file into the FLAIR folder.

![example](https://raw.githubusercontent.com//MELDProject/meld_graph/dev_docker/docs/images/input_structure_meld_format.png)

### BIDS format

The MELD pipeline now accept BIDS format as input data. For more information about BIDS format, please refers to their [instructions](https://bids.neuroimaging.io/)

The main key ingredients are : 
- each subject has a folder following the structure : `sub-<subject_id>`
- (optional) in each subject folder you can have a session folder, e.g. `ses-preop`. 
- in each session folder / subject folder you will need to have a datatype folder called `anat` folder. 
- in the anat folder your T1 and FLAIR nifti images should follow the structure : `sub-<subject_id>_<modality_suffix>.nii.gz` or `sub-<subject_id>_ses-<session>_<modality_suffix>.nii.gz` if you have a session.

A simple example of the BIDS structure for patient sub-test001 is given below:\
![example](https://raw.githubusercontent.com//MELDProject/meld_graph/dev_docker/docs/images/input_structure_bids_format.png)

Additionally, you will need to have two json files in the `input` folder:
- `meld_bids_config.json` containing the key words for session, datatype and modality suffix \
    Example: 
    ```json
    {"T1": {"session": null, 
           "datatype": "anat",
           "suffix": "T1w"},
    "FLAIR": {"session": null, 
              "datatype": "anat",
              "suffix": "FLAIR"}}
    ```

- `dataset_description.json` containing a description of the dataset \
    Example:
    ```json
    {"Name": "Example dataset", 
    "BIDSVersion": "1.0.2"}
    ```

## Prepare the demographic information (required only to compute the harmonisation parameters)

To compute the harmonisation parameters, you will need to provide a couple of information about the subjects into a csv file. 

You can copy the *demographics_file.csv* that you can find in your <meld_data_folder> and create a new version of it with the harmonisation code *demographics_file_<harmo_code>.csv*. (e.g *demographics_file_H1.csv*)

![example](https://raw.githubusercontent.com//MELDProject/meld_graph/dev_docker/docs/images/example_demographic_csv.png)

- ID : subject ID  (this should be the same ID than the one used to create the MRI folder)
- Harmo code: the harmonisation code associated with this subject scan (need to be the same for all the subjects used for the harmonisation) 
- Group: 'patient' if the subject is a patient or 'control' if the subject is a control 
- Age at preoperative: The age of the subject at the time of the preoperative T1 scan (in years)
- Sex: 1 if male, 0 if female
- Scanner: the scanner strenght associated with the MRI data ('3T' for 3 Tesla or '15T' for 1.5 Tesla)

Note: please ensure to not change the name of the columns and to have completed the files with the appropriate values, otherwise the pipeline will fail. 


