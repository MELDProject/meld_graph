# Prepare data to run MELD pipeline

The MELD pipeline relies on the MRI data to be organised in the MELD or BIDS format. 

If you are preparing the data for the harmonisation step, you will also need to prepare the demographic information. 

## Prepare the MRI data (Mandatory)

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient with the id of the subject. 

### MELD format

In each subject folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder.

Place the FLAIR nifti file into the FLAIR folder.

![example](images/example_folder_structure.png)

### BIDS format

**TO UPDATE** The MELD pipeline now accept BIDS format as input data. For the BIDS format the data needs to be organised following ...

## Prepare the demographic information (required only to compute the harmonisation parameters)

To compute the harmonisation parameters, you will need to provide a couple of information about the subjects into a csv file. 

You can copy the *demographics_file.csv* that you can find in your <meld_data_folder> and create a new version of it with the harmonisation code *demographics_file_<harmo_code>.csv*. (e.g *demographics_file_H1.csv*)

![example](images/example_demographic_csv.PNG)

- ID : subject ID  (this should be the same ID than the one used to create the MRI folder)
- Harmo code: the harmonisation code associated with this subject scan (need to be the same for all the subjects used for the harmonisation) 
- Group: 'patient' if the subject is a patient or 'control' if the subject is a control 
- Age at preoperative: The age of the subject at the time of the preoperative T1 scan (in years)
- Sex: 1 if male, 0 if female
- Scanner: the scanner strenght associated with the MRI data ('3T' for 3 Tesla or '15T' for 1.5 Tesla)

Note: please ensure to not change the name of the columns and to have completed the files with the appropriate values, otherwise the pipeline will fail. 


