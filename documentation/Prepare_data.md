# Prepare data to run MELD pipeline

The MELD pipeline relies on the MRI data to be organised in the MELD or BIDS format and to provide few information about the subject.

## Prepare the MRI data

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient with the id of the subject. 

### MELD format

In each subject folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder.

Place the FLAIR nifti file into the FLAIR folder.

![example](images/example_folder_structure.png)

### BIDS format

**TO UPDATE** The MELD pipeline now accept BIDS format as input data. For the BIDS format the data needs to be organised following ...

## Demographic information

Before to run the pipeline you will need to provide a couple of information about the subject into the *demographics_file.csv* that you can find in your <meld_data_folder>. 

If you need to run the harmonisation, you will need to provide the information below:

![example](images/example_demographic_csv.PNG)
- ID : subject ID  (this should be the same ID than the one used to create the MRI folder)
- Harmo code: the harmonisation code associated with this subject scans 
- Group: 'patient' if the subject is a patient or 'control' if the subject is a control 
- Age at preoperative: The age of the subject at the time of the preoperative T1 scan (in years)
- Sex: 1 if male, 0 if female
- Scanner: the scanner strenght associated with the MRI data ('3T' for 3 Tesla or '15T' for 1.5 Tesla)


