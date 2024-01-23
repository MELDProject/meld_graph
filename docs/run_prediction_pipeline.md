# Predict lesion on a new patient 

With the MELD classifier pipeline 

<ins>Existing site</ins>: if you are from an epilepsy centre who's data was used to train the classifier, predicting lesion locations on new patients is easy. In the following, we describe the steps needed to use the trained model to predict on a new patient. 

Note: No demographic information are required for this process.

<ins>New site</ins>: If you would like to predict lesions on patients from **new epilepsy centres** or **new MRI scanners or updated T1 / FLAIR sequences** that were not used to train the classifier, you will need to first compute the harmonisation parameters for your site following the [Harmonisation_new_site.md](Harmonisation_new_site.md). This step needs to be done only once, then you can follow the same guidelines as an existing site. 

Note: Demographic information (e.g age and sex) will be required for this process.

We also have a ["Guide to using the MELD surface-based FCD detection algorithm on a new patient"](https://docs.google.com/document/d/1vF5U1i-B45OkE_8wdde8yHHypp6W9xNN_1DBoEGmn0E/edit?usp=sharing). This explains how to harmonise your data and how to run the classifier in much more detail as well as how to interpret the results.

## Disclaimer

The MELD surface-based FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Information about the pipeline
Before running the below pipeline, ensure that you have [installed MELD classifier](README.md#installation) and activate the meld_classifier environment : 
```bash
conda activate meld_classifier
```
Also you need to make sure that Freesurfer is activated in your terminal (you should have some printed FREESURFER paths when opening the terminal). Otherwise you will need to manually activate Freesurfer on each new terminal by running : 
```bash
export FREESURFER_HOME=<freesurfer_installation_directory>/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```
with `<freesurfer_installation_directory>` being the path to where your Freesurfer has been installed.

NOTES: MELD pipeline has only been tested and validated on Freesurfer up to V7.2. Please do not use higher version than V7.2.0 \


### First step - Organising your data!

You need to organise the MRI data for the patients you want to run the classifier on.

In the 'input' folder where your meld data has / is going to be stored, create a folder for each patient. 

The IDs should follow the same naming structure as before. i.e. MELD\_<site\_code>\_<scanner\_field>\_FCD\_000X

e.g.MELD\_H1\_3T\_FCD\_0001 

In each patient folder, create a T1 and FLAIR folder.

Place the T1 nifti file into the T1 folder. Please ensure 'T1' is in the file name.

Place the FLAIR nifti file into the FLAIR folder. Please ensure 'FLAIR' is in the file name.

![example](images/example_folder_structure.png)

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
python scripts/new_patient_pipeline/new_pt_pipeline.py -site <site_code> -id <subject_id> 
```

You can tune this command using additional variables and flags as detailed bellow:

| **Mandatory variables**         |  Comment | 
|-------|---|
| ```-site <site_code>```  |  the site code should start with H, e.g. H1. If you cannot remember your site code - contact the MELD team. | 
|either ```-id <subject_id>```  |  if you want to run the pipeline on 1 single subject. Needs to be in MELD format MELD\_<site\_code>\_<scanner\_field>\_FCD\_000X |  
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
python scripts/new_patient_pipeline/new_pt_pipeline.py -site H4 -id MELD_H4_3T_FCD_0001 --fastsurfer
```

To run the whole prediction pipeline on multiples subjects with parallelisation:
```bash
python scripts/new_patient_pipeline/new_pt_pipeline.py -site H4 -ids list_subjects.txt --parallelise
```

## Additional information about the 3 different scripts / steps

### Script 1 - FreeSurfer reconstruction and smoothing

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

### Script 2 - Feature Preprocessing

This script : 
  1. Combat harmonise features and write into hdf5
  2. Normalise the smoothed features (intra-subject & inter-subject (by controls)) and write in hdf5
  3. Normalise the raw combat features (intra-subject, asymmetry and then inter-subject (by controls)) and write in hdf5

  Notes: 
  - Features need to have been extracted and smoothed using script 1. 
  - (optional): this script can also be called to harmonise your data for new site but will need to pass a file containing demographics information.

To know more about the script and how to use it on its own:
```bash
python scripts/new_patient_pipeline/run_script_preprocessing.py -h
```

### Script 3 - Lesions prediction & MELD reports

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

The precalculated .png images of predicted lesions and their associated saliencies can be used to look at the predicted clusters and why they were detected by the classifier. The MELD pdf report provides a summary of all the prediction for a subject.

After viewing these images, we recommend then viewing the predictions superimposed on the T1 volume. This will enable:
- Re-review of the T1 /FLAIR at the predicted cluster locations to see if an FCD can now be seen
- Performing quality control
- Viewing the .png images of predicted lesions

### Main outputs

The predictions are saved as NIFTI files in the folder: 
/output/predictions_reports/<subject_id>/predictions
- prediction.nii corresponds to the prediction mask for the whole brain
- lh.prediction.nii and rh.prediction.nii correspond to the predictions masks for left and right hemispheres

***NEW***: You can merge the MELD predictions onto the T1 nifti file using the command below. Note that you will need to have [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) installed on your machine. 
```bash
python scripts/new_patient_pipeline/merge_predictions_t1.py -id <subject_id> -t1 <path_to_t1_nifti> -pred <path_to_meld_prediction_nifti> -output_dir <where_to_save_output>
```
The command will create the file predictions_merged_t1.nii.gz which corresponds to the predictions masks merged with T1 in RGB format. It can be viewed on RGB viewer or used to transfert on PACS system.



### Viewing the predicted clusters
The MELD pdf report and .png images of the predicted lesions are saved in the folder:
 /output/predictions_reports/<subject_id>/reports
 

The first image is called inflatbrain_<subject_id>.png

![inflated](images/inflatbrain_sub_id.png)

This image tells you the number of predicted clusters and shows on the inflated brain where the clusters are located.

The next images are mri_<subject_id>_<hemi>_c*.png

E.g. 

![mri](images/mri_sub_id_lh_c1.png)

These images show the cluster on the volumetric T1 image. Each cluster has its own image e.g.  mri_<subject_id>_<hemi>_c1.png for cluster 1 and  mri_<subject_id>_<hemi>_c2.png for cluster 2.

  
### Saliency
  
The next images are called saliency_<subject_id>_<hemi>_c*.png. Each cluster has a saliency image associated with it. E.g.
  
![saliency](images/saliency_sub_id_lh_c1.png)
  
These detail:
* The hemisphere the cluster is on
* The surface area of the cluster (across the cortical surface)
* The location of the cluster
* The z-scores of the patient’s cortical features averaged within the cluster. In this example, the most abnormal features are the intrinsic curvature (folding measure) and the sulcal depth.
* The saliency of each feature to the network - if a feature is brighter pink, that feature was more important to the network. In this example, the intrinsic curvature is most important to the network’s prediction

The features that are included in the saliency image are:
* **Grey-white contrast**: indicative of blurring at the grey-white matter boundary, lower z-scores indicate more blurring
* **Cortical thickness**: higher z-scores indicate thicker cortex, lower z-scores indicate thinner cortex
* **Sulcal depth**: higher z-scores indicate deeper average sulcal depth within the cluster
* **Intrinsic curvature**: a measure of cortical deformation that captures folding abnormalities in FCD. Lesions are usually characterised by high z-scores
* **WM FLAIR**: FLAIR intensity sampled at 1mm below the grey-white matter boundary. Higher z-scores indicate relative FLAIR hyperintensity, lower z-scores indicate relative FLAIR hypointensity
* **GM FLAIR**: FLAIR intensity sampled at 50% of the cortical thickness. Higher z-scores indicate relative FLAIR hyperintensity, lower z-scores indicate relative FLAIR hypointensity
* **Mean curvature**: Similar to sulcal depth, this indicates whether a vertex is sulcal or gyral. Its utility is mainly in informing the classifier whether a training vertex is gyral or sulcal. Within FCD lesions, it is usually not characterised by high z-scores or high saliency.

If you only provide a T1 image, the FLAIR features will not be included in the saliency plot.

## Viewing the predictions on the T1 and quality control

It is important to check that the clusters detected are not due to obvious FreeSurfer reconstruction errors, scan artifacts etc.

To do this run: 
```bash
cd <path_to_meld_classifier>
conda activate meld_classifier
python scripts/new_patient_pipeline/new_pt_qc_script.py -id <subject_id>
```
![qc_surface](images/qc_surface.png)

This will open FreeView and load the T1 and FLAIR (where available) volumes as well as the classifier predictions on the left and right hemispheres. It will also load the FreeSurfer pial and white surfaces. It will look like this:

You can scroll through and find the predicted clusters.
![qc_surface](images/qc_cluster.png)

Example of a predicted cluster (orange) on the right hemisphere. It is overlaid on a T1 image, with the right hemisphere pial and white surfaces visualised. Red arrows point to the cluster. 

**Things to check for each predicted cluster:**

1. Are there any artifacts in the T1 or FLAIR data that could have caused the classifier to predict that area?

2. Check the .pial and .white surfaces at the locations of any predicted clusters. 
Are they following the grey-white matter boundary and pial surface? If not, you need to try and establish if this is just a reconstruction error or if the error is due to the presence of an FCD. If it is just an error or due to an artifact, exclude this prediction. If it is due to an FCD, be aware that the centroid  / extent of the lesion may have been missed due to the reconstruction error and that some of the lesion may be adjacent to the predicted cluster. 

Note: the classifier is only able to predict areas within the pial and white surfaces.

## Limitations 

**Limitations to be aware of:**

* If there is a reconstruction error due to an FCD, the classifier will only be able to detect areas within the pial and white surfaces and may miss areas of the lesion that are not correctly segmented by FreeSurfer
* There will be false positive clusters. You will need to look at the predicted clusters with an experienced radiologist to identify the significance of detected areas
* The classifier has only been trained on FCD lesions and we do not have data on its ability to detect other pathologies e.g. DNET / ganglioglioma / polymicrogyria. As such, the research tool should only be applied to patients with FCD / suspected FCD
* Performance of the classifier varies according to MRI field strength, data available (e.g. T1 or T1 and FLAIR) and histopathological subtype. For more details of how the classifier performs in different cohorts, see [our paper](https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752).

## How to cite the classifier
  
Spitzer, H., Ripart, M, Whitaker, K., Napolitano, A., De Palma, L., De Benedictis, A., et al. 2022. “Interpretable Surface-Based Detection of Focal Cortical Dysplasias: A Multi-Centre Epilepsy Lesion Detection Study.” Brain: A Journal of Neurology, August. https://doi.org/10.1093/brain/awac224.
