# FAQs

## **Issues & questions with installation**

### **Issue with Native installation MAC Intel user - Issue when running meldsetup.sh**

If you are a MAC user with an intel processor you will run into the issue below when running the command ```./meldsetup.sh```:
```bash
[...]
Installing pip dependencies: / Ran pip subprocess with arguments:
Pip subprocess output:
ERROR: CONDA_BUILD_SYSROOT or SDKROOT has to be set for cross-compiling
[...]
ModuleNotFoundError: No module named '_sysconfigdata_arm64_apple_darwin20_0_0'
```

The issue happens because the code is trying to force-installing ARM64 specific packages on an Intel processor. 
An alternative solution is to follow the steps below: 
1. Remove the meld_graph environment that failed 
`conda remove -n meld_graph --all`
2. Open the `meldsetup.sh` file and replace the line `conda env create -f environment-mac.yml` with `conda env create -f environment.yml`
3. Save the file and rebuild the environment by running:
```bash
./meldsetup.sh
```

If this raises new issues about packages that could not be found or installed, please contact meld.study@gmail.com with information about the issue and the packages missing. 

### **Issue with Singularity - Not enough space when with creating the SIF**
```bash
INFO:    Creating SIF file... 
FATAL:   While performing build: while creating squashfs: create command failed: exit status 1:  
Write failed because No space left on device 
```
It means there is not enough space where the default singularity/apptainer cache and temporary directories are located. Usually the default cache is located in `$HOME/.singularity/cache` and the default temporary directory `$HOME/.tmp`\
Solution:
- You can make space in the default `$HOME` directory
- You can change the singularity/apptainer cache and temporary directories for a folder where there is space:
    ```bash
    export SINGULARITY_CACHEDIR=<path_folder_with_space> 
    export SINGULARITY_TMPDIR=<path_folder_with_space>
    ```
    Or with apptainer
    ```bash
    export APPTAINER_CACHEDIR=<path_folder_with_space> 
    export APPTAINER_TMPDIR=<path_folder_with_space>
    ```
---

## **Issues & questions with pipeline use**

### **I have an issue with FLAIR feature that does not exist**

If you are running a subject with only a T1 scan and no FLAIR scan but you receive an issue like :
```bash
KeyError: "Unable to open object (object '.on_lh.gm_FLAIR_0.25.sm3.mgh' doesn't exist)"
exit status 1
```
You are likely having this issue because you might have previously ran this same subject ID with a FLAIR scan and the FreeSurfer segmentation has been done using the FLAIR scan. Therefore, even if you remove the FLAIR scan from the input data and run again the command, the intermediate FreeSurfer outputs for that subject still contain FLAIR information, which will make the pipeline looks for for FLAIR features but fail to find them.

To avoid this in the future, if you want to run a same subject with and without FLAIR, you should create two separate input folders with two different subject's ID such as `sub-0001noflair` and `sub-0001flair`.

### **I have an issue during the harmonisation**

If your issue looks like :
```bash
INFO - subject id: Creating final training data matrix
[...]
site_code, c_p, scanner = get_group_site(fs_id, demographic_file)
TypeError: cannot unpack non-iterable NoneType object
```
You are likely having an issue with the `demographics_file.csv` or the `list_subjects.txt` you provided. 
- In the `demographics_file.csv`, check that there are no extra columns and that the column names match what was provided as an [example](https://meld-graph.readthedocs.io/en/latest/prepare_data.html). Also, ensure that the file is saved with comma separators (",") and not semicolon (";") which will prevent the code from properly reading the file. 
- In the `list_subjects.txt`, ensure that there is no extra empty line at the end of the file.

### **Issue during prediction - The pipeline works and then stop when running the predictions and saliencies**

The error is likely due to a memory issue when the machine-learning model is called to predict.\
If you are using Docker Desktop, it could be because the memory limit is set very low by default. 
To fix this, you will need to:
1) Increase the memory in the Docker Desktop settings (more help in this [post](https://stackoverflow.com/questions/43460770/docker-windows-container-memory-limit)
2) Run the MELD Graph command again. 


### **Can I use precomputed FreeSurfer outputs in the pipeline ?**

If prior to using this pipeline you already have processed a T1w scan (or T1w and FLAIR scans) with the `recon-all` pipeline from FreeSurfer **V6.0** or **V7.2**, you can use the output FreeSurfer folder for this patient in the pipeline. The pipeline will use those outputs and skip the FreeSurfer segmentation.  

To do this, place the patient's FreeSurfer folder into the meld_data folder in the `output/fs_outputs` folder. You will need to ensure that the freesurfer subject's folder name matchs the subject ID used in the input data. 

For example, if you have a patient 'sub-002` and you already have FreeSurfer output folder for that patient, you can rename the folder by the subject id and place it as follow:
```
output
└── fs_outputs
    └── sub-0002
```
Typical outputs folder from the `recon-all` command would look like this:
```
sub-0002
    ├── label
    ├── mri
    ├── scripts
    ├── stats
    ├── surf
    ├── tmp
    ├── touch
    ├── trash
```

### **Can I use the MELD Graph pipeline on scans that contain previous resection cavities?**

The short answer is no. 

The MELD Graph pipeline has not been trained on scans that contains resection cavities. Such scans will likely induce errors in the brain segmentation which will bias the prediction. 

If the patient already had surgery, we recommand to use the scans that were acquired prior to the surgery and use those to run in the pipeline

###  **Can I use the MELD Graph pipeline on scans with other pathologies as well as FCD e.g. tumours?**
If the other pathology is hippocampal sclerosis (E.g. this is a FCD IIIA), yes you can use MELD Graph. However, please note it will not detect the HS as it only analyses the cortex.

If it is another pathology e.g. a tumour, the pipeline has not been developed / trained on other pathologies and may not detect them. Also, the other pathology may introduce large reconstruction errors in the FreeSurfer pipeline - causing errors. We therefore do not recommend using MELD Graph on patients with other cortical pathologies. 
