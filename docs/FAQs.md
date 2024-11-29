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
conda remove -n meld_graph --all
2. Open the `meldsetup.sh` file and replace the line `conda env create -f environment-mac.yml` by `conda env create -f environment.yml`
3. Save the file and rebuilt the environment by running:
```bash
./meldsetup.sh
```

This might raises new issues about packages that could not be found or installed. Please contact the meld.study@gmail.com with information about the issue and the packages missing. 


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

### **Can I use the MELD Graph pipeline on scans that contains previous resection cavities?**

The short answer is no. 

The MELD Graph pipeline has not been trained on scans that contains resection cavities. Such scans will likely induce errors in the brain segmentation which will bias the prediction. 

If the patient already had surgery, we recommand to use the scans that were acquired prior this surgery and use those to run in the pipeline

###  **Can I use the MELD Graph pipeline on scans with other pathologies as well as FCD e.g. tumours?**
If the other pathology is hippocampal sclerosis (E.g. this is a FCD IIIA), yes you can use MELD Graph. However, please note it will not detect the HS as it only analyses the cortex.

If it is another pathology e.g. a tumour, the pipeline has not been developed / trained on other pathologies and may not detect them. Also, the other pathology may introduce large reconstruction errors in the FreeSurfer pipeline - causing errors. We therefore do not recommend using MELD Graph on patients with other cortical pathologies. 
