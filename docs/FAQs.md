# FAQs

## Issues with native installation
TBC

## Issues with Docker installation
TBC

## Issues with Singularity/Apptainer installation

---
### Issue of space when with creating the SIF
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