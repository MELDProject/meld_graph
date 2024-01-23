# MELD graph classifier
Graph based lesion segmentation for the MELD project.

The manuscript describing the original classifier and dataset can be found here https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752 and the updated graph-based classifier can be found here https://arxiv.org/abs/2306.01375.

*Code Authors : Hannah Spitzer, Mathilde Ripart, Sophie Adler, Konrad Wagstyl*

![overview](images/overview.png)


## Notes

This package comes with two pretrained models that can be used to predict new subjects (with or without harmonising the data beforehand). It also contains code for training and evaluating graph-based U-net lesion segmentation models operating on icosphere meshes. In addition to lesion segmentation, the models also contain auxiliary distance regression and hemisphere classification losses.

**What is the harmonisation process ?**

MRI data acquired on different scanner / site will induce a slight bias in the data compare to the data that have been used to train the MELD classifier. 

The MELD pipeline offers the possibility to run a preliminary harmonisation step to compute the harmonisation parameters for a specific scanner. Note: this step requires data from >20 subjects acquired on this scanner and demographic information (e.g age and sex). This step needs to be done only once, see [harmonisation instructions](/documentation/Harmonisation.md) for more details. 

The MELD pipeline can also be run without harmonisation. This means that the slight bias in the MRI data won't be adjusted, but the new MELD graph classifier has been optimised to remove the need for harmonisation. 

For more details on the harmonisation process and the classifier performances with and without harmonisation, please read our [manuscript](). 

We also have a ["Guide to using the MELD surface-based FCD detection algorithm on a new patient"](https://docs.google.com/document/d/1vF5U1i-B45OkE_8wdde8yHHypp6W9xNN_1DBoEGmn0E/edit?usp=sharing). This explains how to harmonise your data and how to run the classifier in much more detail as well as how to interpret the results.

## Disclaimer

The MELD surface-based graph FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Installation & Use of the MELD prediction pipeline

You can install and use the MELD prediction pipeline with :
- [**docker container (recommended).**](/documentation/Run_with_docker.md) This method is recommended for easy installation and use of the pipeline as all the prerequisite packages are already embeded into the container. Note: Dockers are not working on High Performance Computing (HCP) system.
- [**native installation.**](/documentation/Run_with_native.md) This method is recommended for more advance users that would like to be able to modify the codes and/or use the code to train/test their own classifier. 
- **singularity container (COMING SOON)**: This method is not yet available but will enable to run container on High Performance Computing (HCP) system. 


## Other guidelines
With the [native installation of the MELD classifier](/documentation/Run_with_docker.md) you can reproduce the figures from our paper and train/evaluate your own models.
For more details, check out the guides linked below:
- [Notebooks to reproduce figures](/documentation/figure_notebooks.md)
- [Train and evaluate models](/documentation/Training_and_evaluating_models.md)

## Contribute
If you'd like to contribute to this code base, have a look at our [contribution guide](/documentation/CONTRIBUTING.md)

## Manuscript
Please check out our [original manuscript](https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752) and [graph updated manuscript](https://arxiv.org/abs/2306.01375) to learn more.

An overview of the notebooks that we used to create the figures can be found [here](figure_notebooks.md).


## Acknowledgments

We would like to thank the [MELD consortium](https://meldproject.github.io//docs/collaborator_list.pdf) for providing the data to train this classifier and their expertise to build this pipeline.\
We would like to thank [Lennart Walger](https://github.com/1-w) and [Andrew Chen](https://github.com/andy1764), for their help testing and improving the MELD pipeline to v1.1.0
