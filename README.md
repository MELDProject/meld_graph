<img src="https://raw.githubusercontent.com//MELDProject/meld_graph/main/docs/images/MELD_logo.png" alt="MELD logo" width="100" align="left"/> 

# MELD Graph 

**Full documentation: [here](https://meld-graph.readthedocs.io/en/latest/index.html)**

**Intro to MELD Graph and installation videos: [here](https://youtu.be/PIM1gwYNLns)**

Graph based FCD lesion segmentation for the [MELD project](https://meldproject.github.io/).

This package is a pipeline to segment FCD-lesions from MRI scans. 

**<span style="color: red;">SIGN UP TO THE MELD GRAPH MAILING LIST</span>**:
We request that all MELD Graph users sign up to the mailing list. If you are using MELD Graph, please send an email to `meld.study@gmail.com` with the subject 'Request to be added to the MELD Graph mailing list' and provide use with your name and institute. This will ensure that we can update you about bug fixs and new releases. 

**<span style="color: red;">EXISTING USERS: PLEASE UPDATE TO VERSION V2.2.2</span>**:
We have released MELD Graph V2.2.2 which fixes a couple of issues found by users. For more information about the release please see [MELD Graph V2.2.2](https://github.com/MELDProject/meld_graph/releases/tag/v2.2.2). To update your code please follow the guidelines [Updating MELD Graph to V2.2.2](https://meld-graph.readthedocs.io/en/latest/FAQs.html#Updating-MELD-Graph-to-V2.2.2) from our FAQ.


![overview](https://raw.githubusercontent.com//MELDProject/meld_graph/main/docs/images/Fig1_pipeline.jpg)

*Code Authors : Mathilde Ripart, Hannah Spitzer, Sophie Adler, Konrad Wagstyl*

## Notes

This package is intended to be used as a research tool to segment FCD lesions in patients with focal epilepsy where a FCD is suspected. It can be run on 1.5T or 3T MRI data. A 3D T1 is required and it is optional but advised to include the 3D FLAIR. 

It is not appropriate to use this algorithm on patients with:
- suspected hippocampal sclerosis
- hypothalamic hamartoma
- periventricular nodular heterotopia
- other focal epilepsy pathologies
- previous resection cavities

**Harmonisation** - MRI data from different MRI scanners looks subtly different. This means that feature measurements, e.g. cortical thickness measurements, differ depending on which MRI scanner a patient was scanned on. We harmonise features (using NeuroCombat) to adust for site based differences. We advise new users to harmonise data from their MRI scanner to the MELD graph dataset. Please follow the guidelines to harmonise the data from your site. Note: the model will still produce predictions on new, unharmonised subjects but the number of false positive predictions is higher if the data is not harmonised.

This package also contains code for training and evaluating graph-based U-net lesion segmentation models operating on icosphere meshes. \
In addition to lesion segmentation, the model also contain auxiliary distance regression and hemisphere classification losses.

For more information on how the algorithm was developed and expected performance - check our papers: 
- [Ripart et al.,2025 JAMA Neurology -  Detection of epileptogenic focal cortical dysplasia using graph neural networks: a MELD study](https://jamanetwork.com/journals/jamaneurology/fullarticle/2830410)
- [Spitzer, Ripart et al., 2022 Brain - the original MELD FCD pipeline and dataset](https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752)
- [Spitzer et al., 2023 MICCAI - the updated graph-based model architecture](https://arxiv.org/abs/2306.01375)


## Disclaimer

The MELD surface-based graph FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Installation & Use of the MELD FCD prediction pipeline

### Installations available 
You can install and use the MELD FCD prediction pipeline with :
- [**docker container**](https://meld-graph.readthedocs.io/en/latest/install_docker.html) recommended for easy installation of the pipeline as all the prerequisite packages are already embedded into the container. Note: Dockers are not working on High Performance Computing (HCP) systems
- [**native installation**](https://meld-graph.readthedocs.io/en/latest/install_native.html) recommended for Mac and users that want to modify the code and/or use the code to train/test their own classifier. 
- [**singularity container**](https://meld-graph.readthedocs.io/en/latest/install_singularity.html) enables to run a container on High Performance Computing (HCP) systems.

**IMPORTANT NOTE**: The installations listed above are not supported on Virtual Machines. Please install MELD Graph on full Linux, Windows or MAC computers

**YouTube tutorials available for the [docker installation](https://youtu.be/oduOe6NDXLA) and [native installation](https://youtu.be/jUCahJ-AebM)**


### Running the pipeline 
Once installed you will be able to use the MELD FCD prediction pipeline on your data following the steps:
1. Prepare your data : [guidelines](https://meld-graph.readthedocs.io/en/latest/prepare_data.html)
2. Compute the harmonisation parameters : [guidelines](https://meld-graph.readthedocs.io/en/latest/harmonisation.html) (OPTIONAL but highly recommended)
3. Run the prediction pipeline: [guidelines](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html)
4. Interpret the results: [guidelines](https://meld-graph.readthedocs.io/en/latest/interpret_results.html)

**YouTube tutorials available to run the [harmonisation step](https://youtu.be/te_TR6sA5sQ), to run the [prediction pipeline](https://youtu.be/OZg1HSzqKyc) and to [interpret the pipeline results](https://youtu.be/dSyd1zOn4F8)**

**FAQs** 
If you have a question or if you are running into issues at any stage (installation/use/interpretation), have a look at our [FAQs](https://meld-graph.readthedocs.io/en/latest/FAQs.html) page as we may have already have a solution. 

**What is the harmonisation process ?**

Scanners can induce a bias in the MRI data. The MELD pipeline recommends adjusting for these scanners differences by running a preliminary harmonisation step to compute the harmonisation parameters for that specific scanner. Note: this step needs to be run only once, and requires data from at least 20 subjects acquired on the same scanner and demographic information (e.g age and sex). See [harmonisation instructions](https://meld-graph.readthedocs.io/en/latest/harmonisation.html) for more details. 

Note: The MELD pipeline can also be run without harmonisation, with a small drop in performance.

## Additional information
With the native installation of the MELD classifier you can reproduce the figures from our paper and train/evaluate your own models.
For more details, check out the guides linked below:
- [Notebooks to reproduce figures](https://meld-graph.readthedocs.io/en/latest/figure_notebooks.html)
- [Train and evaluate models](https://meld-graph.readthedocs.io/en/latest/train_evaluate.html)

## Contribute
If you'd like to contribute to this code base, have a look at our [contribution guide](https://meld-graph.readthedocs.io/en/latest/contributing.html)


## Acknowledgments

We would like to thank 
- the [MELD consortium](https://meldproject.github.io//docs/collaborator_list.pdf) for providing the data to train this classifier and their expertise to build this pipeline.\
- [Lennart Walger](https://github.com/1-w) and [Andrew Chen](https://github.com/andy1764), for their help testing and improving the MELD pipeline to v1.1.0. \
- [Ulysses Popple](https://github.com/ulyssesdotcodes) for his help building the docs and dockers.
- [Cornelius Kronlage](https://github.com/ckronlage) highlighting issues in v2.2.1 and suggesting solutions in v2.2.2

## Contacts

Contact the MELD team at `meld.study@gmail.com`

*Please note that we are a small team and only have one day a week dedicated to the support of the MELD tools ([MELD Graph](https://github.com/MELDProject/meld_graph) and [AID-HS](https://github.com/MELDProject/AID-HS)). We will answer your emails as soon as we can!*
