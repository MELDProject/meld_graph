# MELD Graph 
Graph based FCD lesion segmentation for the MELD project.

This package offers a friendly user pipeline to segment FCD-lesions from MRI scans. 

![overview](https://raw.githubusercontent.com//MELDProject/meld_graph/dev_docker/docs/images/Fig1_pipeline.jpg)

*Code Authors : Mathilde Ripart, Hannah Spitzer, Sophie Adler, Konrad Wagstyl*

## Notes

This package comes with a pretrained graph based model, trained on harmonised features. New users are advised to follow the guidelines to harmonise the data from their site, but the model will still produce predictions on new, unharmonised subjects.

It also contains code for training and evaluating graph-based U-net lesion segmentation models operating on icosphere meshes. \
In addition to lesion segmentation, the models also contain auxiliary distance regression and hemisphere classification losses.

For more information check our manuscripts: 
- [Spitzer, Ripart et al., 2022 Brain paper describing the original MELD FCD pipeline and dataset](https://academic.oup.com/brain/advance-article/doi/10.1093/brain/awac224/6659752)
- [Spitzer et al., 2023 MICCAI paper describing the updated graph-based model](https://arxiv.org/abs/2306.01375)
- [Ripart et al., 2024 preprint describing MELD Graph performance and interpretable reports](todo)

## Disclaimer

The MELD surface-based graph FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA), European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. There is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient's own risk.

## Installation & Use of the MELD FCD prediction pipeline

### Installations available 
You can install and use the MELD FCD prediction pipeline with :
- [**docker container**](https://meld-graph.readthedocs.io/en/latest/install_docker.html) recommended for easy installation of the pipeline as all the prerequisite packages are already embeded into the container. Note: Dockers are not working on High Performance Computing (HCP) systems.
- [**native installation**](https://meld-graph.readthedocs.io/en/latest/install_native.html) recommend for more advance users that would like to be able to modify the code and/or use the code to train/test their own classifier. 
- **singularity container (COMING SOON)** enables to run a container on High Performance Computing (HCP) systems. 

### Running the pipeline 
Once installed you will be able to use the MELD FCD prediction pipeline on your data following the steps:
1. Prepare your data : [guidelines](https://meld-graph.readthedocs.io/en/latest/prepare_data.html)
2. (OPTIONAL) Compute the harmonisation parameters : [guidelines](https://meld-graph.readthedocs.io/en/latest/harmonisation.html)
3. Run the prediction pipeline: [guidelines](https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html)
4. Interpret the results: [guidelines](https://meld-graph.readthedocs.io/en/latest/interpret_results.html)


**What is the harmonisation process ?**

Scanners can induce a bias in the MRI data. The MELD pipeline offers the possibility to adjust for these scanners differences by running a preliminary harmonisation step to compute the harmonisation parameters for that specific scanner. Note: this step needs to be run only once, and requires data from at least 20 subjects acquired on the same scanner and demographic information (e.g age and sex). See [harmonisation instructions](https://meld-graph.readthedocs.io/en/latest/harmonisation.html) for more details. 

Note: The MELD pipeline can also be run without harmonisation. 

## Additional information
With the native installation of the MELD classifier you can reproduce the figures from our paper and train/evaluate your own models.
For more details, check out the guides linked below:
- [Notebooks to reproduce figures](https://meld-graph.readthedocs.io/en/latest/figure_notebooks.html)
- [Train and evaluate models](https://meld-graph.readthedocs.io/en/latest/train_evaluate.html)

## Contribute
If you'd like to contribute to this code base, have a look at our [contribution guide](https://meld-graph.readthedocs.io/en/latest/contributing.html)


## Acknowledgments

We would like to thank the [MELD consortium](https://meldproject.github.io//docs/collaborator_list.pdf) for providing the data to train this classifier and their expertise to build this pipeline.\
We would like to thank [Lennart Walger](https://github.com/1-w) and [Andrew Chen](https://github.com/andy1764), for their help testing and improving the MELD pipeline to v1.1.0
We would like to thank [Ulysses Popple](https://github.com/ulyssesdotcodes) for his help building the docs and dockers.

## Contacts

Contact the MELD team at `meld.study@gmail.com`
