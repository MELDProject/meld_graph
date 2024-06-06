## Code Structure


### Config files 
Example config files can be found in [`scripts/config_files/example_experiment_config.py`](scripts/config_files/example_experiment_config.py). They define the data that the model is trained on, the model architecture, and training parameters.

### Experiment class
Main class is the Experiment (`meld_graph/experiment.py`). It is initialised with data_parameters and network_parameters. Experiments are saved in folders in EXPERIMENT_PATH (defined in `meld_graph/paths.py`). Experiments are only saved when `network_parameters['name']` is not None.

### Data loader
Data is loaded by GraphDataset in `meld_graph/dataset.py` using Preprocess in `meld_graph/data_preprocessing.py`. Preprocess returns per subject vertices and labels. In GraphDataset, each hemisphere is treated as a single datapoint, with optional stacking of hemisphere features (to eg allow the model to calculate assymetry) -- use `data_parameters['combine_hemis']` to modify this behaviour. 

If desired, train samples are augmented using `Augmentation` in `meld_graph/augmentation.py`, using `data_parameters['augment_data']`. Intensity, mesh, and lesion augmentation are implemented.

### Models
Models are defined in `meld_graph/models.py`. There is an implementation for a simple MoNet (just convolutions) and a MoNetUnet (convolutions and hex pooling). Model architecture can be modified using `network_parameters['model_parameters']`. As convolutions GMMConv and SpiralConv can be used (`conv_type` parameter).

Models instanciate the `IcoSpheres` class. This computes and returns icospheres (edge matrices, neighbours, edge attributes) at different levels. Level 7 is the highest resolution. GMMConv, SpiralConv, HexPool, and HexUnpool all use the edges/neighbours at different levels. IcoSpheres needs a few parameters for initalisation (`data_parameters['icosphere_parameters']`) - some parameters (`conv_type`) are automatically added to this dict in `Experiment.load_model()`. 

### Training
Training is done by Trainer in `meld_graph/training.py`. Relevant params are in `network_parameters['training_parameters']`. Several metrics can be tracked during training (dice_lesion, dice_nonlesion, etc). Training is possible with multiple hemispheres (data points) at once (by specifying a batch_size larger than 1). Internally this is achieved by looping over all elements in this batch and stacking them afterwards (the GMMConv and SpiralConv expect the batch dimension to be the number of vertices in the graph). Deep supervision can be achieved by adding a "deep_supervision" dict to `training_parameters`, containing "levels" (the isosphere levels at which to add supervision) and "weight" (the weight of the loss for the auxiliary loss).

Training can be started from pretrained models. These models need to have an *identical* configuration to the current model. Point `network_parameters['training_parameters']['init_weights']` to a saved `model.pt` file to initialise this model with the saved weights.

Patience is implemented, with the best model being saved in the experiment directory. Choose the stopping metric with `network_parameters['training_parameters']['stopping_metric']`.
Training logs and train/val scores are also saved in the experiment directory.

To train on an easier task, use the frontal lobe parcellation task. This is another binary classification task,
that can be used as a drop-in task for the harder lesion segmentation.
In `data_parameters`, set `lobe = True`, to train on this task.

### Evaluation
Evaluation functions are found in `meld_graph/evaluation.py`. We have different scripts for saving predictions, running and ensemble of different fold and for plotting results. 