import os, datetime

# model and training parameters, passed to model and Trainer, respectively
network_parameters = {
    'network_type': 'MoNetUnet',
    ## model_parameters: passed to model class initialiser
    'model_parameters': {
        # model architecture: list of lists for Unet, and list for MoNet (simple convs)
        'layer_sizes': [[32,32,32],[32,32,32],[64,64,64],[64,64,64],[128,128,128],[128,128,128],[256,256,256]],
        # activation_fn: activation function, one of: relu, leaky_relu
        'activation_fn': 'leaky_relu',
        # conv_type: convolution to use, one of: SpiralConv, GMMConv.
        'conv_type': 'SpiralConv',
        # dim: coord dim for GMMConv
        'dim': 2,
        # kernel_size: number of gaussian kernels for GMMConv
        'kernel_size': 3, # number of gaussian kernels
        # spiral_len: size of the spiral for SpiralConv.
        # TODO implement dilation / different spiral len per unet block
        'spiral_len': 7, 
    },

#     # MONET
#     # network_type: model class, one of: MoNet, MoNetUnet (see models.py)
#     'network_type': 'MoNet',
#     # model_parameters: passed to model class initialiser
#     'model_parameters': {
#         # model architecture: list of lists for Unet, and list for MoNet (simple convs)
#         'layer_sizes': [16,16,16],
#         # activation_fn: activation function, one of: relu, leaky_relu
#         'activation_fn': 'leaky_relu',
#         # conv_type: convolution to use, one of: SpiralConv, GMMConv.
#         'conv_type': 'SpiralConv',
#         # dim: coord dim for GMMConv
#         'dim': 2,
#         # kernel_size: number of gaussian kernels for GMMConv
#         'kernel_size': 3, # number of gaussian kernels
#         # spiral_len: size of the spiral for SpiralConv.
#         # TODO implement dilation / different spiral len per unet block
#         'spiral_len': 7, 
#     },

    # training_parameters: used by Trainer to set up model training
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 50,
        # optimiser: optimiser to use, one of: adam, sgd
        "optimiser": 'sgd',
        # optimiser_parameters: parameters passed to torch optimiser class
        # for sgd with nesterov momentum use: momentum:0.99, nesterov:True
        "optimiser_parameters": {
            "lr": 1e-2,
            "momentum": 0.99,
            "nesterov": True
        },
        # lr_decay: exponent for exponential learning rate decay: lr*(1-epoch/max_epochs)**lr_decay
        # set to 0 to turn lr decay off
        'lr_decay': 0,
        # loss_dictionary: losses to be used for model training and parameters for losses
        # possible keys: cross_entropy, focal_loss, dice
        # values: dict with keys: "weight" and loss arguments (alpha/gamma for focal_loss, class_weights for dice)
        'loss_dictionary': {  
            #'cross_entropy':{'weight':1},
            'focal_loss':{'weight':1, 'alpha':0.4, 'gamma':2},
            #'dice':{'weight': 1, 'class_weights': [0.5, 0.5]}
        },
         # metrics: list of metrics that should be printed during training
         # possible values: dice_lesion, dice_nonlesion, precision, recall, tp, fp, fn, tn
        'metrics': ['dice_lesion', 'dice_nonlesion', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn'], 
        "batch_size": 2,
        "shuffle_each_epoch": True,
        # deep_supervision: add loss at specified levels of the unet (for MoNetUnet).
        # Set to list of levels (eg [6,5,4]), for which to add output layers for additional supervision.
        # 7 is highest level. (standard output).  # TODO add some error checking here, max val should be < 7.
        'deep_supervision': {
            'levels': [], #[4,5,6], 
            'weight': 0.5
        },
        # ovesampling: oversample lesional vertices to 33% lesional and 66% random.
        # size of epoch will be num_lesional_examples * 3
        'oversampling':True,
    },
    # name: experiment name. If none, experiment is not saved
    'name': datetime.datetime.now().strftime("%y-%m-%d") + '_synth_unet_gamma_2',
}

# data parameters, passed to GraphDataset and Preprocess
data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix_combat_6.hdf5",
    'site_codes': [
    #    "H1",
    #    "H2",
    #    "H3",
        "H4",
    #    "H5",
    #    "H6",
    #    "H7",
    #    "H9",
    #    "H10",
    #    "H11",
    #    "H12",
    #    "H14",
    #    "H15",
    #    "H16",
    #    "H17",
    #    "H18",
    #    "H19",
    #    "H21",
    #    "H23",
    #    "H24",
    #    "H26",
    #    "H27",
    ],
    'scanners': ['3T'],
    'dataset': 'MELD_dataset_V6.csv',
    'group': 'control',
    "features_to_exclude": [],
    "subject_features_to_exclude": [],
    # features: manually specify features (instead of features_to_exclude)
    "features": [
       # '.on_lh.lesion.mgh',
        #    '.on_lh.curv.mgh',
        #    '.on_lh.gm_FLAIR_0.25.mgh',
        #    '.on_lh.gm_FLAIR_0.5.mgh',
        #    '.on_lh.gm_FLAIR_0.75.mgh',
        #    '.on_lh.gm_FLAIR_0.mgh',
        #    '.on_lh.pial.K_filtered.sm20.mgh',
        #    '.on_lh.sulc.mgh',
        #    '.on_lh.thickness.mgh',
        #    '.on_lh.w-g.pct.mgh',
        #    '.on_lh.wm_FLAIR_0.5.mgh',
        #    '.on_lh.wm_FLAIR_1.mgh',
        '.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.combat.on_lh.thickness.sm10.mgh',
        '.combat.on_lh.w-g.pct.sm10.mgh',
        '.combat.on_lh.sulc.sm5.mgh',
        '.combat.on_lh.curv.sm5.mgh',
        '.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.combat.on_lh.wm_FLAIR_1.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.inter_z.intra_z.combat.on_lh.thickness.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.sulc.sm5.mgh',
        '.inter_z.intra_z.combat.on_lh.curv.sm5.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.thickness.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.sulc.sm5.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.curv.sm5.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
        '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',
    ],
    # specify this if manually specifying features
    "features_to_replace_with_0": [], 
    "number_of_folds": 10,
    "fold_n": 0,
    # preprocessing_parameters: params for data_preprocessing
    "preprocessing_parameters": {
        "scaling": None, #"scaling_params_GDL.json"
        # zscore: normalise all values by overall mu std. ignores 0s.
        "zscore": True,
    },
    # icosphere_parameters: passed to Icospheres class
    "icosphere_parameters": {
        # distance_type: coords to return as edge attributes (for GMMConv), one of: exact, pseudo
        "distance_type": "exact",
    },
    # augment_data: parameters passed to Augment class
    # dictionary containing augmentation method as keys, and Transform params as values ("p" and "file")
    # possible augmentation methods: spinning, warping, flipping
    "augment_data": {
    },
    # combine_hemis: how to combine hemisphere data, one of: None, stack
    # None: no combination of hemispheres. 
    # "stack": stack features of both hemispheres.
    "combine_hemis": None,
    # WARNING: parameters below change the lesion prediction task
    # lobes: if True, train on predicting frontal lobe vs other instead of the lesion predicting task
    "lobes": False,
    # lesion_bias: add this value to lesion values to make prediction task easier
    "lesion_bias": 0,
    'synthetic_data': {
        #master switch for whether to run the synthetic task. True means run it.
        'run_synthetic':True,
        #controls the number of subjects. randomly sampled from subject ids (i.e. duplicates will exist)
        'n_subs': 200,
        #amount of bias - controls mean from which actual bias per feature will be calculated
        'bias': 1,
        #mean radii of lesions, in units of XX
        'radius':0.5,
        #number of histological subtypes - controls number of "fingerprint" seeds
        #a fingerprint which features change, by how much (multiplied by the bias term).
        'n_subtypes':1,
        #proportion of the features that are abnormal. 0.2 means only 20% of features, all others remain unchanged.
      'proportion_features_abnormal':0.9,
        #proportion subjects lesional, controls a random variable that determines whether a lesion is added to the control data
        #in the training this could mean two hemispheres from the same subject both have lesions
        #I think this is the easiest way to control this.
        'proportion_hemispheres_lesional':0.5 
    }
}

# run several experiments
# Nested levels are represented by $
# e.g. "network_parameters$training_parameters$loss_dictionary$focal_loss" will set values for the focal loss.
# if left empty, the above configuration is run.
variable_parameters = {
    "data_parameters__synthetic_data__bias": [1] , #0.01, 0.05,0.1,1] #, 0.5, 1, 2, 5],
  # "data_parameters__synthetic_data__radius": [0.2] #,0.2,0.3,0.5, 1]
}
