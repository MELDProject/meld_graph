import os, datetime

# model and training parameters, passed to model and Trainer, respectively
network_parameters = {
    # network_type: model class, one of: MoNet, MoNetUnet (see models.py)
    'network_type': 'MoNetUnet',
    # model_parameters: passed to model class initialiser
    'model_parameters': {
        # model architecture: list of lists for Unet, and list for MoNet (simple convs)
        'layer_sizes': [[32,32,32],[32,32,32],[64,64,64],[64,64,64],[128,128,128],[128,128,128],[256,256,256]],
        # activation_fn: activation function, one of: relu, leaky_relu
        'activation_fn': 'leaky_relu',
        # conv_type: convolution to use, one of: SpiralConv, GMMConv. Only for MoNetUnet
        'conv_type': 'SpiralConv',
        # dim: coord dim for GMMConv
        'dim': 2,
        # kernel_size: number of gaussian kernels for GMMConv
        'kernel_size': 3, # number of gaussian kernels
        # spiral_len: size of the spiral for SpiralConv. Only for MoNetUnet
        # TODO implement dilation / different spiral len per unet block
        'spiral_len': 10, 
    },
    # training_parameters: used by Trainer to set up model training
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 20,
        # optimiser: optimiser to use, one of: adam, sgd
        "optimiser": 'sgd',
        # optimiser_parameters: parameters passed to torch optimiser class
        # for sgd with nesterov momentum use: momentum:0.99, nesterov:True
        "optimiser_parameters": {
            "lr": 1e-4,
            "momentum": 0.99,
            "nesterov": True
        },
        # lr_decay: exponent for exponential learning rate decay: lr*(1-epoch/max_epochs)**lr_decay
        # set to 0 to turn lr decay off
        'lr_decay': 0.9,
        # loss_dictionary: losses to be used for model training and parameters for losses
        # possible keys: cross_entropy, focal_loss, dice
        # values: dict with keys: "weight" and loss arguments (alpha/gamma for focal_loss, class_weights for dice)
        'loss_dictionary': {  
            'cross_entropy':{'weight':1},
            'dice': {'weight': 1,'class_weights':[0.5,0.5]},
            #'focal_loss':{'weight':1, 'gamma':4, 'alpha': 0.4},
        },
         # metrics: list of metrics that should be printed during training
         # possible values: dice_lesion, dice_nonlesion, precision, recall, tp, fp, fn, tn
        'metrics': ['dice_lesion', 'dice_nonlesion', 'precision', 'recall', 'tp', 'fp', 'fn'], 
        "batch_size": 1,
        "shuffle_each_epoch": True,
        # deep_supervision: add loss at specified levels of the unet (for MoNetUnet).
        # Set to list of levels (eg [6,5,4]), for which to add output layers for additional supervision.
        # 7 is highest level. (standard output).  # TODO add some error checking here, max val should be < 7.
        'deep_supervision': {
            'levels': [], #[4,5,6], 
            'weight': 0.5
            
        },
        'oversampling':True,
    },
    # name: experiment name. If none, experiment is not saved
    'name': datetime.datetime.now().strftime("%y-%m-%d") + '_lr_decay',
}

# data parameters, passed to GraphDataset and Preprocess
data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix_combat_6.hdf5",
    'site_codes': [
   #     "H1",
   #     "H2",
   #     "H3",
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
    'scanners': ['15T','3T'],
    'dataset': 'MELD_dataset_V6.csv',
    'group': 'both',
    "features_to_exclude": [],
    "subject_features_to_exclude": [],
    # features: manually specify features (instead of features_to_exclude)
    "features": [#'.on_lh.lesion.mgh',
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
<<<<<<< HEAD
        "scaling":  None,  #"scaling_params_GDL.json"
        "zscore": False,
=======
        "scaling": None, #"scaling_params_GDL.json"
        # zscore: normalise all values (per subject)
        "zscore": True,
>>>>>>> 1fbe85e44ac714b49ed91a157680b7ff9e3508d0
    },
    # icosphere_parameters: passed to Icospheres class
    "icosphere_parameters": {
        # distance_type: coords to return as edge attributes (for GMMConv), one of: exact, pseudo
        "distance_type": "exact",
    },
<<<<<<< HEAD
     "augment_data":  {
        "spinning": {
                  'p': 0.01,
                  'file': 'data/spinning/spinning_ico7_10.npy'
                  },
         "warping": 
                  {'p': 0.01,
                   'file': 'data/warping/warping_ico7_10.npy'
                  },
                "flipping":
                  {'p': 0.01,
                   'file': 'data/flipping/flipping_ico7_3.npy'
                  },
                    },
    "combine_hemis": None,  # None, "stack", TODO: combine with graph
    "lobes": False, # If true task is frontal lobe parcellation, not lesion segmentation
    "lesion_bias": 10, # value is added to lesion values
=======
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
    "lesion_bias": 10,
>>>>>>> 1fbe85e44ac714b49ed91a157680b7ff9e3508d0
}

# run several experiments
# Nested levels are represented by $
# e.g. "network_parameters$training_parameters$loss_dictionary$focal_loss" will set values for the focal loss.
# if left empty, the above configuration is run.
variable_parameters = {
<<<<<<< HEAD
    'data_parameters$lesion_bias': [10.],
=======
>>>>>>> 1fbe85e44ac714b49ed91a157680b7ff9e3508d0
}
