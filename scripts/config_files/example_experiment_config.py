import os, datetime

network_parameters = {
    # -- type of model used, MoNet or MoNetUnet --
    'network_type': 'MoNetUnet',
    'model_parameters': {
        # model architecture: list of lists for Unet, and list for MoNet (simple convs)
        'layer_sizes': [[32],[32],[64],[64],[128],[128],[256]],
        # convolution to use: SpiralConv or GMMConv
        'conv_type': 'SpiralConv',
        # coord dim for GMMConv
        'dim': 2, # coord dim
        # kernel size of GMMConv
        'kernel_size': 3, # number of gaussian kernels
        # size of the spiral for SpiralConv
        'spiral_len': 10, # TODO implement dilation / different spiral len per unet block
    },
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 800,
        'lr': 1e-2,
        # losses to be used for model training and parameters for losses
        'loss_dictionary': {  
            #'cross_entropy':{'weight':1},
            #'focal_loss':{'weight':1, 'alpha':0.01, 'gamma':2},
            'dice':{'weight': 1, 'class_weights': [0.5, 0.5]}
        },
         # list of metrics that should be printed during training
        'metrics': ['dice_lesion', 'dice_nonlesion', 'precision', 'recall', 'tp', 'fp', 'fn'], 
        "batch_size": 1,
        "shuffle_each_epoch": True,
    },
    # experiment name. If none, experiment is not saved TODO implement
    'name': datetime.datetime.now().strftime("%y-%m-%d") + '_example',
}

data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix.hdf5",
    'site_codes': ['H4'],
    'scanners': ['15T','3T'],
    'dataset': 'MELD_dataset_V6.csv',
    'group': 'both',
    "features_to_exclude": [],
    "subject_features_to_exclude": [],
    # manually specify features (instead of features_to_exclude)
    "features": [#'.on_lh.lesion.mgh',
            '.on_lh.curv.mgh',
            '.on_lh.gm_FLAIR_0.25.mgh',
            '.on_lh.gm_FLAIR_0.5.mgh',
            '.on_lh.gm_FLAIR_0.75.mgh',
            '.on_lh.gm_FLAIR_0.mgh',
            '.on_lh.pial.K_filtered.sm20.mgh',
            '.on_lh.sulc.mgh',
            '.on_lh.thickness.mgh',
            '.on_lh.w-g.pct.mgh',
            '.on_lh.wm_FLAIR_0.5.mgh',
            '.on_lh.wm_FLAIR_1.mgh'
    ],
    "features_to_replace_with_0": [], # specify this if manually specifying features
    "number_of_folds": 10,
    "fold_n": 0,
    # -- params for data_preprocessing --
    "preprocessing_parameters": {
        "scaling": None, #"scaling_params_GDL.json"
    },
    # -- params for IcoSpheres --
    "icosphere_parameters": {
        # coords to return as edge attributes (for GMMConv)
        "distance_type": "exact", #"exact",  # exact or pseudo
    },
    # -- how to combine hemisphere data --
    # None: no combination of hemispheres. 
    # "stack": stack features of both hemispheres.
    "combine_hemis": None,  # None, "stack", TODO: combine with graph
    # WARNING: if True, will train on predicting frontal lobe vs other instead of the lesion predicting task
    "lobes": True
}
