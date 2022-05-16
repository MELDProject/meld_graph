import datetime
import os

network_parameters = {
    'network_type': 'MoNetUnet',
    'model_parameters': {
        'layer_sizes': [[32,32,32],[32,32,32],[64,64,64],[64,64,64],[128,128,128],[128,128,128],[256,256,256]],
        'dim': 2, # coord dim
        'kernel_size': 3, # number of gaussian kernels
        'conv_type': 'SpiralConv', #'SpiralConv', # TODO test that
        'spiral_len': 10, # TODO implement dilation / different spiral len 
    },
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 20,
        'lr': 1e-3,
        'loss_dictionary': {  
            'cross_entropy':{'weight':1},
            'dice': {'weight': 1,'class_weights':[0.5,0.5]},
            'focal_loss':{'weight':1, 'gamma':4, 'alpha': 0.4},
        },
        # list of metrics that should be printed during training
        'metrics': ['dice_lesion', 'dice_nonlesion', 'precision', 'recall', 'tp', 'fp', 'fn'], 
        "batch_size": 1,
        "shuffle_each_epoch": True,
        # Set to list of levels (eg [6,5,4]), for which to add output layers for additional supervision.
        # 7 is highest level. (standard output).  # TODO add some error checking here, max val should be < 7.
        'deep_supervision': {
            'levels': [4,5,6], 
            'weight': 0.5
        }
    },
    # experiment name. If none, experiment is not saved
    'name': None #datetime.datetime.now().strftime("%y-%m-%d") + '_full_cohort_deepsup',
}

data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix_combat_6.hdf5",
    'site_codes': [
        "H1",
        "H2",
        "H3",
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
    "features": [
#             '.on_lh.curv.mgh',
#             '.on_lh.gm_FLAIR_0.25.mgh',
#             '.on_lh.gm_FLAIR_0.5.mgh',
#             '.on_lh.gm_FLAIR_0.75.mgh',
#             '.on_lh.gm_FLAIR_0.mgh',
#             '.on_lh.pial.K_filtered.sm20.mgh',
#             '.on_lh.sulc.mgh',
#             '.on_lh.thickness.mgh',
#             '.on_lh.w-g.pct.mgh',
#             '.on_lh.wm_FLAIR_0.5.mgh',
#             '.on_lh.wm_FLAIR_1.mgh',
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
    "features_to_replace_with_0": [], # specify this if manually specifying features
    "number_of_folds": 10,
    "fold_n": 0,
    "preprocessing_parameters": {
        "scaling":  None,  #"scaling_params_GDL.json"
        "zscore": True,
    },
    "icosphere_parameters": {
        "distance_type": "exact", #"exact",  # exact or pseudo
    },
    "augment_data": {
        "spinning": None
    },
    "combine_hemis": None,  # None, "stack", TODO: combine with graph
    "lobes": False, # If true task is frontal lobe parcellation, not lesion segmentation
    "lesion_bias": False, # value is added to lesion values
}

variable_parameters = {
    'data_parameters$lesion_bias': [0.0,0.2,0.4,0.6],
}
