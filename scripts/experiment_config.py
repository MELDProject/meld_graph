
network_parameters = {
    'network_type': 'MoNetUnet',
    'model_parameters': {
        'layer_sizes': [[32],[32],[64],[64],[128],[128],[256]],
        'dim': 2, # coord dim
        'kernel_size': 3, # number of gaussian kernels
    },
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 800,
        'lr': 1e-2,
        'loss_dictionary': {  
            'cross_entropy':1,
            #'dice':1000.0
        },
        "batch_size": 1,
        "shuffle_each_epoch": True,
    },
    # experiment name. If none, experiment is not saved TODO implement
    'name': None,   #"date": datetime.datetime.now().strftime("%y-%m-%d"),
}

data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix.hdf5",
    'site_codes': ['H4'],
    'scanners': ['3T'],
    'dataset': 'MELD_dataset_V6.csv',
    'group': 'both',
    "features_to_exclude": [],
    "subject_features_to_exclude": [],
    "features": ['.on_lh.curv.mgh',
        '.on_lh.gm_FLAIR_0.25.mgh',
        '.on_lh.gm_FLAIR_0.5.mgh',
        '.on_lh.gm_FLAIR_0.75.mgh',
        '.on_lh.gm_FLAIR_0.mgh',
        '.on_lh.pial.K_filtered.sm20.mgh',
        '.on_lh.sulc.mgh',
        '.on_lh.thickness.mgh',
        '.on_lh.w-g.pct.mgh',
        '.on_lh.wm_FLAIR_0.5.mgh',
        '.on_lh.wm_FLAIR_1.mgh'],
    "features_to_replace_with_0": [], # specify this if manually specifying features
    "number_of_folds": 10,
    "fold_n": 0,
    "preprocessing_parameters": {
        "scaling": "scaling_params_GDL.json"
    },
    "icosphere_parameters": {
        "distance_type": "exact", #"exact",  # exact or pseudo
    },
    "combine_hemis": None,  # None, "stack", TODO: combine with graph
}
