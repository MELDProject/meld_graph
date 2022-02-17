
network_parameters = {
    'network_type': 'MoNetUnet',
    'model_parameters': {
        'layer_sizes':  [[32,32,32],[32,32,32],[64,64,64],[64,64,64],[128,128,128],[128,128,128],[256,256,256]],
        'dim': 2, # coord dim
        'kernel_size': 3, # number of gaussian kernels
        'conv_type': 'GMMConv', # 'GMMConv', #'SpiralConv', # TODO test that
        'spiral_len': 10, # TODO implement dilation / different spiral len 
    },
    'training_parameters': {
        "max_patience": 400,
        "num_epochs": 50,
        'lr': 1e-4,
        'loss_dictionary': {  
#            'cross_entropy':{'weight':1},
      'dice': {'weight': 1,},
        },
        'metrics': [ 'dice_lesion', 'dice_nonlesion','precision', 'recall', 'tp', 'fp', 'fn'], 
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
    "features": [
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
            '.on_lh.wm_FLAIR_1.mgh',
#         '.combat.on_lh.pial.K_filtered.sm20.mgh',
#         '.combat.on_lh.thickness.sm10.mgh',
#         '.combat.on_lh.w-g.pct.sm10.mgh',
#         '.combat.on_lh.sulc.sm5.mgh',
#         '.combat.on_lh.curv.sm5.mgh',
#         '.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
#         '.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
#         '.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
#         '.combat.on_lh.gm_FLAIR_0.sm10.mgh',
#         '.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
#         '.combat.on_lh.wm_FLAIR_1.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
#         '.inter_z.intra_z.combat.on_lh.thickness.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.sulc.sm5.mgh',
#         '.inter_z.intra_z.combat.on_lh.curv.sm5.mgh',
#         '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
#         '.inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.thickness.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.sulc.sm5.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.curv.sm5.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm10.mgh',
#         '.inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm10.mgh',
    ],
    "features_to_replace_with_0": [], # specify this if manually specifying features
    "number_of_folds": 10,
    "fold_n": 0,
    "preprocessing_parameters": {
        "scaling": "scaling_params_GDL.json",
    },
    "icosphere_parameters": {
        "distance_type": "exact", #"exact",  # exact or pseudo
    },
    "combine_hemis": None,  # None, "stack", TODO: combine with graph
    "lobes":True, # "False" if true task is frontal lobe parcellation, not lesion segmentation
}
