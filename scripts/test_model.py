
#import importlib
import meld_graph
import meld_graph.models
import meld_graph.experiment
import meld_graph.dataset
#importlib.reload(meld_graph)
#importlib.reload(meld_graph.models)
#importlib.reload(meld_graph.dataset)
#importlib.reload(meld_graph.experiment)

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    network_parameters = {
    'network_type': 'MoNet',
    'model_parameters': {
        'layer_sizes': [30,30,30],
        'dim': 2, # pseudo-coord dim
    },
    'training_parameters': {
        "max_patience": 10,
        "num_epochs": 200,
        'lr': 1e-3,
        'loss': 'cross_entropy',
        "batch_size": 4,
        "shuffle_each_epoch": True,
    }
    }

    data_parameters = {
    'hdf5_file_root': "{site_code}_{group}_featurematrix.hdf5",
    'site_codes': ['H4'],
    'scanners': ['15T', '3T'],
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
    "combine_hemis": "stack",  # "currently just stack is implemented"
    }


    exp = meld_graph.experiment.Experiment(network_parameters, data_parameters, save=False)
    _ = exp.get_train_val_test_ids()
    exp.data_parameters['train_ids'] = exp.data_parameters['train_ids'][:10]
    exp.data_parameters['val_ids'] = exp.data_parameters['train_ids'][:10]
    ds = meld_graph.dataset.GraphDataset.from_experiment(exp, mode='train')

    exp.train()