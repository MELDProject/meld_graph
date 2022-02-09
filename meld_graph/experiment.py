from meld_graph.paths import EXPERIMENT_PATH
import json
import logging
import os
import torch
import torch_geometric.data
import meld_graph.models
from torch_geometric.transforms import Polar
from meld_classifier.meld_cohort import MeldCohort
from meld_graph.dataset import GraphDataset
import numpy as np

#import torch
#from torch_geometric.data import Data

#from torch_geometric.loader import DataLoader
#def get_fake_edge_index(level=1):
#    # TODO define this for multiple levels for the real data
#    edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1],[0,0],[1,1],[2,2]], dtype=torch.long)
#    # get relative cartensian / polar coords as edge_attrs to use a pseudo coordinates later on
#    position = torch.tensor([[0,0],[1,1],[2,0]], dtype=torch.float)
#    data = Data(edge_index=edge_index.t().contiguous(), pos=position)
#    edge_attr_polar = Polar(norm=True, cat=False)(data).edge_attr  # TODO figure out how exactly want to do coord transform (should be relative difference between nodes)
#    return edge_index.t().contiguous(), edge_attr_polar

#def get_fake_data():
#    # fake some patient data with 3 nodes and per-node labels, and 5 features
#    # build data list
#    data_list = []
#    for i in range(100):
#        x = torch.tensor(np.random.rand(3,5), dtype=torch.float)
#        y = torch.tensor(np.random.choice([0,1], size=(3,1)))
#        data_list.append(Data(x=x, y=y))#

 #   loader = DataLoader(data_list, batch_size=2)

    # can put loader + model on GPU here
    # could use transforms on data loader here
    # example loading of data + custom dataset for brain data: https://github.com/Abdulah-Fawaz/Benchmarking-Surface-DL/blob/main/Segmentation_UGSCNN/GraphMethods/segmentation.py

 #   return loader

class Trainer:
    def __init__(self, experiment):
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        self.params = self.experiment.network_parameters['training_parameters']

    def train(self):
        # set up model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.experiment.load_model()
        model = self.experiment.model
        model.to(device)

        # get data TODO use dataset class here
        train_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='train'), 
            shuffle=self.params['shuffle_each_epoch'],
            batch_size=self.params['batch_size'])
        val_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='val'),
            shuffle=False, batch_size=self.params['batch_size'])
        # get edge_

        # set up training loop
        optimiser = torch.optim.Adam(model.parameters(),lr=self.params['lr'])
    
        validation_losses = []
        train_losses = []
    
        best_loss = 100000
        patience = 0
        for epoch in range(self.params['num_epochs']):
            running_losses = []
            
            for i, data in enumerate(train_data_loader):  
                data = data.to(device)
                model.train()     
                optimiser.zero_grad()
                estimates = model(data.x)
                labels = data.y.squeeze()
                loss = torch.nn.NLLLoss()(estimates, labels)
                loss.backward()
                optimiser.step()
                running_losses.append(loss.item())
          
            self.log.info('Epoch {} :: Train loss {:.3f}'.format(epoch,np.mean(running_losses)))
            train_losses.append(np.mean(running_losses))
        
            if epoch%1 ==0:
                with torch.no_grad():
                    running_losses  = []
                    for i, data in enumerate(val_data_loader):
                        data = data.to(device)
                        estimates = model(data.x)
                        labels = data.y.squeeze()
                        loss = torch.nn.NLLLoss()(estimates, labels)
                        running_losses.append(loss.item())
                            
                    val_loss = np.mean(running_losses)
                    validation_losses.append(val_loss)
                    self.log.info('Val loss {:.3f}'.format(val_loss))
                    # TODO implement dice score
                    # TODO fix logging 
                        
                    if val_loss < best_loss:
                        best_loss = val_loss
                        if self.experiment.experiment_path is not None:
                            # TODO save in correct place?
                            fname = os.path.join(EXPERIMENT_PATH, self.experiment.experiment_path, 'best_model.pt')
                            torch.save(model.state_dict(), fname)
                            self.log.info('saved_new_best')
                        patience = 0
                    else:
                        patience+=1
                    if patience >= self.params['max_patience']:
                        break
                    
                    self.log.info('----------')



class Experiment:
    def __init__(self, network_parameters, data_parameters, save=False):
        self.log = logging.getLogger(__name__)
        self.network_parameters = network_parameters
        self.data_parameters = data_parameters
        self.model = None # loaded by self.load_model()
        self.experiment_path = None
        self.cohort = MeldCohort(
            hdf5_file_root=self.data_parameters["hdf5_file_root"], dataset=self.data_parameters["dataset"]
        )

        if save:
            self.experiment_path = save
            os.makedirs(os.path.join(EXPERIMENT_PATH, self.experiment_path), exist_ok=True)
            self.save_parameters()
        # TODO need to load cohort!

    def from_folder(self, experiment_path):
        # TODO implement -- read files from json
        pass

    def save_parameters(self):
        """
        Save dictionaries to experiment_path using json
        """
        if self.experiment_path is not None:
            self.log.info(f"saving parameter files to {self.experiment_path}")
            # data_parameters
            fname = os.path.join(EXPERIMENT_PATH, self.experiment_path, "data_parameters.json")
            json.dump(self.data_parameters, open(fname, "w"), indent=4)
            # network_parameters
            fname = os.path.join(EXPERIMENT_PATH, self.experiment_path, "network_parameters.json")
            json.dump(self.network_parameters, open(fname, "w"), indent=4)
        else:
            self.log.info("experiment_path is None, could not save parameters")

    def get_features(self):
        """
        get list of features that model should be trained on.
        Either read from data_parameters, or calculated and written to data_parameters
        """
        if "features" not in self.data_parameters:
            self.log.info("get features to train on")
            # get features
            features = self.cohort.get_features(features_to_exclude=self.data_parameters["features_to_exclude"])
            # get features that should be ignored
            _, features_to_ignore = self.cohort._filter_features(
                features_to_exclude=self.data_parameters.get("features_to_replace_with_0", []), return_excluded=True
            )
            self.log.debug(f"features {features}")
            self.log.debug(f"features_to_ignore {features_to_ignore}")

            # put train_ids, val_ids, test_ids, features in data_parameters
            self.data_parameters.update(
                {
                    "features": features,
                    "features_to_replace_with_0": features_to_ignore,
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return self.data_parameters["features"], self.data_parameters["features_to_replace_with_0"]


    def load_model(self, checkpoint_path=None, force=False):
        """
        build model and optionally load weights from checkpoint
        """
        if self.model is not None and not force:
            self.log.info("model already exists. Specify force=True to force reloading and initialisation")

        # build model using network_parameters
        network_type = self.network_parameters['network_type']
        if network_type == 'MoNet':
            num_features = len(self.get_features()[0])
            self.model = meld_graph.models.MoNet(**self.network_parameters['model_parameters'], num_features=num_features) # edge_index_fn=get_fake_edge_index)
        else:
            raise(NotImplementedError, network_type)
        
        # TODO below code is unchecked
        if checkpoint_path is not None and os.path.isdir(checkpoint_path):
            # checkpoint contains both model architecture + weights
            self.log.info("Loading model weights from checkpoint")
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.eval()

    def train(self):
        trainer = Trainer(self)
        trainer.train()

    def get_train_val_test_ids(self):
        """
        return train val test ids.
        Either read from data_parameters (if exist), or created using _train_val_test_split_folds.

        returns train_ids, val_ids, test_ids
        """
        if "train_ids" not in self.data_parameters:
            self.log.info("getting train val test split")
            # get subject ids restricted to desired subjects
            subject_ids = self.cohort.get_subject_ids(**self.data_parameters)
            # get train val test split
            train_ids, val_ids, test_ids = self._train_val_test_split_folds(
                subject_ids,
                iteration=self.data_parameters["fold_n"],
                number_of_folds=self.data_parameters["number_of_folds"],
            )
            # put in data_parameters
            self.data_parameters.update(
                {
                    "train_ids": list(train_ids),
                    "test_ids": list(test_ids),
                    "val_ids": list(val_ids),
                }
            )
            # save updated data_parameters
            self.save_parameters()
        return self.data_parameters["train_ids"], self.data_parameters["val_ids"], self.data_parameters["test_ids"]

    def _train_val_test_split_folds(self, subject_ids, iteration=0, number_of_folds=10):
        """split subject_ids into train val and test.
        test_ids are defined in dataset_name.
        The remaining ids are split randomly (but with a fixed seed) in number_of_folds folds.

        Args:
            list_ids (list of str): subject ids to split
            number_of_folds (int): number of folds to split the train/val ids into
            iteration (int): number of validation fold, values 0,..,number_of_folds-1
        Returns:
            train_ids, val_ids, test_ids
        """
        np.random.seed(0)

        _, dataset_trainval_ids, dataset_test_ids = self.cohort.read_subject_ids_from_dataset()
        subject_ids = np.array(subject_ids)

        # get test_ids
        test_mask = np.in1d(subject_ids, dataset_test_ids)
        test_ids = subject_ids[test_mask]

        # get trainval_ids
        trainval_ids = subject_ids[~test_mask]
        trainval_ids = np.intersect1d(trainval_ids, dataset_trainval_ids)
        # split trainval_ids in folds
        np.random.shuffle(trainval_ids)
        folds = np.array_split(trainval_ids, number_of_folds)
        folds = np.roll(folds, shift=iteration, axis=0)
        train_ids = np.concatenate(folds[0:-1]).ravel()
        val_ids = folds[-1]
        return train_ids, val_ids, test_ids
