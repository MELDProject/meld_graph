import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    epsilon = 10e-8

    # have to use contiguous since they may from a torch.view op
    iflat = pred.view(-1).contiguous()
    tflat = target.view(-1).contiguous()
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0)

    return  dice

def precision_recall(pred, target):
    tp = torch.sum(torch.logical_and((target==1), (pred==1)))
    fp = torch.sum(torch.logical_and((target==0), (pred==1)))
    fn = torch.sum(torch.logical_and((target==1), (pred==0)))
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall
    

class Trainer:
    def __init__(self, experiment):
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        self.params = self.experiment.network_parameters['training_parameters']

    def train_epoch(self, data_loader, optimiser):
        """
        train for one epoch. Return loss and metrics
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.experiment.model
        model.train()

        # TODO also measure acc + dice
        running_scores = {'loss':[], 'dice':[], 'precision':[], 'recall':[]}
        for i, data in enumerate(data_loader):  
            data = data.to(device)
            model.train()
            optimiser.zero_grad()
            estimates = model(data.x)
            labels = data.y.squeeze()
            loss = torch.nn.NLLLoss()(estimates, labels)
            loss.backward()
            optimiser.step()
            running_scores['loss'].append(loss.item())
            # metrics
            pred = torch.argmax(torch.exp(estimates), axis=1)
            running_scores['dice'].append(dice_coeff(pred, labels).item())
            precision, recall = precision_recall(pred, labels)
            running_scores['precision'] = precision.item()
            running_scores['recall'] = recall.item()
        return {key: np.mean(val) for key, val in running_scores.items()}

    def val_epoch(self, data_loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.experiment.model
        model.eval()
        with torch.no_grad():
            running_scores = {'loss':[], 'dice':[], 'precision':[], 'recall':[]}
            for i, data in enumerate(data_loader):
                data = data.to(device)
                estimates = model(data.x)
                labels = data.y.squeeze()
                loss = torch.nn.NLLLoss()(estimates, labels)
                running_scores['loss'].append(loss.item())
                # metrics
                pred = torch.argmax(torch.exp(estimates), axis=1)
                running_scores['dice'].append(dice_coeff(pred, labels).item())
                precision, recall = precision_recall(pred, labels)
                running_scores['precision'] = precision.item()
                running_scores['recall'] = recall.item()

        model.train()
        return {key: np.mean(val) for key, val in running_scores.items()}
        

    def train(self):
        """
        Train val loop with patience and best model saving
        """
        # set up model & put on correct device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.experiment.load_model()
        self.experiment.model.to(device)

        # get data
        train_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='train'), 
            shuffle=self.params['shuffle_each_epoch'],
            batch_size=self.params['batch_size'])
        val_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='val'),
            shuffle=False, batch_size=self.params['batch_size'])

        # set up training loop
        optimiser = torch.optim.Adam(self.experiment.model.parameters(), lr=self.params['lr'])
    
        scores = {'train':[], 'val':[]}
        best_loss = 100000
        patience = 0
        for epoch in range(self.params['num_epochs']):
            cur_scores = self.train_epoch(train_data_loader, optimiser)
            log_str = ", ".join(f"{key} {val:.3f}" for key, val in cur_scores.items())
            self.log.info(f'Epoch {epoch} :: Train {log_str}')
            scores['train'].append(cur_scores)
        
            if epoch%1 ==0:
                cur_scores = self.val_epoch(val_data_loader)
                log_str = ", ".join(f"{key} {val:.3f}" for key, val in cur_scores.items())
                self.log.info(f'Epoch {epoch} :: Val   {log_str}')
                scores['val'].append(cur_scores)
                
                if cur_scores['loss'] < best_loss:
                    best_loss = cur_scores['loss']
                    if self.experiment.experiment_path is not None:
                        # TODO save in correct place?
                        fname = os.path.join(EXPERIMENT_PATH, self.experiment.experiment_path, 'best_model.pt')
                        torch.save(self.experiment.model.state_dict(), fname)
                        self.log.info('saved_new_best')
                    patience = 0
                else:
                    patience+=1
                if patience >= self.params['max_patience']:
                    self.log.info(f'stopping early at epoch {epoch}, with patience {patience}')
                    break
        #print(scores)
        # TODO store train/val loss and metrics in csv file
