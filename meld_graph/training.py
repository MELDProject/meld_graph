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

        # get data
        train_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='train'), 
            shuffle=self.params['shuffle_each_epoch'],
            batch_size=self.params['batch_size'])
        val_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode='val'),
            shuffle=False, batch_size=self.params['batch_size'])

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
                    running_metrics = []
                    for i, data in enumerate(val_data_loader):
                        data = data.to(device)
                        estimates = model(data.x)
                        labels = data.y.squeeze()
                        loss = torch.nn.NLLLoss()(estimates, labels)
                        running_losses.append(loss.item())
                        running_metrics.append(dice_coeff(torch.exp(estimates), labels))
                            
                    val_loss = np.mean(running_losses)
                    val_dice = np.mean(running_metrics)
                    validation_losses.append(val_loss)
                    self.log.info('Val loss {:.3f}, val dice {:.3f}'.format(val_loss, val_dice))
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

