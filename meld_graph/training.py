import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH


def dice_coeff(pred, target, for_class=1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.  # TODO - need this?
    # have to use contiguous since they may from a torch.view op
    iflat = pred.view(-1).contiguous()
    tflat = target.view(-1).contiguous()
    if for_class == 0:  # reverse labels of pred and target 
        iflat = torch.logical_not(iflat)
        tflat = torch.logical_not(tflat)

    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    dice = dice.mean(dim=0)
    dice = torch.clamp(dice, 0, 1.0)
    return  dice

def tp_fp_fn(pred, target):
    tp = torch.sum(torch.logical_and((target==1), (pred==1)))
    fp = torch.sum(torch.logical_and((target==0), (pred==1)))
    fn = torch.sum(torch.logical_and((target==1), (pred==0)))
    return tp, fp, fn
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
        running_scores = {'loss':[], 'dice_lesion':[], 'dice_nonlesion':[]}
        tp, fp, fn = 0,0,0
        for i, data in enumerate(data_loader):  
            data = data.to(device)
            model.train()
            optimiser.zero_grad()
            #fake_x = torch.vstack([data.y for _ in range(22)]).t().type(torch.float)
            #print(data.y.shape)
            #print(fake_x.shape)
            #print(data.x.shape)
            estimates = model(data.x)
            #estimates = model(data.x)
            labels = data.y.squeeze()
            loss = torch.nn.NLLLoss()(estimates, labels)
            loss.backward()
            optimiser.step()
            running_scores['loss'].append(loss.item())
            # metrics
            pred = torch.argmax(torch.exp(estimates), axis=1)
            # dice
            running_scores['dice_lesion'].append(dice_coeff(pred, labels).item())
            running_scores['dice_nonlesion'].append(dice_coeff(pred, labels, for_class=0).item())
            # tp, fp, fn for precision/recall
            cur_tp, cur_fp, cur_fn = tp_fp_fn(pred, labels)
            tp+= cur_tp
            fp+= cur_fp
            fn+= cur_fn
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        scores = {key: np.mean(val) for key, val in running_scores.items()} 
        scores.update({'precision': precision.item(), 'recall': recall.item(), 'tp': tp.item(), 'fp': fp.item(), 'fn': fn.item()})
        return scores

    def val_epoch(self, data_loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.experiment.model
        model.eval()
        tp, fp, fn = 0,0,0
        with torch.no_grad():
            running_scores = {'loss':[], 'dice_lesion':[], 'dice_nonlesion':[]}
            for i, data in enumerate(data_loader):
                data = data.to(device)
                #fake_x = torch.vstack([data.y for _ in range(22)]).t().type(torch.float)
                estimates = model(data.x)
                labels = data.y.squeeze()
                loss = torch.nn.NLLLoss()(estimates, labels)
                running_scores['loss'].append(loss.item())
                # metrics
                pred = torch.argmax(torch.exp(estimates), axis=1)
                # dice
                running_scores['dice_lesion'].append(dice_coeff(pred, labels).item())
                running_scores['dice_nonlesion'].append(dice_coeff(pred, labels, for_class=0).item())
                # tp, fp, fn for precision/recall
                cur_tp, cur_fp, cur_fn = tp_fp_fn(pred, labels)
                tp+= cur_tp
                fp+= cur_fp
                fn+= cur_fn
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        scores = {key: np.mean(val) for key, val in running_scores.items()}
        scores.update({'precision': precision.item(), 'recall': recall.item(), 'tp': tp.item(), 'fp': fp.item(), 'fn': fn.item()})
        # set model back to training mode
        model.train()
        return scores
        

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
        # TODO store train/val loss and metrics in csv file
