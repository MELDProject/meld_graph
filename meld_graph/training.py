import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH
from functools import partial
import pandas as pd

def dice_coeff(pred, target,mask=False):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch (not one-hot encoded)
    mask: if mask is true, we want to ignore hemispheres without lesions for the 1 column
        otherwise loss averages lots of 0s for these examples
    NOTE assumes that pred is softmax output of model, might need torch.exp before
    """
    n_vert = 163842
    target_hot = torch.nn.functional.one_hot(target,num_classes=2)
    smooth = 1e-15 
    iflat = pred.contiguous()
    tflat = target_hot.contiguous()
    #here split into subjects
    if mask:
        full_len = iflat.shape
        iflat = iflat.view(n_vert,full_len[0]//n_vert,full_len[1])
        tflat = tflat.view(n_vert,full_len[0]//n_vert,full_len[1])
    intersection = (iflat * tflat).sum(dim=0)
    A_sum = torch.sum(iflat * iflat ,dim=0)
    B_sum = torch.sum(tflat * tflat ,dim=0)
    #at this point mask B_sum all zeros in second column
    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    return  dice


class DiceLoss(torch.nn.Module):
    def __init__(self, loss_weight_dictionary=None):
        super(DiceLoss, self).__init__()
        self.class_weights = [0.5,0.5]
        if 'dice' in loss_weight_dictionary.keys():
            if 'class_weights' in loss_weight_dictionary['dice']:
                self.class_weights = loss_weight_dictionary['dice']['class_weights']

    def forward(self, inputs, targets, mask=False,device=None):
        dice = dice_coeff(torch.exp(inputs),targets,mask=mask)
        if device is not None:
            class_weights = torch.tensor(self.class_weights,dtype=float).to(device)
        dice = dice[0]*class_weights[0] + dice[1]*class_weights[1]
        return 1 - dice

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets):
        # inputs are log softmax, pass directly to NLLLoss
        return self.loss(inputs, targets)

class FocalLoss(torch.nn.Module):
    def __init__(self, params, size_average=True):
        super(FocalLoss, self).__init__()
        try:
            self.gamma = params['focal_loss']['gamma']
        except:
            self.gamma = 0
        try:
            self.alpha = params['focal_loss']['alpha']
        except:
            self.alpha=None
        if isinstance(self.alpha,(float,int)): 
            self.alpha = torch.Tensor([self.alpha,1-self.alpha])
        self.size_average = size_average

    def forward(self, inputs, target, gamma=0, alpha=None,):
        target = target.view(-1,1)
#         logpt = torch.nn.functional.log_softmax(inputs)
        logpt = inputs
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()


def tp_fp_fn(pred, target):
    tp = torch.sum(torch.logical_and((target==1), (pred==1)))
    fp = torch.sum(torch.logical_and((target==0), (pred==1)))
    fn = torch.sum(torch.logical_and((target==1), (pred==0)))
    return tp, fp, fn
    
def calculate_loss(loss_weight_dictionary,estimates,labels, device=None):
    """ 
    calculate loss. Can combine losses with weights defined in loss_weight_dictionary
    loss_dictionary= {'dice':{'weight':1},
                    'cross_entropy':{'weight':1},
                    'focal_loss':{'weight':1, 'alpha':0.5, 'gamma':0},
                    'other_losses':weights}

    NOTE estimates are the logSoftmax output of the model. For some losses, applying torch.exp is necessary!
    """
    # TODO could use class_weights for dice loss (but not using dice loss atm)
    loss_functions = {
        'dice': partial(DiceLoss(loss_weight_dictionary=loss_weight_dictionary),
                         device=device),
        'cross_entropy': CrossEntropyLoss(),
        'focal_loss': FocalLoss(loss_weight_dictionary),
                       }
    total_loss = 0
    for loss_def in loss_weight_dictionary.keys():
        total_loss += loss_weight_dictionary[loss_def]['weight'] * loss_functions[loss_def](estimates,labels)
    return total_loss

class Metrics:
    def __init__(self, metrics):
        self.metrics = metrics
        self.metrics_to_track = self.metrics
        if 'precision' in self.metrics or 'recall' in self.metrics:
            self.metrics_to_track = list(set(self.metrics_to_track + ['tp', 'fp', 'fn']))
        self.running_scores = self.reset()

    def reset(self):
        self.running_scores = {metric: [] for metric in self.metrics_to_track}
        return self.running_scores

    def update(self, pred, target):
        if len(set(['dice_lesion', 'dice_nonlesion']).intersection(self.metrics_to_track)) > 0:
            dice_coeffs = dice_coeff(torch.nn.functional.one_hot(pred), target)
            if 'dice_lesion' in self.metrics_to_track:
                self.running_scores['dice_lesion'].append(dice_coeffs[1].item())
            if 'dice_nonlesion' in self.metrics_to_track:
                self.running_scores['dice_nonlesion'].append(dice_coeffs[0].item())
        if len(set(['dice_masked_lesion', 'dice_masked_nonlesion']).intersection(self.metrics_to_track)) > 0:
            dice_coeffs = dice_coeff(torch.nn.functional.one_hot(pred), target, mask=True)
            if 'dice_masked_lesion' in self.metrics_to_track:
                self.running_scores['dice_masked_lesion_masked'].append(dice_coeffs[1].item())
            if 'dice_masked_nonlesion' in self.metrics_to_track:
                self.running_scores['dice_masked_nonlesion'].append(dice_coeffs[0].item())
        if 'tp' in self.metrics_to_track:
            tp, fp, fn = tp_fp_fn(pred, target)
            self.running_scores['tp'].append(tp.item())
            self.running_scores['fp'].append(fp.item())
            self.running_scores['fn'].append(fn.item())

    def get_aggregated_metrics(self):
        metrics = {}
        if 'tp' in self.metrics_to_track:
            tp = np.sum(self.running_scores['tp'])
            fp = np.sum(self.running_scores['fp'])
            fn = np.sum(self.running_scores['fn'])
        for metric in self.metrics:
            if metric == 'precision':
                metrics['precision'] = (tp/(tp+fp)).item()
            elif metric == 'recall':
                metrics['recall'] = (tp/(tp+fn)).item()
            elif 'dice' in metric:
                metrics[metric] = np.mean(self.running_scores[metric])
            else:
                metrics[metric] = np.sum(self.running_scores[metric])
        return metrics

class Trainer:
    def __init__(self, experiment):
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        self.params = self.experiment.network_parameters['training_parameters']
        self.deep_supervision = self.params.get('deep_supervision', {'levels':[], 'weight': 1})


    def train_epoch(self, data_loader, optimiser):
        """
        train for one epoch. Return loss and metrics
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.experiment.model
        model.train()

        metrics = Metrics(self.params['metrics'])  # for keeping track of running metrics
        running_loss = []
        for i, data in enumerate(data_loader):  
            data = data.to(device)
            model.train()
            optimiser.zero_grad()
            estimates = model(data.x)
            labels = data.y.squeeze()
            loss = calculate_loss(self.params['loss_dictionary'],estimates[0], labels, device=device)
            # add deep supervision outputs # TODO add loss weight param for deep supervision?
            for i,level in enumerate(sorted(self.deep_supervision['levels'])):
                cur_estimates = estimates[i+1]
                cur_labels = getattr(data, f"output_level{level}")
                #print(cur_estimates.shape, cur_labels.shape)
                loss += self.deep_supervision['weight'] * calculate_loss(self.params['loss_dictionary'], cur_estimates, cur_labels, device=device)
            loss.backward()
            optimiser.step()
            running_loss.append(loss.item())
            # metrics
            pred = torch.argmax(estimates[0], axis=1)
            # update running metrics
            metrics.update(pred, labels)
            
        scores = {'loss': np.mean(running_loss)}
        scores.update(metrics.get_aggregated_metrics())
        return scores

    def val_epoch(self, data_loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.experiment.model
        model.eval()
        with torch.no_grad():
            metrics = Metrics(self.params['metrics'])  # for keeping track of running metrics
            running_loss = []
            for i, data in enumerate(data_loader):
                data = data.to(device)
                #fake_x = torch.vstack([data.y for _ in range(22)]).t().type(torch.float)
                estimates = model(data.x)
                labels = data.y.squeeze()
                loss = calculate_loss(self.params['loss_dictionary'],estimates[0], labels, device=device)
                # add deep supervision outputs # TODO add loss weight param for deep supervision?
                for i,level in enumerate(sorted(self.deep_supervision['levels'])):
                    cur_estimates = estimates[i+1]
                    cur_labels = getattr(data, f"output_level{level}")
                    #print(cur_estimates.shape, cur_labels.shape)
                    loss += self.deep_supervision['weight'] * calculate_loss(self.params['loss_dictionary'], cur_estimates, cur_labels, device=device)
                running_loss.append(loss.item())
                # metrics
                pred = torch.argmax(estimates[0], axis=1)
                # update running metrics
                metrics.update(pred, labels)
     
        scores = {'loss': np.mean(running_loss)}
        scores.update(metrics.get_aggregated_metrics())
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
                        fname = os.path.join(self.experiment.experiment_path, 'best_model.pt')
                        torch.save(self.experiment.model.state_dict(), fname)
                        self.log.info('saved_new_best')
                    patience = 0
                else:
                    patience+=1
                if patience >= self.params['max_patience']:
                    self.log.info(f'stopping early at epoch {epoch}, with patience {patience}')
                    break
        # save train/val scores
        if self.experiment.experiment_path is not None:
            pd.DataFrame(scores['train']).to_csv(os.path.join(self.experiment.experiment_path, 'train_scores.csv'))
            pd.DataFrame(scores['val']).to_csv(os.path.join(self.experiment.experiment_path, 'val_scores.csv'))
