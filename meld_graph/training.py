import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH
from functools import partial

def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch (not one-hot encoded)

    NOTE assumes that pred is softmax output of model, might need torch.exp before
    """
    target_hot = torch.nn.functional.one_hot(target,num_classes=2)
    smooth = 1. 
    iflat = pred.contiguous()
    tflat = target_hot.contiguous()
    intersection = (iflat * tflat).sum(dim=0)
    A_sum = torch.sum(iflat * iflat ,dim=0)
    B_sum = torch.sum(tflat * tflat ,dim=0)

    dice = (2. * intersection + smooth) / (A_sum + B_sum + smooth)
    return  dice

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, class_weights=[0.5,0.5], device=None):
        dice = dice_coeff(torch.exp(inputs),targets)
        if device is not None:
            class_weights = torch.tensor(class_weights,dtype=float).to(device)
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
            if isinstance(self.alpha,(float,int)): self.alpha = torch.Tensor([self.alpha,1-self.alpha])
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
    loss_dictionary= {'dice':weight,
                    'cross_entropy':weight,
                    'focal_loss':weight,
                    'other_losses':weights}

    NOTE estimates are the logSoftmax output of the model. For some losses, applying torch.exp is necessary!
    """
    # TODO could use class_weights for dice loss (but not using dice loss atm)
    loss_functions = {
        'dice': partial(DiceLoss(), device=device),
        'cross_entropy': CrossEntropyLoss(),
        'focal_loss': FocalLoss(loss_weight_dictionary),
                       }
    total_loss = 0
    for loss_def in loss_weight_dictionary.keys():
        total_loss += loss_weight_dictionary[loss_def]['weight'] * loss_functions[loss_def](estimates,labels)
    return total_loss

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
            loss = calculate_loss(self.params['loss_dictionary'],estimates, labels, device=device)
            loss.backward()
            optimiser.step()
            running_scores['loss'].append(loss.item())
            # metrics
            pred = torch.argmax(estimates, axis=1)
            # dice
            # TODO dice is on non-thresholded values -- change that for final calc of dice score
            dice_coeffs = dice_coeff(torch.exp(estimates), labels)
            running_scores['dice_lesion'].append(dice_coeffs[1].item())
            running_scores['dice_nonlesion'].append(dice_coeffs[0].item())
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
                loss = calculate_loss(self.params['loss_dictionary'],estimates, labels, device =device)
                running_scores['loss'].append(loss.item())
                # metrics
                pred = torch.argmax(estimates, axis=1)
                # dice
                dice_coeffs = dice_coeff(torch.exp(estimates), labels)
                running_scores['dice_lesion'].append(dice_coeffs[1].item())
                running_scores['dice_nonlesion'].append(dice_coeffs[0].item())
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
