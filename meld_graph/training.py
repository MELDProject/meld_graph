import logging
import os, psutil
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset, Oversampler
import numpy as np
from meld_graph.paths import EXPERIMENT_PATH
from functools import partial
import pandas as pd
import time
from scipy.ndimage import gaussian_filter1d
import sklearn.metrics as skmetrics


def dice_coeff(pred, target, smooth=1e-15):
    """
    Differentiable Dice coefficient implementation.

    This definition generalizes to real valued pred and target vector.
    NOTE assumes that pred is softmax output of model, might need torch.exp on pred before.

    Args:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch (not one-hot encoded)
    """
    # make target one-hot encoded (also works for soft targets)
    target_hot = torch.transpose(torch.stack((1 - target, target)), 0, 1)

    iflat = pred.contiguous()
    tflat = target_hot.contiguous()
    intersection = (iflat * tflat).sum(dim=0)
    A_sum = torch.sum(iflat * iflat, dim=0)
    B_sum = torch.sum(tflat * tflat, dim=0)
    dice = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)
    return dice


class DiceLoss(torch.nn.Module):
    """
    Dice loss.

    Args:
        loss_weight_dictionary (dict): loss dict from experiment_config.py
    """

    def __init__(self, loss_weight_dictionary=None):
        super(DiceLoss, self).__init__()
        self.class_weights = [0.0, 1.0]
        if "dice" in loss_weight_dictionary.keys():
            if "class_weights" in loss_weight_dictionary["dice"]:
                self.class_weights = loss_weight_dictionary["dice"]["class_weights"]
            if "epsilon" in loss_weight_dictionary["dice"]:
                self.epsilon = loss_weight_dictionary["dice"]["epsilon"]
            else:
                self.epsilon = 1e-15

    def forward(self, inputs, targets, device=None, **kwargs):
        dice = dice_coeff(torch.exp(inputs), targets, smooth=self.epsilon)
        class_weights = torch.tensor(self.class_weights, dtype=float)
        if device is not None:
            class_weights = class_weights.to(device)

        dice = dice[0] * class_weights[0] + dice[1] * class_weights[1]
        return 1 - dice


class CrossEntropyLoss(torch.nn.Module):
    """
    Cross entropy loss (NLLLoss).
    """

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        return self.loss(inputs, targets)


class SoftCrossEntropyLoss(torch.nn.Module):
    """
    Soft version of cross entropy loss.
    Equivalent to CE if labels/targets are hard.
    """

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, do not need to log
        # formula: non-lesional (inputs[:0]) + lesional (inputs[:1])
        ce = -(1 - targets) * inputs[:, 0] - targets * inputs[:, 1]
        return torch.mean(ce)


class MAELoss(torch.nn.Module):
    """
    L1 loss.
    """

    def __init__(self, weight=None, size_average=True):
        super(MAELoss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, inputs, targets, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        return self.loss(inputs, targets)

class SmoothL1Loss(torch.nn.Module):
    """
    Smooth L1 loss for object detection. includes mask
    """
    def __init__(self, weight=None, size_average=True):
        super(SmoothL1Loss, self).__init__()
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, inputs, targets,xyzr, **kwargs):
        # inputs are log softmax, pass directly to NLLLoss
        # mask out non-lesional examples from the loss using targets  
        xyzr_reshaped = xyzr.view(-1,4)
        inputs = inputs * targets.unsqueeze(1).float()
        xyzr_reshaped = xyzr_reshaped * targets.unsqueeze(1).float()
        loss = self.loss(inputs, xyzr_reshaped)
       # masked_loss = loss * targets.float()  # Apply binary mask
        return loss
    

class DistanceRegressionLoss(torch.nn.Module):
    """
    Distance regression loss. Either MSE, MAE, MLE
    Args:
        params (dict): loss dict from experiment_config.py
    """

    def __init__(self, params):
        super(DistanceRegressionLoss, self).__init__()
        if "distance_regression" in params.keys():
            self.weigh_by_gt = params["distance_regression"].get("weigh_by_gt", False)
            self.loss = params["distance_regression"].get("loss", "mse")
            assert self.loss in ["mse", "mae", "mle"]
        else:
            self.weigh_by_gt = False
            self.loss='mse'
    
    def forward(self, inputs, target, distance_map, **kwargs):
        inputs = torch.squeeze(inputs)
        # normalise distance map
        distance_map = torch.div(distance_map, 300)
        # calculate loss
        if self.loss == "mse":
            loss = torch.square(torch.subtract(inputs, distance_map))
        elif self.loss == "mae":
            loss = torch.abs(torch.subtract(inputs, distance_map))
        elif self.loss == "mle":
            loss = torch.log(torch.add(torch.abs(torch.subtract(inputs, distance_map)), 1))
        # weigh loss
        if self.weigh_by_gt:
            loss = torch.div(loss, torch.add(distance_map, 1))
        loss = loss.mean()
        return loss


class FocalLoss(torch.nn.Module):
    """
    Focal loss.
    """

    def __init__(self, params, size_average=True):
        super(FocalLoss, self).__init__()
        try:
            self.gamma = params["focal_loss"]["gamma"]
        except:
            self.gamma = 0
        try:
            self.alpha = params["focal_loss"]["alpha"]
        except:
            self.alpha = None
        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.Tensor([self.alpha, 1 - self.alpha])
        self.size_average = size_average

    def forward(self, inputs, target, **kwargs):
        target = target.view(-1, 1)
        logpt = inputs
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def get_sensitivity(pred, target):
    """
    Sample-level sensitivity.
    Returns 1 if any TP, 0 otherwise.
    """
    if torch.sum(torch.logical_and((target == 1), (pred == 1))) > 0:
        return 1
    else:
        return 0


def tp_fp_fn_tn(pred, target):
    """
    Returns TP, FP, FN, TN.
    """
    tp = torch.sum(torch.logical_and((target == 1), (pred == 1)))
    fp = torch.sum(torch.logical_and((target == 0), (pred == 1)))
    fn = torch.sum(torch.logical_and((target == 1), (pred == 0)))
    tn = torch.sum(torch.logical_and((target == 0), (pred == 0)))
    return tp, fp, fn, tn
    

def calculate_loss(loss_dict, estimates_dict, labels,
                    distance_map=None, xyzr=None,
                    deep_supervision_level=None, device=None, n_vertices=None):
    """ 
    Calculate loss. Can combine losses with weights defined in loss_dict

    Example loss_dict:
    ```
    loss_dict = {
        'cross_entropy':{'weight':1},
        'focal_loss':{'weight':1, 'alpha':0.4, 'gamma':4},
        'dice':{'weight': 1, 'class_weights': [0.0, 1.0]},
        'distance_regression': {'weight': 1, 'weigh_by_gt': True},
        'lesion_classification': {'weight': 1, 'apply_to_bottleneck': True}
        }
    ```
    NOTE Estimates are the logSoftmax output of the model. For some losses, applying torch.exp is necessary!

    Args:
        loss_dict (dict): define losses that should be caluclated.
        estimates_dict (dict): model outputs dictionary. 
        labels (tensor): groundtruth lesion labels.
        distance_map (optional, tensor): groundtruth distance map. 
        xyzr (optional, tensor): groundtruth xyzr coordinates for object detection
        deep_supervision_level (optional, int): calculate_loss is called for every deep supervision level. 
            This arg indicates which level we are currenly at.
            Used to get the correct outputs from estimates_dict.
        n_vertices: number of vertices at current level.
    """
    loss_functions = {
        'dice': partial(DiceLoss(loss_weight_dictionary=loss_dict),
                         device=device),
        'cross_entropy': CrossEntropyLoss(),
        'soft_cross_entropy': SoftCrossEntropyLoss(),
        'focal_loss': FocalLoss(loss_dict),
        'distance_regression': DistanceRegressionLoss(loss_dict),
        'lesion_classification': CrossEntropyLoss(),
        'mae_loss': MAELoss(),
        'object_detection': SmoothL1Loss()
    }
    if distance_map is not None:
        distance_map.to(device)
    losses = {}
    for loss_def in loss_dict.keys():
        # TODO if deep supverision level
        if deep_supervision_level is None:
            prefix = ""
        else:
            prefix = f"ds{deep_supervision_level}_"

        cur_labels = labels
        if loss_def in ['dice', 'cross_entropy', 'focal_loss','mae_loss','soft_cross_entropy']:
            cur_estimates = estimates_dict[f'{prefix}log_softmax']
        elif loss_def == 'distance_regression':
            cur_estimates = estimates_dict[f'{prefix}non_lesion_logits']
        elif loss_def == 'object_detection':
            #object detection only on bottleneck. pass current labels for classification mask
            if deep_supervision_level is not None:
                continue
            else:
                cur_estimates = estimates_dict['object_detection_linear']
                cur_labels = torch.any(labels.view(labels.shape[0]//n_vertices, -1), dim=1).long()
        elif loss_def == 'lesion_classification':
            if loss_dict[loss_def].get('apply_to_bottleneck', False):
                # if apply lc to bottleneck, do not apply it on deep supervision levels
                if deep_supervision_level is not None:
                    continue
                else:
                    # on highest level, can apply lc
                    cur_estimates = estimates_dict["hemi_log_softmax"]
            else:
                cur_estimates = estimates_dict[f"{prefix}log_sumexp"]
            cur_labels = torch.any(labels.view(labels.shape[0] // n_vertices, -1), dim=1).long()
        else:
            raise NotImplementedError(f"Unknown loss def {loss_def}")

        losses[loss_def] = loss_dict[loss_def]['weight'] * loss_functions[loss_def](cur_estimates, cur_labels, 
                                                                                    distance_map=distance_map,xyzr=xyzr)
    return losses


class Metrics:
    """
    Metrics tracked during training.
    """

    def __init__(self, metrics, n_vertices=None, device=None):
        self.device = device
        self.metrics = metrics
        self.metrics_to_track = self.metrics
        if "precision" in self.metrics or "recall" in self.metrics:
            self.metrics_to_track = list(set(self.metrics_to_track + ["tp", "fp", "fn", "tn"]))
        if "cl_precision" in self.metrics or "cl_recall" in self.metrics:
            self.metrics_to_track = list(set(self.metrics_to_track + ["cl_tp", "cl_fp", "cl_fn", "cl_tn"]))
        if "auroc" in self.metrics:
            import torchmetrics

            self.auroc = torchmetrics.AUROC(task="binary", thresholds=10).to(self.device)
        if 'sub_auroc' in self.metrics_to_track:
            self.n_thresh=21
            self.roc_dictionary = {"sensitivity_plus": np.zeros(self.n_thresh),
                                   "specificity":np.zeros(self.n_thresh)}
        self.running_scores = self.reset()
        self.n_vertices = n_vertices

    def reset(self):
        self.running_scores = {metric: [] for metric in self.metrics_to_track}
        if "sub_auroc" in self.metrics_to_track:
            self.roc_dictionary = {"sensitivity_plus": np.zeros(self.n_thresh),
                                   "specificity":np.zeros(self.n_thresh)}
        return self.running_scores

    def update(self, pred, target, pred_class, estimates, borderzone=None):
        if len(set(["dice_lesion", "dice_nonlesion"]).intersection(self.metrics_to_track)) > 0:
            dice_coeffs = dice_coeff(torch.nn.functional.one_hot(pred, num_classes=2), target)
            if "dice_lesion" in self.metrics_to_track:
                self.running_scores["dice_lesion"].append(dice_coeffs[1].item())
            if "dice_nonlesion" in self.metrics_to_track:
                self.running_scores["dice_nonlesion"].append(dice_coeffs[0].item())
        if "tp" in self.metrics_to_track:
            tp, fp, fn, tn = tp_fp_fn_tn(pred, target)
            self.running_scores["tp"].append(tp.item())
            self.running_scores["fp"].append(fp.item())
            self.running_scores["fn"].append(fn.item())
            self.running_scores["tn"].append(tn.item())
        if "sensitivity" in self.metrics_to_track:
            self.running_scores["sensitivity"].append(get_sensitivity(pred, target))
        if "auroc" in self.metrics_to_track:
            # binarise target because might be soft target
            cur_auroc = self.auroc(torch.exp(estimates[:, 1]), target > 0.5)
            self.running_scores["auroc"].append(cur_auroc.item())
        if "sub_auroc" in self.metrics_to_track:
            roc_curves_thresholds = np.linspace(0, 1, self.n_thresh)
            continuous_predictions = torch.exp(estimates[:, 1])
            #paired hemispheres into subjects
            reshaped_preds = continuous_predictions.view(continuous_predictions.shape[0] // (2 * self.n_vertices), -1)
            reshaped_borders = borderzone.view(borderzone.shape[0] // (2 * self.n_vertices), -1)
            for example_index, prediction in enumerate(reshaped_preds):
                sub_borderzone = reshaped_borders[example_index]
                self.roc_curves(prediction,sub_borderzone,roc_curves_thresholds)
        # classification metrics
        target_class = torch.any(target.view(target.shape[0] // self.n_vertices, -1), dim=1).long()
        if "cl_tp" in self.metrics_to_track:
            tp, fp, fn, tn = tp_fp_fn_tn(pred_class, target_class)
            self.running_scores["cl_tp"].append(tp.item())
            self.running_scores["cl_fp"].append(fp.item())
            self.running_scores["cl_fn"].append(fn.item())
            self.running_scores["cl_tn"].append(tn.item())


    def roc_curves(self, prediction, 
                   borderzone, 
                   thresholds):
        """calculate performance at multiple thresholds"""
        for t_i, threshold in enumerate(thresholds):
            predicted = prediction >= threshold
            # if we want tpr vs fpr curve too
            # tp,fp,fn, tn = tp_fp_fn_tn(predicted, subject_dictionary['input_labels'])
            # store sensitivity and sensitivity_plus for each patient (has a label)
            if borderzone.sum() > 0:
                if torch.sum(torch.logical_and((borderzone == 1),
                                                (predicted == 1))) > 0:
                    self.roc_dictionary["sensitivity_plus"][t_i] += 1
                #if no overlaps exists then exit
                else :
                    break
            # store specificity for controls (no label)
            else:
                #if any false positives, specificity not added to
                if  predicted.any():
                    pass
                else:
                    # if not false positives, add 1 to all and then break
                    self.roc_dictionary["specificity"][t_i:] += 1
                    break
        return
                
    def calculate_sub_auroc(self):
        """calculate subject level auroc"""
        # normalise to 0-1
        x = 1 - self.roc_dictionary["specificity"] / self.roc_dictionary["specificity"][-1]
        y2 = self.roc_dictionary["sensitivity_plus"] / self.roc_dictionary["sensitivity_plus"][0]
        auc = skmetrics.auc(x, y2)
        return auc

    def get_aggregated_metrics(self):
        metrics = {}
        if "tp" in self.metrics_to_track:
            tp = np.sum(self.running_scores["tp"])
            fp = np.sum(self.running_scores["fp"])
            fn = np.sum(self.running_scores["fn"])
        if "cl_tp" in self.metrics_to_track:
            cl_tp = np.sum(self.running_scores["cl_tp"])
            cl_fp = np.sum(self.running_scores["cl_fp"])
            cl_fn = np.sum(self.running_scores["cl_fn"])
        for metric in self.metrics:
            if metric == "precision":
                metrics["precision"] = (tp / (tp + fp)).item()
            elif metric == "recall":
                metrics["recall"] = (tp / (tp + fn)).item()
            elif metric == "sub_auroc":
                metrics["sub_auroc"] = self.calculate_sub_auroc()
            elif ("dice" in metric) or ("auroc" in metric):
                metrics[metric] = np.mean(self.running_scores[metric])
            elif "sensitivity" in metric:
                sensitivity = self.running_scores["sensitivity"]
                metrics["sensitivity"] = np.sum(sensitivity) / len(sensitivity) * 100
            elif metric == "cl_precision":
                metrics["cl_precision"] = (cl_tp / (cl_tp + cl_fp)).item()
            elif metric == "cl_recall":
                metrics["cl_recall"] = (cl_tp / (cl_tp + cl_fn)).item()

            else:
                metrics[metric] = np.sum(self.running_scores[metric])

        return metrics


class Trainer:
    """Training class.

    Args:
        experiment (Experiment): experiment to train."""

    def __init__(self, experiment):
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        self.params = self.experiment.network_parameters["training_parameters"]
        self.deep_supervision = self.params.get("deep_supervision", {"levels": [], "weight": []})

        init_weights = self.params.get("init_weights", None)
        if init_weights is not None:
            init_weights = os.path.join(EXPERIMENT_PATH, init_weights)
            assert os.path.isfile(init_weights), f"Weights file {init_weights} does not exist"
        self.init_weights = init_weights

    def train_epoch(self, data_loader, optimiser):
        """
        Train for one epoch.

        Returns:
            scores (dict): loss and metrics
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.experiment.model
        model.train()
        train_metrics = self.params["metrics"].copy()
        #remove sub_auroc from train as it makes no sense to track.
        if 'sub_auroc' in train_metrics:
            train_metrics.remove('sub_auroc')
        metrics = Metrics(
            train_metrics,
            n_vertices=self.experiment.model.n_vertices,
            device=device,
        )  # for keeping track of running metrics
        running_losses = {key: [] for key in self.params["loss_dictionary"].keys()}
        for key in list(running_losses.keys()):
            for level in self.deep_supervision["levels"]:
                running_losses[f"ds{level}_{key}"] = []
        running_losses["loss"] = []
        for i, data in enumerate(data_loader):
            data = data.to(device)
            model.train()
            optimiser.zero_grad()
            estimates = model(data.x)
            labels = data.y.squeeze()
            distance_map = getattr(data, "distance_map", None)
            xyzr = getattr(data, "xyzr", None)
            losses = calculate_loss(self.params['loss_dictionary'], estimates, labels,
                                     distance_map=distance_map,xyzr=xyzr,
                                       deep_supervision_level=None, device=device, 
                n_vertices=self.experiment.model.n_vertices)
            # add deep supervision outputs
            for i, level in enumerate(sorted(self.deep_supervision["levels"])):
                cur_labels = getattr(data, f"output_level{level}")
                cur_distance_map = getattr(data, f"output_level{level}_distance_map", None)
                n_vertices = len(self.experiment.model.icospheres.icospheres[level]["coords"])
                ds_losses = calculate_loss(
                    self.params["loss_dictionary"],
                    estimates,
                    cur_labels,
                    distance_map=cur_distance_map,
                    deep_supervision_level=level,
                    device=device,
                    n_vertices=n_vertices,
                )
                losses.update(
                    {f"ds{level}_{key}": self.deep_supervision["weight"][i] * val for key, val in ds_losses.items()}
                )
            # calculate overall loss
            loss = sum(losses.values())
            loss.backward()
            optimiser.step()
            for i, key in enumerate(self.params["loss_dictionary"].keys()):
                running_losses[key].append(losses[key].item())
                if model.classification_head and key == "lesion_classification":
                    # no ds for lesion classification in this case
                    continue
                if model.object_detection_head and key=='object_detection':
                    # no ds for object detection in this case
                    continue
                for level in self.deep_supervision['levels']:
                    running_losses[f'ds{level}_{key}'].append(losses[f'ds{level}_{key}'].item())
            running_losses['loss'].append(loss.item())

            # metrics
            pred = torch.argmax(estimates["log_softmax"], axis=1)
           # pred_obj=None
            if model.classification_head:
                pred_class = torch.argmax(estimates['hemi_log_softmax'], axis=1)
       #     elif model.object_detection_head:
       #         pred_obj = estimates['object_detection_linear']
            else:
                pred_class = torch.argmax(estimates["log_sumexp"], axis=1)
            # update running metrics
            borderzone = None
            if "sub_auroc" in metrics.metrics_to_track:
                borderzone = distance_map <= 20
            metrics.update(pred, labels, pred_class=pred_class,# pred_obj=pred_obj,
                           estimates=estimates['log_softmax'],
                           borderzone = borderzone)
            # TODO add distance regression to metrics

        scores = {key: np.mean(running_losses[key]) for key in running_losses.keys()}
        scores.update(metrics.get_aggregated_metrics())
        return scores

    def val_epoch(self, data_loader):
        """
        Validate after one epoch.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.experiment.model
        model.eval()
        with torch.no_grad():
            metrics = Metrics(
                self.params["metrics"],
                n_vertices=self.experiment.model.n_vertices,
                device=device,
            )  # for keeping track of running metrics
            running_losses = {key: [] for key in self.params["loss_dictionary"].keys()}
            for key in list(running_losses.keys()):
                for level in self.deep_supervision["levels"]:
                    running_losses[f"ds{level}_{key}"] = []
            running_losses["loss"] = []
            for i, data in enumerate(data_loader):
                data = data.to(device)
                estimates = model(data.x)
                labels = data.y.squeeze()
                distance_map = getattr(data, "distance_map", None)
                xyzr = getattr(data, "xyzr", None)
                losses = calculate_loss(self.params['loss_dictionary'], estimates, labels, 
                                        distance_map=distance_map, xyzr=xyzr,
                                        deep_supervision_level=None, device=device, 
                    n_vertices=self.experiment.model.n_vertices)
                # add deep supervision outputs
                for i, level in enumerate(sorted(self.deep_supervision["levels"])):
                    cur_labels = getattr(data, f"output_level{level}")
                    cur_distance_map = getattr(data, f"output_level{level}_distance_map", None)
                    n_vertices = len(self.experiment.model.icospheres.icospheres[level]["coords"])
                    ds_losses = calculate_loss(
                        self.params["loss_dictionary"],
                        estimates,
                        cur_labels,
                        distance_map=cur_distance_map,
                        deep_supervision_level=level,
                        device=device,
                        n_vertices=n_vertices,
                    )
                    losses.update(
                        {f"ds{level}_{key}": self.deep_supervision["weight"][i] * val for key, val in ds_losses.items()}
                    )
                # calculate overall loss
                loss = sum(losses.values())
                # keep track of loss
                for i, key in enumerate(self.params["loss_dictionary"].keys()):
                    running_losses[key].append(losses[key].item())
                    if model.classification_head and key=='lesion_classification':
                    # no ds for lesion classification in this case
                        continue
                    if model.object_detection_head and key=='object_detection':
                        # no ds for object detection in this case
                        continue
                    for level in self.deep_supervision["levels"]:
                        running_losses[f"ds{level}_{key}"].append(losses[f"ds{level}_{key}"].item())
                running_losses["loss"].append(loss.item())

                # metrics
                pred = torch.argmax(estimates["log_softmax"], axis=1)
                pred_obj=None
                if model.classification_head:
                    pred_class = torch.argmax(estimates['hemi_log_softmax'], axis=1)
            #    elif model.object_detection_head:
            #        pred_obj = estimates['object_detection_linear']
                else:
                    pred_class = torch.argmax(estimates["log_sumexp"], axis=1)
                # update running metrics
                # TODO add distance regression metrics here?
                borderzone = None
                if "sub_auroc" in metrics.metrics_to_track:
                    borderzone = distance_map <= 20
                metrics.update(pred, labels, pred_class=pred_class, # pred_obj=pred_obj,
                                estimates=estimates['log_softmax'],
                            borderzone = borderzone)
     
        scores = {key: np.mean(running_losses[key]) for key in running_losses.keys()}
        scores.update(metrics.get_aggregated_metrics())
        # set model back to training mode
        model.train()
        return scores

    def train(self, wandb_logging=False):
        """
        Train val loop with patience and best model saving.
        """
        # set up model & put on correct device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model if start epoch > 0
        if self.params.get("start_epoch", False):
            self.init_weights = os.path.join(self.experiment.experiment_path, f"epoch_{self.params['start_epoch']}.pt")
            self.experiment.load_model(checkpoint_path=self.init_weights)

        self.experiment.load_model(checkpoint_path=self.init_weights)
        self.experiment.model.to(device)
        if wandb_logging:
            import wandb

            wandb.init(entity="meld", project="classification")
            model = self.experiment.model
            # wandb.watch(model)

        # get dataset
        train_dset = GraphDataset.from_experiment(self.experiment, mode="train")
        sampler = None
        shuffle = self.params["shuffle_each_epoch"]
        if self.params["oversampling"]:
            sampler = Oversampler(train_dset)
            shuffle = False  # oversampler will do shuffling

        train_data_loader = torch_geometric.loader.DataLoader(
            train_dset,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=self.params["batch_size"],
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_data_loader = torch_geometric.loader.DataLoader(
            GraphDataset.from_experiment(self.experiment, mode="val"),
            shuffle=False,
            #THIS shouldn't be hard coded as 2, but currently is because otherwise sub_auroc
            #is a bit off
            batch_size= 2, #self.params["batch_size"],
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        # set up training loop
        # set up optimiser
        if self.params["optimiser"] == "adam":
            optimiser = torch.optim.Adam(
                self.experiment.model.parameters(),
                **self.params["optimiser_parameters"],
            )
        elif self.params["optimiser"] == "sgd":
            optimiser = torch.optim.SGD(
                self.experiment.model.parameters(),
                **self.params["optimiser_parameters"],
            )
        self.optimiser = optimiser
        # option to choose stopping metric for patience
        if self.params["stopping_metric"] is None:
            self.params["stopping_metric"] = {"name": "loss", "sign": 1}
        name = self.params["stopping_metric"]["name"]
        self.log.info(f"Stopping metric set to {name} ")

        #extra auc stopping metric
       # self.params["auc_stopping"] = {'name':'sub_auroc', "sign":-1}

        # set up learning rate scheduler
        max_epochs_lr_decay = self.params.get("max_epochs_lr_decay", None)
        if max_epochs_lr_decay is None:
            max_epochs_lr_decay = self.params["num_epochs"]
        if self.params["metric_smoothing"] is None:
            self.params["metric_smoothing"] = False
        self.log.info(f"using max_epochs {max_epochs_lr_decay} for lr decay")
        lambda1 = lambda epoch: (1 - epoch / max_epochs_lr_decay) ** self.params["lr_decay"]
        # NOTE: when resuming training, need to set last epoch to epoch-1
        if self.params.get("start_epoch", False):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda1,
             last_epoch=-1)
            for epoch in range(self.params["start_epoch"]):
                scheduler.step()
            scores = {"train": pd.read_csv(os.path.join(self.experiment.experiment_path, "train_scores.csv")),
                    "val": pd.read_csv(os.path.join(self.experiment.experiment_path, "val_scores.csv"))}
            best_loss = np.min(scores["val"][self.params["stopping_metric"]["name"] * self.params["stopping_metric"]["sign"]])
            #convert them back in to list for appending new epoch scores
            scores["train"] = scores["train"].to_dict('records')
            scores["val"] = scores["val"].to_dict('records')

            #discard scores after to start epoch
            scores["train"] = scores["train"][:self.params["start_epoch"]]
            scores["val"] = scores["val"][:self.params["start_epoch"]]
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lambda1,
             last_epoch=-1)
            scores = {"train": [], "val": []}
            best_loss = 100000
        #auc hack
      #  auc_loss = 10000
        patience = 0
        running_metrics = []
        for epoch in range(self.params["num_epochs"]):
            #introduce start epoch here
            if epoch > self.params.get("start_epoch", 0):
                
                self.log.info(f"Epoch {epoch} :: learning rate {scheduler.get_last_lr()[0]}")
                start = time.time()
                cur_scores = self.train_epoch(train_data_loader, optimiser)
                self.log.info(f"Epoch {epoch} :: time {time.time()-start}")
                scheduler.step()  # update lr
                # get memory usage
                process = psutil.Process(os.getpid())
                # TODO remove?
                with open("memory_usage_gpu_parrallel.txt", "a") as f:
                    f.write(
                        f"Epoch {epoch} :: memory usage {process.memory_info().rss / 1024 ** 2}MB-  time {time.time()-start} \n "
                    )
                self.log.info(f"Epoch {epoch} :: memory usage {process.memory_info().rss / 1024 ** 2}MB")  # in bytes

                # only log non-deep supervision losses
                log_keys = list(cur_scores.keys())
                for i, key in enumerate(self.params["loss_dictionary"].keys()):
                    for level in self.deep_supervision["levels"]:
                        log_keys.remove(f"ds{level}_{key}")
                log_str = ", ".join(f"{key} {val:.3f}" for key, val in cur_scores.items() if key in log_keys)
                self.log.info(f"Epoch {epoch} :: Train {log_str}")
                scores["train"].append(cur_scores)

                if wandb_logging:
                    parameters = model.state_dict()
                    wandb.log(parameters)
                    for name, param in parameters.items():
                        wandb.log({name + "_grad": param.grad}, step=epoch)
                    wandb.log({"train_losses": cur_scores})
                    # weights = self.experiment.model.get_weights()
                    # biases = self.experiment.model.get_biases()
                    # wandb.log({"weights": weights,'biases':biases})
                if epoch % 1 == 0:
                    cur_scores = self.val_epoch(val_data_loader)
                    log_str = ", ".join(f"{key} {val:.3f}" for key, val in cur_scores.items() if key in log_keys)
                    self.log.info(f"Epoch {epoch} :: Val   {log_str}")
                    scores["val"].append(cur_scores)
                    epoch_stopping_metric = (
                        cur_scores[self.params["stopping_metric"]["name"]] * self.params["stopping_metric"]["sign"]
                    )
                    running_metrics.append(epoch_stopping_metric)
                    

                    # TODO remove metric smoothing? If not, add to example_experiment_config!
                    # if self.params['metric_smoothing']:
                    # smooth metric to limit noise
                    #    epoch_stopping_metric = gaussian_filter1d(running_metrics,2)[-1]

                    if epoch_stopping_metric < best_loss:
                        best_loss = epoch_stopping_metric
                        if self.experiment.experiment_path is not None:
                            fname = os.path.join(self.experiment.experiment_path, "best_model.pt")
                            torch.save(self.experiment.model.state_dict(), fname)
                            self.log.info(f"Saved new best model to {fname}")
                        patience = 0
                    else:
                        patience += 1

                    #hacky auc
                #  extra_stopping_metric = ( cur_scores[self.params["auc_stopping"]["name"]] * self.params["auc_stopping"]["sign"])
                #  if extra_stopping_metric < auc_loss and epoch>99 :
                #      auc_loss = extra_stopping_metric
                #      if self.experiment.experiment_path is not None:
                #          fname = os.path.join(self.experiment.experiment_path, "best_auc_model.pt")
                #          torch.save(self.experiment.model.state_dict(), fname)
                #          self.log.info(f"Saved new best auc model to {fname}")

                    if patience >= self.params["max_patience"]:
                        self.log.info(f"Stopping early at epoch {epoch}, with patience {patience}")
                        break
                if epoch % 5 == 0:
                    # save train/val scores
                    if self.experiment.experiment_path is not None:
                        pd.DataFrame(scores["train"]).to_csv(
                            os.path.join(self.experiment.experiment_path, "train_scores.csv")
                        )
                        pd.DataFrame(scores["val"]).to_csv(os.path.join(self.experiment.experiment_path, "val_scores.csv"))
                if epoch % 200 ==0:
                    if self.experiment.experiment_path is not None:
                            fname = os.path.join(self.experiment.experiment_path, f"epoch_{epoch}.pt")
                            torch.save(self.experiment.model.state_dict(), fname)
                            self.log.info(f"Saved checkpoint model to {fname}")

        self.log.info(f"Finished training")
        self.log.info(f"Saving final model")
        if self.experiment.experiment_path is not None:
                        fname = os.path.join(self.experiment.experiment_path, "final_model.pt")
                        torch.save(self.experiment.model.state_dict(), fname)
                        self.log.info(f"Saved final model to {fname}")
        # save train/val scores
        if self.experiment.experiment_path is not None:
            pd.DataFrame(scores["train"]).to_csv(os.path.join(self.experiment.experiment_path, "train_scores.csv"))
            pd.DataFrame(scores["val"]).to_csv(os.path.join(self.experiment.experiment_path, "val_scores.csv"))
