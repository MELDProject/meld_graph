import logging
import os
import torch
import torch_geometric.data
from meld_graph.dataset import GraphDataset
import numpy as np
import h5py
import scipy
import json
import pandas as pd
from meld_graph.training import tp_fp_fn_tn, dice_coeff
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class Evaluator:
    """ """

    def __init__(
        self,
        experiment,
        mode="test",
        checkpoint_path=None,
        make_images=False,
        dataset=None,
        cohort=None,
        subject_ids=None,
        save_dir=None,
    ):

        # set class params
        self.log = logging.getLogger(__name__)
        self.experiment = experiment
        assert mode in (
            "test",
            "val",
            "train",
            "inference",
        ), "mode needs to be either test or val or train or inference"
        self.mode = mode
        self.make_images = make_images

        self.data_dictionary = None
        self._roc_dictionary=None

        # TODO: add clustering and thershold
        # self.threshold = self.experiment.network_parameters["optimal_threshold"]
        # if threshold was not optimized, use 0.5
        # if not isinstance(self.threshold, float):
        #     self.threshold = 0.5
        # self.min_area_threshold = self.experiment.data_parameters["min_area_threshold"]
        # self.log.info("Evalution {}, {}".format(self.mode, self.threshold))

        # Initialised directory to save results and plots
        if save_dir is None:
            self.save_dir = self.experiment.path
        else:
            self.save_dir = save_dir
        if not os.path.isdir(os.path.join(save_dir,'results')):
            os.makedirs(os.path.join(save_dir,'results'),exist_ok = True)
        
        # if checkpoint load model
        if checkpoint_path:
            self.experiment.load_model(
                checkpoint_path=os.path.join(checkpoint_path, "best_model.pt"),
                force=True,
            )
        
        # update dataset, cohort and subjects if provided or take from experiment     
        if cohort != None:
            self.cohort = cohort
        else:
            self.cohort = self.experiment.cohort

        if subject_ids!= None:
            self.subject_id = subject_ids
        else:
            # set subject_ids
            train_ids, val_ids, test_ids = self.experiment.get_train_val_test_ids()
            if mode == "train":
                self.subject_ids = train_ids
            elif mode == "val":
                self.subject_ids = val_ids
            elif mode == "test":
                self.subject_ids = test_ids

        if dataset != None: 
            self.dataset = dataset
            self.subject_ids = self.dataset.subject_ids
            self.cohort = self.dataset.cohort
        else:
            self.dataset = GraphDataset(self.subject_ids, self.cohort, self.experiment.data_parameters, mode=mode)
            
    
    def evaluate(self,):
        """
        Evaluate the model.
        Runs `self.get_metrics(); self.plot_prediction_space(); self.plot_subjects_prediction()`
        and saves images to results folder.
        """
        # need to load and predict data
        if self.data_dictionary is None:
            self.load_predict_data()
        # calculate stats 
        self.stat_subjects()
        # make images if asked for
        if self.make_images:
            self.plot_subjects_prediction()


    def load_predict_data(
        self,store_predictions=True,
        roc_curves_thresholds=np.linspace(0,1,21),
        save_prediction=True,
    ):
        """ """
        self.log.info("loading data and predicting model")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # predict on data
        #TODO: enable batch_size > 1
        data_loader = torch_geometric.loader.DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=1,
            
        )
        self.data_dictionary = {}
        
        for i, data in enumerate(data_loader):
            subject_index =  i//2
            hemi = ['lh','rh'][i%2]
            if hemi=='lh':
                prediction_array = []
                distance_map_array = []
                labels_array = []
                features_array = []
                geodesic_array = []
            subj_id = self.subject_ids[subject_index]
            data = data.to(device)
            estimates = self.experiment.model(data.x)
            labels = data.y.squeeze()
            geo_distance = data.distance_map
            prediction = torch.exp(estimates['log_softmax'])[:,1]
            #get distance map if exist in loss, otherwise return array of NaN
            if 'distance_regression' in self.experiment.network_parameters['training_parameters']['loss_dictionary'].keys():
                distance_map = estimates['non_lesion_logits'][:,0]
            else: 
                distance_map = torch.full((len(prediction),1), torch.nan)[:,0]
            prediction_array.append(prediction.detach().numpy()[self.cohort.cortex_mask])
            labels_array.append(labels.numpy()[self.cohort.cortex_mask])
            features_array.append(data.x.numpy()[self.cohort.cortex_mask])
            distance_map_array.append(distance_map.detach().numpy()[self.cohort.cortex_mask])
            geodesic_array.append(geo_distance.numpy()[self.cohort.cortex_mask])
            #only save after right hemi has been run.
            if hemi=='rh':
                subject_dictionary =  {
                    "input_labels": np.concatenate(labels_array),
                    "result": np.concatenate(prediction_array),
                    "distance_map": np.concatenate(distance_map_array),
                    'borderzone':np.concatenate(geodesic_array)<20
                }
                #save prediction
                if save_prediction:
                    self.save_prediction(subj_id, subject_dictionary["result"])
                    #save distance map
                    self.save_prediction(subj_id, subject_dictionary["distance_map"],
                            dataset_str='distance_map')
                #save features if mode is training 
                if self.mode != "train":
                    subject_dictionary["input_features"] = np.concatenate(features_array)
                if store_predictions:
                    self.data_dictionary[subj_id] = subject_dictionary
                if roc_curves_thresholds is not None:
                    self.thresholds = roc_curves_thresholds
                    self.roc_curves(subject_dictionary)
        if roc_curves_thresholds is not None:
            self.calculate_aucs()
            self.save_roc_scores()


    def calculate_aucs(self):
        import sklearn.metrics as metrics
        x = 1-self.roc_dictionary['specificity']/self.roc_dictionary['specificity'][-1]
        y1 = self.roc_dictionary['sensitivity']/self.roc_dictionary['sensitivity'][0]
        y2 = self.roc_dictionary['sensitivity_plus']/self.roc_dictionary['sensitivity_plus'][0]
        self.roc_dictionary['auc'] = metrics.auc(x,y1)
        self.roc_dictionary['auc_plus'] = metrics.auc(x,y2)
        self.roc_dictionary['thresholds'] = self.thresholds
        return

    def save_roc_scores(self):
        import pickle
        filename = os.path.join(self.save_dir, "results", f"roc_auc.pickle")
        with open(filename, "wb") as write_file:
            pickle.dump(self.roc_dictionary, write_file, protocol=pickle.HIGHEST_PROTOCOL) 
        return

    def roc_curves(self,subject_dictionary):
        """calculate performance at multiple thresholds"""
        
        for t_i,threshold in enumerate(self.thresholds):
            predicted = subject_dictionary['result']>= threshold
            # if we want tpr vs fpr curve too
           # tp,fp,fn, tn = tp_fp_fn_tn(predicted, subject_dictionary['input_labels'])
           #store sensitivity and sensitivity_plus for each patient (has a label)
            if subject_dictionary['input_labels'].sum()>0:
                self.roc_dictionary['sensitivity'][t_i] += np.logical_and(predicted, subject_dictionary['input_labels']).any()
                self.roc_dictionary['sensitivity_plus'][t_i] += np.logical_and(predicted, subject_dictionary['borderzone']).any()
                #could break if no longer detecting
            #store specificity for controls (no label)
            else:
                self.roc_dictionary['specificity'][t_i] += ~predicted.any()
                #could break if no longer predicting

                #initialise dictionary
    @property
    def roc_dictionary(self):
        if self._roc_dictionary is None:
            self._roc_dictionary = {'sensitivity':  np.zeros(len(self.thresholds)),
                                'sensitivity_plus': np.zeros(len(self.thresholds)),
                                     'specificity':np.zeros(len(self.thresholds))}
        return self._roc_dictionary

    def stat_subjects(self, suffix="", fold=None):
        """calculate stats for each subjects
        """
        
        #TODO: need to add boundaries and clusters
        # boundary_label = MeldSubject(subject, self.experiment.cohort).load_boundary_zone(max_distance=20)
        
        # calculate stats first
        threshold=0.5
        for subject in self.data_dictionary.keys():
            prediction = self.data_dictionary[subject]["result"]
            labels = self.data_dictionary[subject]["input_labels"]
            group = labels.sum()!= 0
            
            detected = np.logical_and(prediction>threshold, labels).any()
            difference = np.setdiff1d(np.unique(prediction), np.unique(prediction[labels]))
            difference = difference[difference > 0]
            n_clusters = len(difference)
        # # if not detected, does a cluster overlap boundary zone and if so, how big is the cluster?
        # if not detected and prediction[np.logical_and(boundary_label, ~labels)].sum() > 0:
        #     border_verts = prediction[np.logical_and(boundary_label, ~labels)]
        #     i, counts = np.unique(border_verts, return_counts=True)
        #     counts = counts[i > 0]
        #     i = i[i > 0]
        #     cluster_index = i[np.argmax(counts)]
        #     border_detected = np.sum(prediction == cluster_index)
        # else:
        #     border_detected = 0
            patient_dice_vars = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            mask = torch.as_tensor(np.array(prediction>threshold)).long()
            label = torch.as_tensor(np.array(labels.astype(bool))).long()
            dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2),label)
            patient_dice_vars["TP"], patient_dice_vars["FP"], patient_dice_vars["FN"],patient_dice_vars["TN"]=tp_fp_fn_tn(mask, label)
            patient_dice_vars["Dice non-lesion"], patient_dice_vars["Dice lesion"] = list(dices)
            
            sub_df = pd.DataFrame(
                np.array([subject, group, detected, patient_dice_vars["TP"].numpy(), patient_dice_vars["FP"].numpy(), patient_dice_vars["FN"].numpy(), patient_dice_vars["TN"].numpy(), patient_dice_vars["Dice lesion"].numpy(), patient_dice_vars["Dice non-lesion"].numpy()]).reshape(-1, 1).T,
                columns=["ID", "group", "detected", 'tp', 'fp', 'fn', 'tn', 'dice lesional', 'dice non-lesional'],
            )
            #save results
            filename = os.path.join(self.save_dir, "results", f"test_results{suffix}.csv")
            if fold is not None:
                filename = os.path.join(self.save_dir, "results", f"test_results_{fold}{suffix}.csv")

            if os.path.isfile(filename):
                done = False
                while not done:
                    try:
                        df = pd.read_csv(filename, index_col=False)
                        # df = df.append(sub_df, ignore_index=True)
                        df = pd.concat([df, sub_df], ignore_index=True, sort=False)
                        df.to_csv(filename, index=False)
                        done = True
                    except pd.errors.EmptyDataError:
                        done = False
            else:
                sub_df.to_csv(filename, index=False)

    
    def plot_subjects_prediction(self, rootfile=None, flat_map=True):
        """plot predicted subjects"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        plt.close("all")

        #create directory to save images
        if not os.path.isdir(os.path.join(self.save_dir,'results','images')):
            os.makedirs(os.path.join(self.save_dir,'results','images'),exist_ok = True)

        for subject in self.data_dictionary.keys():
            if rootfile is not None:
                filename = os.path.join(rootfile.format(subject))
            else:
                filename = os.path.join(
                    self.save_dir, "results", "images", "{}.jpg".format(subject)
                )
                os.makedirs(os.path.join(self.save_dir, "results", "images",), exist_ok=True)

            result = self.data_dictionary[subject]["result"]
            distance_map = self.data_dictionary[subject]["distance_map"]
            # thresholded = self.data_dictionary[subject]["cluster_thresholded"]
            label = self.data_dictionary[subject]["input_labels"]
            result = np.reshape(result, len(result))

            result_hemis = self.experiment.cohort.split_hemispheres(result)
            distance_map_hemis = self.experiment.cohort.split_hemispheres(distance_map)
            label_hemis = self.experiment.cohort.split_hemispheres(label)
            
            # initialise the icosphere or flat map
            if flat_map != True:
                from meld_graph.icospheres import IcoSpheres

                icos = IcoSpheres()
                ico_ini = icos.icospheres[7]
                coords = ico_ini["coords"]
                faces = ico_ini["faces"]
            else:
                import nibabel as nb
                from meld_classifier.paths import BASE_PATH

                flat = nb.load(
                    os.path.join(
                        BASE_PATH, "fsaverage_sym", "surf", "lh.full.patch.flat.gii"
                    )
                )
                coords, faces = flat.darrays[0].data, flat.darrays[1].data
            
           
             # round up to get the square grid size
            fig= plt.figure(figsize=(11,8), constrained_layout=True)
            gs1 = GridSpec(3, 2, width_ratios=[1, 1],  wspace=0.1, hspace=0.1)
            if not np.isnan(distance_map_hemis['left']).any():
                data_to_plot= [result_hemis['left'], result_hemis['right'], 
                            distance_map_hemis['left'], distance_map_hemis['right'],
                            label_hemis['left'], label_hemis['right']]
                titles=['predictions left hemi', 'predictions right hemi', 
                    'distance map left hemi', 'distance map right hemi', 
                    'labels left hemi', 'labels right hemi']
            else:
                data_to_plot= [result_hemis['left'], result_hemis['right'], 
                            label_hemis['left'], label_hemis['right']]
                titles=['predictions left hemi', 'predictions right hemi', 
                    'labels left hemi', 'labels right hemi']
            for i,overlay in enumerate(data_to_plot):
                ax = fig.add_subplot(gs1[i])
                if 'distance' in titles[i]:
                    im = create_surface_plots(coords,faces,overlay,flat_map=True)
                else:
                    im = create_surface_plots(coords,faces,overlay,flat_map=True, limits=[0.4,0.6])
                ax.imshow(im)
                ax.axis('off')
                ax.set_title(titles[i], loc='left', fontsize=20)  
            fig.savefig(filename, bbox_inches='tight')
            plt.close("all")
    
    def save_prediction(self, subject, prediction, dataset_str="prediction", dtype=None):
        """
        saves prediction to {experiment_path}/results/predictions_{experiment_name}.hdf5.
        the hdf5 has the structure (subject_id/hemisphere/prediction).
        and contains predictions for all vertices inside the cortex mask
        dataset_str: name of the dataset to save prediction. If is 'prediction', also saves threshold
        dtype: dtype of the dataset. If none, use dtype of prediction.
        suffix: suffix for the filename for the prediction: "predictions_<self.experiment.name><suffix>.hdf5" is used
        """
        # make sure that give prediction has expected length
        nvert_hemi = len(self.experiment.cohort.cortex_label)
        assert len(prediction) == nvert_hemi * 2
        # get dtype
        if dtype is None:
            dtype = prediction.dtype

        filename = os.path.join(self.save_dir, "results", f"predictions.hdf5")
        if not os.path.isfile(filename):
            mode = "a"
        else:
            mode = "r+"
        done = False
        while not done:
            try:
                with h5py.File(filename, mode=mode) as f:
                    self.log.info(f"saving {dataset_str} for {subject}")
                    for i, hemi in enumerate(["lh", "rh"]):
                        shape = tuple([nvert_hemi] + list(prediction.shape[1:]))
                        # create dataset
                        dset = f.require_dataset(f"{subject}/{hemi}/{dataset_str}", shape=shape, dtype=dtype)
                        # save prediction in dataset
                        dset[:] = prediction[i * nvert_hemi : (i + 1) * nvert_hemi]
                        # if dataset_str == "prediction":
                            # save threshold as attribute in dataset
                            # dset.attrs["threshold"] = self.threshold
                    done = True
            except OSError:
                done = False


def save_json(json_filename, json_results):
    """
    Save dictionaries to json
    """
    # data_parameters
    json.dump(json_results, open(json_filename, "w"), indent=4)
    return


def create_surface_plots(coords,faces,overlay,flat_map=True, limits=None):
    """plot and reload surface images"""
    from meld_classifier.meld_plotting import trim
    import matplotlib_surface_plotting.matplotlib_surface_plotting as msp
    from PIL import Image
    
    if limits==None:
        vmin=np.min(overlay)
        vmax=np.max(overlay)
    else:
        vmin=limits[0]
        vmax=limits[1]
    msp.plot_surf(coords,faces, 
                overlay,
                flat_map=flat_map,
                rotate=[90, 270],
                filename='tmp.png',
                vmin=vmin,
                vmax=vmax,
             )
    im = Image.open('tmp.png')
    im = trim(im)
    im = im.convert("RGBA")
    im1 = np.array(im)
    return im1



def sens_spec_curves(roc_dict):
    """normalise sensitivity and specificity curves to 0-1"""
    sensitivity_curve = roc_dict['sensitivity_plus']/max(roc_dict['sensitivity_plus'])
    specificity_curve = roc_dict['specificity']/max(roc_dict['specificity'])
    return sensitivity_curve,specificity_curve

def plot_roc_multiple(roc_dictionary,roc_curves_thresholds):
    fig, ax = plt.subplots(1,1)   
    for mi,model in enumerate(roc_dictionary.keys()):
        sensitivity_curve,specificity_curve = sens_spec_curves(roc_dictionary[model])
        ax.plot(1-specificity_curve,sensitivity_curve,label=model)
        auc = metrics.auc(1-specificity_curve,sensitivity_curve)
        ax.text(0,1-mi/10,f'{model} AUC: {auc:.2f}')
    fig.legend()
    return fig


def load_prediction(subject,hdf5,dset='prediction'):
    """load network predictions"""
    results={}
    with h5py.File(hdf5, "r") as f:
        for hemi in ['lh','rh']:
            results[hemi] = f[subject][hemi][dset][:]
    return results

def roc_curves(subject_dictionary,roc_dictionary,roc_curves_thresholds):
    """calculate performance at multiple thresholds"""
    for t_i,threshold in enumerate(roc_curves_thresholds):
        predicted = subject_dictionary['result']>= threshold
        # if we want tpr vs fpr curve too
        #store sensitivity and sensitivity_plus for each patient (has a label)
        if subject_dictionary['input_labels'].sum()>0:
            roc_dictionary['sensitivity'][t_i] += np.logical_and(predicted, subject_dictionary['input_labels']).any()
            roc_dictionary['sensitivity_plus'][t_i] += np.logical_and(predicted, subject_dictionary['borderzone']).any()
        #store specificity for controls (no label)
        else:
            roc_dictionary['specificity'][t_i] += ~predicted.any()
