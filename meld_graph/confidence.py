# scripts for creating per-vertex and per-cluster calibration plots
import matplotlib.pyplot as plt
import numpy as np
from meld_classifier.meld_cohort import MeldSubject


def get_lesion(subjects, cohort, results_dict=None):
    if results_dict is None:
        results_dict = {}
    for subj_id in subjects:
        if subj_id not in results_dict.keys():
            results_dict[subj_id] = {}
        subj = MeldSubject(subj_id, cohort=cohort)
        results_dict[subj_id]['lesion'] = np.concatenate([
            np.ceil(subj.load_feature_values(".on_lh.lesion.mgh", hemi='lh')).astype(int)[cohort.cortex_mask],
            np.ceil(subj.load_feature_values(".on_lh.lesion.mgh", hemi='rh')).astype(int)[cohort.cortex_mask]])
    return results_dict

def get_confidence(eva, subjects, cohort, threshold=0.5, confidence_suffix=""):
    """
    Return pixel-wise confidence for a prediction hdf5 file and a list of subjects.
    Returned dict will contain for each subject:
        prediction: the prediction mask
        clusters: prediction clusters
        confidence_lesion: the prediction score for lesion
        confidence_nonlesion: the prediction score for nonlesion
        
    Args:
        eva: Evaluator object
        subjects: list of subject ids to compute confidence for
        cohort: MELDCohort
        threshold: threshold of lesional predictions
    """
    results_dict = get_lesion(subjects, cohort)
    
    for subj_id in subjects:
        if subj_id not in results_dict.keys():
            results_dict[subj_id] = {}
            
        subj_results_dict = eva.load_data_from_file(subj_id, keys=['cluster_thresholded','result'])
        if confidence_suffix == "":
            subj_results_dict_confidence = subj_results_dict
        else:
            subj_results_dict_confidence = eva.load_data_from_file(subj_id, keys=['result',], 
                save_prediction_suffix=confidence_suffix)
        
        # get prediction & confidence from potentially different results files
        thresholded_prediction = (subj_results_dict['cluster_thresholded'] > 0).astype(int)
        confidence_lesion = subj_results_dict_confidence['result']
        
        results_dict[subj_id]['prediction'] = thresholded_prediction
        results_dict[subj_id]['confidence_lesion'] = confidence_lesion
        results_dict[subj_id]['confidence_nonlesion'] = 1 - confidence_lesion
        results_dict[subj_id]['clusters'] = subj_results_dict['cluster_thresholded']
        
    return results_dict


def calibration_plot(results_dict, n_bins=10, confidence='confidence_lesion'):
    """
    calculate ECE as described in literature
    calclulate calibration plot as calculated by sklearn.calibration
    """
    # sort all results in bins according to confidence score
    bins = np.linspace(0,1,n_bins+1)
    binned_prediction = {bin_idx: np.array([]) for bin_idx in range(len(bins[:-1]))}
    binned_confidence = {bin_idx: np.array([]) for bin_idx in range(len(bins[:-1]))}
    binned_lesion = {bin_idx: np.array([]) for bin_idx in range(len(bins[:-1]))}
    for subj_results in results_dict.values():
        for bin_idx in binned_prediction.keys():
            bin_min = bins[bin_idx]
            bin_max = bins[bin_idx+1]
            # get idx of all vertices in current bin (according to confidence)
            mask = (subj_results[confidence] > bin_min) & (subj_results[confidence] <= bin_max)
            #lesion_mask = subj_results['lesion']
            binned_prediction[bin_idx] = np.concatenate([binned_prediction[bin_idx], subj_results['prediction'][mask]])
            binned_lesion[bin_idx] = np.concatenate([binned_lesion[bin_idx], subj_results['lesion'][mask]])
            binned_confidence[bin_idx] = np.concatenate([binned_confidence[bin_idx], subj_results[confidence][mask]])    
    # calculate accuracy as frequency of correct labels in each bin
    freq = []
    acc = []
    conf = []
    n = []
    for bin_idx in binned_prediction.keys():
        if confidence == 'confidence_lesion':
            freq.append(binned_lesion[bin_idx].sum()/len(binned_lesion[bin_idx]))
        if confidence == 'confidence_nonlesion':
            freq.append(1 - binned_lesion[bin_idx].sum()/len(binned_lesion[bin_idx]))
        conf.append(binned_confidence[bin_idx].sum()/len(binned_lesion[bin_idx]))
        acc.append((binned_prediction[bin_idx] == binned_lesion[bin_idx]).sum() / len(binned_lesion[bin_idx]))
        n.append(len(binned_prediction[bin_idx]) / sum([len(binned_prediction[i]) for i in binned_prediction.keys()]))
    # NOTE also calculate ECE as freq - conf, instead of using acc
    ece = (np.abs(np.array(freq) - np.array(conf))*np.array(n)).sum()
    #print('ECE: ', ece)
        
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(bins[:-1] + (bins[1:]-bins[:-1])/2, freq, 'o-')
    ax.bar(bins[:-1] + (bins[1:]-bins[:-1])/2, n, width=0.05, color='black', alpha=0.5)
    ax.set_xlabel(confidence)
    ax.set_ylabel('frequency of label')
    ax.set_title('Per vertex confidence (ECE: {:.2f})'.format(ece))
    return fig

def calculate_per_cluster_confidence(results_dict, aggregation_fn='median'):
    """
    calculate mean confidence per cluster & whether it was detected
    Returns:
        per_cluster_confidence, per_cluster_label
    """
    per_cluster_confidence = []
    per_cluster_label = []
    for subj in results_dict.keys():
        #print(results_dict[subj]['clusters'].max())
        for i in range(int(results_dict[subj]['clusters'].max())+1):
            if i == 0:
                # don't do anything for background class
                continue
            # for foreground cluster, calculate mean confidence
            mask = results_dict[subj]['clusters'] == i
            if aggregation_fn == 'mean':
                conf  = np.mean(results_dict[subj]['confidence_lesion'][mask])
            elif aggregation_fn == 'median':
                conf  = np.median(results_dict[subj]['confidence_lesion'][mask])
            else:
                raise NotImplementedError(f'aggregation function {aggregation_fn}')
            per_cluster_confidence.append(conf)
            per_cluster_label.append(results_dict[subj]['lesion'][mask].max())
            
    return np.array(per_cluster_confidence), np.array(per_cluster_label)

def cluster_calibration_plot(confidence, label, n_bins=10):
    """
    calculate ECE as described in literature
    calclulate calibration plot as calculated by sklearn.calibration
    """
    # sort all results in bins according to confidence score
    bins = np.linspace(0,1,n_bins+1)
    binned_confidence = {bin_idx: np.array([]) for bin_idx in range(len(bins[:-1]))}
    binned_lesion = {bin_idx: np.array([]) for bin_idx in range(len(bins[:-1]))}
    for bin_idx in binned_lesion.keys():
        bin_min = bins[bin_idx]
        bin_max = bins[bin_idx+1]
        # get idx of all vertices in current bin (according to confidence)
        mask = (confidence > bin_min) & (confidence <= bin_max)
        binned_lesion[bin_idx] = np.concatenate([binned_lesion[bin_idx], label[mask]])
        binned_confidence[bin_idx] = np.concatenate([binned_confidence[bin_idx], confidence[mask]])    
    # calculate accuracy as frequency of correct labels in each bin
    freq = []
    acc = []
    conf = []
    n = []
    for bin_idx in binned_lesion.keys():
        freq.append(binned_lesion[bin_idx].sum()/len(binned_lesion[bin_idx]))
        conf.append(binned_confidence[bin_idx].sum()/len(binned_lesion[bin_idx]))
        #acc.append((binned_prediction[bin_idx] == binned_lesion[bin_idx]).sum() / len(binned_lesion[bin_idx]))
        n.append(len(binned_confidence[bin_idx]) / sum([len(binned_confidence[i]) for i in binned_confidence.keys()]))
    # NOTE also calculate ECE as freq - conf, instead of using acc
    ece = np.nansum(np.abs(np.array(freq) - np.array(conf))*np.array(n))
    #print('ECE: ', ece)
        
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(bins[:-1] + (bins[1:]-bins[:-1])/2, freq, 'o-')
    ax.bar(bins[:-1] + (bins[1:]-bins[:-1])/2, n, width=0.05, color='black', alpha=0.5)
    ax.set_xlabel('confidence')
    ax.set_ylabel('frequency of label')
    ax.set_title('Per cluster confidence (ECE: {:.2f})'.format(ece))
    return fig