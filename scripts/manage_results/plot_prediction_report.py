from meld_graph.evaluation import Evaluator
from meld_graph.experiment import Experiment
from meld_graph.meld_cohort import MeldCohort, MeldSubject
from meld_graph.meld_plotting import trim
from meld_graph.confidence import get_confidence
from meld_graph.paths import (
    NVERT,
    MELD_PARAMS_PATH,
    MELD_DATA_PATH,
    DK_ATLAS_FILE,
    EXPERIMENT_PATH, 
    MODEL_PATH,
    MODEL_NAME,
    SURFACE_PARTIAL, 
    DEFAULT_HDF5_FILE_ROOT,
    SCRIPTS_DIR,
)
import os
import json
import glob
import h5py
import argparse
import numpy as np
import nibabel as nb
from nilearn import plotting, image
import pandas as pd
import matplotlib_surface_plotting as msp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
import meld_graph.mesh_tools as mt
from datetime import date
from fpdf import FPDF
from meld_graph.tools_commands_prints import get_m

class PDF(FPDF):    
    def lines(self):
        self.set_line_width(0.0)
        self.line(5.0,5.0,205.0,5.0) # top one
        self.line(5.0,292.0,205.0,292.0) # bottom one
        self.line(5.0,5.0,5.0,292.0) # left one
        self.line(205.0,5.0,205.0,292.0) # right one
    
    def custom_header(self, logo, txt1, txt2=None):
        # Log
        self.image(logo, 10, 8, 33)
        # Arial bold 
        self.set_font('Arial', 'B', 30)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(w=30, h=10, txt=txt1, border=0, ln=0, align='C')
        if txt2 != None:
            # Arial bold 15
            self.ln(20)
            self.cell(80)
            self.set_font('Arial', 'B', 20)
            self.cell(w=30, h=5, txt=txt2, border=0, ln=0, align='C')
        # Line break
        self.ln(20)
    
    def custom_footer(self, txt):
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Position at 1.5 cm from bottom
        self.set_y(-30)
        # add text
        self.cell(w=0, h=0, txt=txt, border=0, ln=2, align='C') 
        # Date
        today = date.today()
        today = today.strftime("%d/%m/%Y")
        self.cell(w=5, h=8, txt=str(today) , border=0, ln=0, align='L')
        # Page number
        self.cell(w=180, h=8, txt='Page ' + str(self.page_no()), border=0, ln=2, align='R')
                
    
    def info_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 10)
        #set box color
        self.set_fill_color(160,214,190)
        # add text box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)
        
    def info_box_clust(self, txt):
        self.ln(40)
        # set font
        self.set_font('Arial', 'I', 10)
        #set box color
        self.set_fill_color(160,214,190)
        # add text box info
        self.multi_cell(w=160, h=5, txt=txt , border=1, align='L', fill=True)
    
    def disclaimer_box(self, txt):
        # set font
        self.set_font('Arial', 'I', 6)
        #set box color
        self.set_fill_color(240,128,128)
        # add texte box info
        self.multi_cell(w=190, h=5, txt=txt , border=1, align='L', fill=True)
        
    def subtitle_inflat(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Title
        self.cell(w=10, h=80, txt='Overview clusters on inflated brain', border=0, ln=2, align='L')
       
    def imagey(self,im, y):
        self.image(im, 5, y, link='', type='', w=190, h=297/3)

        
def load_prediction(subject,hdf5):
    results={}
    with h5py.File(hdf5, "r") as f:
        for hemi in ['lh','rh']:
            results[hemi] = f[subject][hemi]['prediction'][:]
    return results

def create_surface_plots(surf,prediction,c, base_size=20):
    """plot and reload surface images"""
    cmap, colors =  load_cmap()
    msp.plot_surf(surf['coords'],
              surf['faces'],prediction,
              rotate=[90],
              mask=prediction==0,pvals=np.ones_like(c.cortex_mask),
              colorbar=False,vmin=1,vmax=len(colors) ,cmap=cmap,
              base_size=base_size,
              filename='tmp.png'
             );
#    subprocess.call(f"convert ./tmp.png -trim ./tmp1.png", shell=True)
    im = Image.open('tmp.png')
    im = trim(im)
    im = im.convert("RGBA")
    im1 = np.array(im)
    msp.plot_surf(surf['coords'],
            surf['faces'],prediction,
              rotate=[270],
              mask=prediction==0,pvals=np.ones_like(c.cortex_mask),
              colorbar=False,vmin=1,vmax=len(colors),cmap=cmap,
              base_size=base_size,
              filename='tmp.png'
             );
 #   subprocess.call(f"convert ./tmp.png -trim ./tmp1.png", shell=True)
    im = Image.open('tmp.png')
    im = trim(im)
    im = im.convert("RGBA")
    im2 = np.array(im)
    plt.close('all')
    return im1,im2

def load_cluster(file, subject):
    df=pd.read_csv(file,index_col=False)
    n_clusters = df[df['ID']==subject]['n_clusters']
    return np.array(n_clusters)[0]

def get_key(dic, val):
    # function to return key for any value in dictionnary
    for key, value in dic.items():
        if val == value:
            return key
    return "No key for value {}".format(val)

def define_atlas():
    atlas = nb.freesurfer.io.read_annot(os.path.join(MELD_PARAMS_PATH, DK_ATLAS_FILE))
    vertex_i = np.array(atlas[0]) - 1000  # subtract 1000 to line up vertex
    rois_prop = [
        np.count_nonzero(vertex_i == x) for x in set(vertex_i)
    ]  # proportion of vertex per rois
    rois = [x.decode("utf8") for x in atlas[2]]  # extract rois label from the atlas
    rois = dict(zip(rois, range(len(rois))))  # extract rois label from the atlas
    rois.pop("unknown")  # roi not part of the cortex
    rois.pop("corpuscallosum")  # roi not part of the cortex
    return rois, vertex_i, rois_prop

def get_cluster_location(cluster_array):
    cluster_array = np.array(cluster_array)
    rois, vertex_i, rois_prop = define_atlas()
    pred_rois = list(vertex_i[cluster_array])
    pred_rois = np.array([[x, pred_rois.count(x)] for x in set(pred_rois) if x != 0])
    ind = pred_rois[np.where(pred_rois == pred_rois[:,1].max())[0]][0][0]
    location = get_key(rois,ind)
    return location

def save_mgh(filename, array, demo):
    """save mgh file using nibabel and imported demo mgh file"""
    mmap = np.memmap("/tmp/tmp", dtype="float32", mode="w+", shape=demo.get_data().shape)
    mmap[:, 0, 0] = array[:]
    output = nb.MGHImage(mmap, demo.affine, demo.header)
    nb.save(output, filename)

def load_cmap():
    """ create the colors dictionarry for the clusters"""
    from matplotlib.colors import ListedColormap
    import numpy as np
    colors =  [
        [255,0,0],     #red
        [255,215,0],   #gold
        [0,0,255],     #blue
        [0,128,0],     #green
        [148,0,211],   #darkviolet
        [255,0,255],   #fuchsia
        [255,165,0],   #orange
        [0,255,255],   #cyan
        [0,255,0],     #lime
        [106,90,205],  #slateblue
        [240,128,128], #lightcoral
        [184,134,11],  #darkgoldenrod
        [100,149,237], #cornflowerblue
        [102,205,170], #mediumaquamarine
        [75,0,130],    #indigo
        [250,128,114], #salmon
        [240,230,140], #khaki
        [176,224,230], #powderblue
        [128,128,0],   #olive
        [221,160,221], #plum
        [255,127,80],  #coral
        [255,250,205], #lemonchiffon
        [240,255,255], #azure
        [152,251,152], #palegreen
        [255,192,203], #pink
    ]
    colors=np.array(colors)/255
    dict_c = dict(zip(np.arange(1, len(colors)+1), colors))
    cmap = ListedColormap(colors)
    return cmap, dict_c

def get_subj_data(subject_id, eva):
    
    #load data for that subject
    data_dictionary = eva.load_data_from_file(subject_id, keys=['result','cluster_thresholded','input_features'], 
                                          split_hemis=True, )
    features_vals = data_dictionary['input_features']
    predictions = data_dictionary['cluster_thresholded']

    #find thresholds used if two thresholds
    if isinstance(eva.threshold, np.ndarray):
        if max(data_dictionary['result']['left'].max(), data_dictionary['result']['right'].max()) > eva.threshold[1]:
            threshold_text = "high confidence cluster"
        else :
            threshold_text = "No high confidence cluster\nLow confidence cluster given instead"
    else:
        threshold_text = ""
        
    #find clusters and load saliencies and confidence
    list_clust = {}
    confidences = {}
    saliencies = {}
    for hemi in ['left','right']:
        list_clust[hemi] = set(predictions[hemi])
        list_clust[hemi].remove(0.0)
        keys = [f'saliencies_{cl}' for cl in list_clust[hemi]] + [f'mask_salient_{cl}' for cl in list_clust[hemi]]
        saliencies.update(eva.load_data_from_file(subject_id, 
                                            keys=keys, 
                                            split_hemis=True))
    
        for cl in list_clust[hemi]:
            mask_salient = saliencies[f'mask_salient_{cl}'][hemi].astype(bool)
            confidence_cl_salient = data_dictionary['result'][hemi][mask_salient].mean()
            confidences[f'confidence_{cl}'] =  confidence_cl_salient

    return list_clust, features_vals, predictions, threshold_text, saliencies, confidences

def get_info_soft( subject_id):
    ''' Report information of softwares (e.g Freesurfer) '''
    from meld_graph import __version__
    from meld_graph.paths import BASE_PATH

    if __version__ != None:
        meld_version = __version__
    else:
        meld_version = "Unknown"


    fs_scripts = os.path.join(MELD_DATA_PATH, 'output','fs_outputs', subject_id, 'scripts')
    if os.path.isfile(os.path.join(fs_scripts,'build-stamp.txt')):
        with open(os.path.join(fs_scripts,'build-stamp.txt')) as f:
            FS_version = f.readlines()[0].strip()
        if os.path.isfile(os.path.join(fs_scripts,'deep-seg.log')):
            Fastsurfer_use =  'True'
        else:
            Fastsurfer_use =  'False'
    else:
        FS_version = 'Unknown'
    
    text = "\n".join((
                "Information about softwares:",
                f"MELD package version: {meld_version}",
                f"Freesurfer version: {FS_version}",
                f"Use of FastSurfer: {Fastsurfer_use}",
                ))
    return text

def get_t1_file(subject_id, subject_dir):
    ''' 
    return path of T1 if BIDs format or MELD format
    TODO : improve flexibility of BIDS
    '''
    t1_files_MELD = glob.glob(os.path.join(subject_dir, "T1", "*.nii*"))
    t1_files_bids = glob.glob(os.path.join(subject_dir, "anat", "*T1*.nii*"))
    if len(t1_files_MELD)==1:
        t1_path= t1_files_MELD[0]
        print(get_m(f'T1 file used : {t1_path} ', subject_id, 'INFO'))
    elif len(t1_files_MELD)>1:
        print(get_m(f'Find too much volumes for T1. Check and remove the additional volumes with same key name', subject_id, 'WARNING'))
        return None
    elif len(t1_files_bids)==1:
        t1_path = t1_files_bids[0]
        print(get_m(f'T1 file used : {t1_path} ', subject_id, 'INFO'))
    elif len(t1_files_bids)>1:
        print(get_m(f'Find too much volumes for T1. Check and remove the additional volumes with same key name', subject_id, 'WARNING'))
        return None
    else:
        print(get_m(f'Could not find any T1w nifti file. Please ensure your data are in MELD or BIDS format', subject_id, 'ERROR'))
        return None
    return t1_path
      
def generate_prediction_report(
    subject_ids, data_dir, prediction_path, output_dir, 
    experiment_path=EXPERIMENT_PATH, hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=None):
    ''' Create images and report of predictions on inflated brain, on native T1 accompanied with saliencies explaining the predictions
    inputs: 
        subject_ids: subjects ID
        data_dir: data directory containing the T1. Should be "input" in MELD structure
        hdf_predictions: hdf5 containing the MELD predictions
        exp: an experiment initialised
        output_dir: directory to save final reports
        '''
    # setup parameters
    base_feature_sets = [
        ".on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".on_lh.wm_FLAIR_1.sm3.mgh",
        ".on_lh.curv.sm3.mgh",
        ".on_lh.pial.K_filtered.sm20.mgh",
        ".on_lh.sulc.sm3.mgh",
        ".on_lh.thickness_regression.sm3.mgh",
        ".on_lh.w-g.pct.sm3.mgh",
    ]

    feature_names_sets = [
        "GM FLAIR (50%)",
        "WM FLAIR (1mm)",
        "Mean curvature",
        "Intrinsic Curvature",
        "Sulcal depth",
        "Cortical thickness",
        "Grey-white contrast",
    ]

    c = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=dataset)
    surf = mt.load_mesh_geometry(os.path.join(MELD_PARAMS_PATH, SURFACE_PARTIAL))

    # load cmap and colors
    cmap, colors = load_cmap()

    # create evaluator
    exp = Experiment.from_folder(experiment_path)
    features = exp.data_parameters['features']
    eva = Evaluator(
        experiment=exp,
        save_dir = prediction_path,
        make_images=False,
        cohort=c,
        subject_ids=subject_ids,
        mode="test",
        thresh_and_clust=True,
    )

    for subject_id in subject_ids:
        # find subject directory containing T1
        subject = MeldSubject(subject_id, cohort=c)
        # create output directory
        output_dir_sub = os.path.join(output_dir, subject_id, "reports")
        os.makedirs(os.path.join(output_dir_sub), exist_ok=True)
        # Open their MRI data if available
        t1_file = get_t1_file(subject_id, os.path.join(data_dir, subject_id))
        prediction_file_lh = glob.glob(os.path.join(output_dir, subject_id, "predictions", "lh.prediction*"))[0]
        prediction_file_rh = glob.glob(os.path.join(output_dir, subject_id, "predictions", "rh.prediction*"))[0]
        # load image
        imgs = {
            "anat": nb.load(t1_file),
            "pred_left": nb.load(prediction_file_lh),
            "pred_right": nb.load(prediction_file_rh),
        }
        # Resample and move to same shape and affine than t1
        imgs["pred_left"] = image.resample_img(
            imgs["pred_left"],
            target_affine=imgs["anat"].affine,
            target_shape=imgs["anat"].shape,
            interpolation="nearest",
            copy=True,
            order="F",
            clip=False,
            fill_value=0,
            force_resample=False,
        )
        imgs["pred_right"] = image.resample_img(
            imgs["pred_right"],
            target_affine=imgs["anat"].affine,
            target_shape=imgs["anat"].shape,
            interpolation="nearest",
            copy=True,
            order="F",
            clip=False,
            fill_value=0,
            force_resample=False,
        )
        # initialise parameter for plot
        fig = plt.figure(figsize=(15, 8), constrained_layout=True)
        if subject.has_flair:
            base_features = base_feature_sets
            feature_names = feature_names_sets
        else:
            base_features = base_feature_sets[2:]
            feature_names = feature_names_sets[2:]
        # load predictions and data subject
        list_clust, features_vals, predictions, threshold_text, saliencies, confidences  = get_subj_data(subject_id, eva)
        #save info cluster
        df=pd.DataFrame()
        info_cl={}
        # Loop over hemi
        for i, hemi in enumerate(["left", "right"]):
            # prepare grid plot
            gs1 = GridSpec(2, 3, width_ratios=[1, 1, 1], wspace=0.1, hspace=0.1)
            gs2 = GridSpec(2, 4, height_ratios=[1, 3], width_ratios=[1, 1, 0.5, 2], wspace=0.1)
            gs3 = GridSpec(1, 1)
            # plot predictions on inflated brain
            im1, im2 = create_surface_plots(surf, prediction=predictions[hemi], c=c)
            if hemi == "right":
                im1 = im1[:, ::-1]
                im2 = im2[:, ::-1]
            ax = fig.add_subplot(gs1[i, 1])
            ax.imshow(im1)
            ax.axis("off")
            ax.set_title(hemi, loc="left", fontsize=20)
            ax = fig.add_subplot(gs1[i, 2])
            ax.imshow(im2)
            ax.axis("off")
           # initiate params for saliencies
            prefixes = [".combat", ".inter_z.intra_z.combat", ".inter_z.asym.intra_z.combat"]
            lims = 50   
            norm = mpl.colors.Normalize(vmin=-lims, vmax=lims)
            norm = mpl.colors.Normalize(vmin=-lims, vmax=lims)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "grpr",
                colors=[
                    "#276419",
                    "#FFFFFF",
                    "#8E0152",
                ],
            )
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            labels = ["Combat", "Normalised", "Asymmetry"]
            hatching = ["\\\\", "//", "--"]
            # loop over clusters
            for cluster in list_clust[hemi]:
                fig2 = plt.figure(figsize=(17, 9))
                info_cl['cluster']=cluster
                # get and plot saliencies
                saliencies_cl = saliencies[f'saliencies_{cluster}'][hemi]
                saliencies_cl = saliencies_cl * (NVERT/2)
                 # plot prediction and salient vertices
                mask = np.array([predictions[hemi] == cluster])[0]
                mask_salient = saliencies[f'mask_salient_{cluster}'][hemi].astype(bool)
                mask_comb = mask.astype(int)+mask_salient.astype(int)                
                im1, im2 = create_surface_plots(surf, prediction=mask_comb, c=c, base_size=10)
                if hemi == "right":
                    im1 = im1[:, ::-1]
                    im2 = im2[:, ::-1]
                ax2 = fig2.add_subplot(gs2[1, 0])
                ax2.imshow(im1)
                ax2.axis("off")                
                ax2 = fig2.add_subplot(gs2[1, 1])
                ax2.imshow(im2)
                ax2.axis("off")
                # ax2.set_title('Predicted cluster and\n20% most salient vertices', loc="left", fontsize=15)
                ax2 = fig2.add_subplot(gs2[:, 3],)
                for pr, prefix in enumerate(prefixes):
                    cur_data = np.zeros(len(base_features))
                    cur_err = np.zeros(len(base_features))
                    saliency_data = np.zeros(len(base_features))
                    for b, bf in enumerate(base_features):
                        cur_data[b] = np.mean(
                            features_vals[hemi][mask_salient, features.index(prefix + bf)]
                        )
                        cur_err[b] = np.std(
                            features_vals[hemi][mask_salient, features.index(prefix + bf)]
                        )
                        saliency_data[b] = np.mean(
                            saliencies_cl[mask_salient ,features.index(prefix + bf)]
                            )
                        #add fingerprints
                        info_cl[labels[pr] + ' ' + feature_names[b] +' mean'] = cur_data[b]
                        info_cl[labels[pr] + ' ' + feature_names[b] +' std'] = cur_err[b]
                        info_cl[labels[pr] + ' ' + feature_names[b] +' saliency'] = saliency_data[b]
                    ax2.barh(
                        y=np.array(range(len(base_features))) - pr * 0.3,
                        width=cur_data,
                        hatch=hatching[pr],
                        height=0.3,
                        edgecolor="k",
                        xerr=cur_err,
                        label=labels[pr],
                        color=m.to_rgba(saliency_data),
                    )
                ax2.set_xlim([-8, 8])
                ax2.set_xticks([])
                ax2.set_yticks(np.array(range(len(base_features))) - 0.23)
                ax2.set_yticklabels(feature_names, fontsize=16)
                ax2.set_xlabel("Z score", fontsize=16)
                ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=16)
                fig2.colorbar(m, label=f"Saliency", ax=ax2, ticks=[-50, -25, 0, 25, 50])
                ax2.set_autoscale_on(True)
                ## display info cluster
                # get size
                size_clust = np.sum(c.surf_area[predictions[hemi] == cluster]) / 100
                size_clust = round(size_clust, 3)
                info_cl['size'] = size_clust
                # get location
                location = get_cluster_location(predictions[hemi] == cluster)
                info_cl['location'] = location
                # get confidence
                confidence = round(confidences[f'confidence_{cluster}'].mean(),2)
                info_cl['confidence'] = confidence
                info_cl['high_low_threshold']=threshold_text
                # plot info in text box in upper left in axes coords
                textstr = "\n".join(
                    (
                        f" cluster {int(cluster)} on {hemi} hemi",
                        " ",
                        f" size cluster = {size_clust} cm2",
                        " ",
                        f" location =  {location}",
                        " ",
                        f"{threshold_text}",
                        " ",
                        f" confidence =  {confidence}",
                    )
                )
                props = dict(boxstyle="round", facecolor=colors[int(cluster)], alpha=0.5)
                ax2 = fig2.add_subplot(gs2[0, 0:2])
                ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=18, verticalalignment="top", bbox=props)
                ax2.axis("off")
                #                 fig2.tight_layout()
                fig2.savefig(f"{output_dir_sub}/saliency_{subject.subject_id}_{hemi}_c{int(cluster)}.png")
                # plot cluster on anat MRI volume
                fig3 = plt.figure(figsize=(15, 8))
                ax3 = fig3.add_subplot(gs3[0])
                min_v = cluster - 1
                max_v = cluster + 1
                mask = image.math_img(f"(img < {max_v}) & (img > {min_v})", img=imgs[f"pred_{hemi}"])
                coords = plotting.find_xyz_cut_coords(mask)
                vmax = np.percentile(imgs["anat"].get_fdata(), 99)
                display = plotting.plot_anat(
                    t1_file, colorbar=False, cut_coords=coords, draw_cross=True, figure=fig3, axes=ax3, vmax=vmax
                )
                display.add_contours(prediction_file_lh, filled=True, alpha=0.7, levels=[0.5], colors="darkred")
                display.add_contours(prediction_file_rh, filled=True, alpha=0.7, levels=[0.5], colors="darkred")
                # save figure for each cluster
                #                 fig3.tight_layout()
                fig3.savefig(f"{output_dir_sub}/mri_{subject.subject_id}_{hemi}_c{int(cluster)}.png")
                # Add info to df
                df = pd.concat([df,pd.DataFrame([info_cl])])
        # Add information subject in text box
        n_clusters = len(list_clust["left"]) + len(list_clust["right"])
        ax = fig.add_subplot(gs1[0, 0])
        textstr = "\n".join((f"patient {subject.subject_id}", " ", f"number of predicted clusters = {n_clusters}"))
        # place a text box in upper left in axes coords
        props = dict(boxstyle="round", facecolor="gray", alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment="top", bbox=props)
        ax.axis("off")
        # save overview figure
        fig.savefig(f"{output_dir_sub}/inflatbrain_{subject.subject_id}.png")
        # save info subject
        df.to_csv(f"{output_dir_sub}/info_clusters_{subject.subject_id}.csv")
        # create PDF overview
        pdf = PDF()  # pdf object
        pdf = PDF(orientation="P")  # landscape
        pdf = PDF(unit="mm")  # unit of measurement
        pdf = PDF(format="A4")  # page format. A4 is the default value of the format, you don't have to specify it.

        logo = os.path.join(SCRIPTS_DIR, "MELD_logo.png")

        text_info_1 = "Information: \n The MRI data of this patient has been processed through the MELD surface-based FCD detection algorithm. \n Page 1 of this report will show all detected clusters on an inflated view of the brain. \n Each subsequent page is an individual cluster. \n Last page summarises the softwares version used to creates this report"

        text_info_2 = "The next pages present information for each cluster: \n   -The hemisphere the cluster is on \n   -The surface area of the cluster (across the cortical surface) \n   -The location of the cluster \n   -The z-scores of the patient cortical features averaged within the cluster. \n   -The saliency of each feature to the network - if a feature is brighter pink, that feature was more important to the network. \n \n For more information, please read the Guide to using the MELD surface-based FCD detection."

        disclaimer = "Disclaimer: The MELD surface-based FCD detection algorithm is intended for research purposes only and has not been reviewed or approved by the Medicines and Healthcare products Regulatory Agency (MHRA),European Medicine Agency (EMA) or by any other agency. Any clinical application of the software is at the sole risk of the party engaged in such application. \nThere is no warranty of any kind that the software will produce useful results in any way. Use of the software is at the recipient own risk"

        footer_txt = "This report was created by Mathilde Ripart, Hannah Spitzer, Sophie Adler and Konrad Wagstyl on behalf of the MELD Project"

        text_info_3 = get_info_soft(subject.subject_id)
        
        #### create main page with overview on inflated brain
        # add page
        pdf.add_page()
        # add line contours
        pdf.lines()
        # add header
        pdf.custom_header(logo, txt1="MELD report", txt2=f"Patient ID: {subject.subject_id}")
        # add info box
        pdf.info_box(text_info_1)
        # add disclaimer box
        pdf.disclaimer_box(disclaimer)
        # add image
        pdf.subtitle_inflat()
        im_inflat = os.path.join(output_dir_sub, f"inflatbrain_{subject.subject_id}.png")
        pdf.image(im_inflat, 5, 110, link='', type='', w=190, h=297/3)
        # add info cluster analysis 
        pdf.info_box_clust(text_info_2)
        # add footer date
        pdf.custom_footer(footer_txt)

        clusters = range(1, n_clusters + 1)
        #### Create page for each cluster with MRI view and saliencies
        for cluster in clusters:
            # add page
            pdf.add_page()
            # add line contours
            pdf.lines()
            # add header
            pdf.custom_header(logo, txt1="MRI view & saliencies", txt2=f"Cluster {cluster}")
            # add image
            im_mri = glob.glob(os.path.join(output_dir_sub, f"mri_{subject.subject_id}_*_c{cluster}.png"))[0]
            # pdf.imagey(im_mri, 50)
              #add segmentation figure left
            pdf.image(im_mri, 5, 50, link='', type='', w=190, h=297/3)
            # add image
            im_sal = glob.glob(os.path.join(output_dir_sub, f"saliency_{subject.subject_id}_*_c{cluster}.png"))[0]
            # pdf.imagey(im_sal, 150)
            pdf.image(im_sal, 5, 150, link='', type='', w=190, h=297/3)
            # add footer date
            pdf.custom_footer(footer_txt)
        
        #### create last page with info for reproducibility
        # add page
        pdf.add_page()
        # add line contours
        pdf.lines()
        # add header
        pdf.custom_header(logo, txt1="MELD report", txt2=f"Patient ID: {subject.subject_id}")
        # add info box
        pdf.info_box(text_info_3)
        # add footer date
        pdf.custom_footer(footer_txt)
        
        # save pdf
        file_path = os.path.join(output_dir_sub, f"MELD_report_{subject.subject_id}.pdf")
        pdf.output(file_path, "F")

        print(get_m(f'MELD prediction report ready at {file_path}', subject_id, 'INFO'))

if __name__ == "__main__":
    # Set up experiment
    parser = argparse.ArgumentParser(description="create mgh file with predictions from hdf5 arrays")
    parser.add_argument(
        "--experiment_folder",
        help="Experiments folder",
    )
    parser.add_argument(
        "--experiment_name",
        help="subfolder to use, typically the ensemble model",
        default="ensemble_iteration",
    )
    parser.add_argument("--fold", default=None, help="fold number to use (by default all)")
    parser.add_argument(
        "--data_dir", default="", help="folder containing the input data T1 and FLAIR"
    )
    parser.add_argument(
        "--output_dir", default="", help="folder containing the output prediction and reports"
    )
    parser.add_argument("-ids","--subject_list",
                        default="",
                        help="Relative path to subject List containing id and site_code.",
                        required=False,
    )
    parser.add_argument('-id','--id',
                    help='Subjects ID',
                    required=False,
                    default=None)
    args = parser.parse_args()
    fold=args.fold
    data_dir=args.data_dir
    output_dir=args.output_dir
    subject_ids = np.loadtxt(args.list_ids, dtype="str", ndmin=1)
    experiment_path = os.path.join(MELD_DATA_PATH, args.experiment_folder)
    experiment_name=args.experiment_name
    model_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)

    if args.list_ids:
        try:
            sub_list_df = pd.read_csv(args.list_ids)
            subject_ids=np.array(sub_list_df.participant_id.values)
        except:
            subject_ids=np.array(np.loadtxt(args.list_ids, dtype='str', ndmin=1))     
    elif args.id:
        subject_ids=np.array([args.id])
    else:
        print('No ids were provided')
        subject_ids=None

    # select predictions files
    if fold == None:
        hdf_predictions = os.path.join(experiment_path, "results", f"predictions_{experiment_name}.hdf5")
    else:
        hdf_predictions = os.path.join(experiment_path, f"fold_{fold}", "results", f"predictions_{experiment_name}.hdf5")
       
    # Provide models parameter
    exp = Experiment(experiment_path= model_path, experiment_name=experiment_name)

    generate_prediction_report(
        subject_ids,
        data_dir=data_dir,
        hdf_predictions=hdf_predictions,
        exp=exp,
        output_dir=output_dir,
    )