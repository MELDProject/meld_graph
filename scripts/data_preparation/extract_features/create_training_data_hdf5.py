##############################################################################

# This script writes out the per-vertex features and lesion classification for each patient and control to a .hdf5 file. 

#import relevant packages
import numpy as np
import nibabel as nb
import argparse
from scripts.data_preparation.extract_features.io_meld import save_subject
from meld_graph.paths import MELD_DATA_PATH, DEMOGRAPHIC_FEATURES_FILE
from meld_graph.tools_pipeline import get_m
import os

def create_training_data_hdf5(subject, subject_dir, output_dir):
    #list features
    features = np.array(['.on_lh.thickness.mgh', '.on_lh.w-g.pct.mgh', '.on_lh.curv.mgh','.on_lh.sulc.mgh',
        '.on_lh.gm_FLAIR_0.75.mgh', '.on_lh.gm_FLAIR_0.5.mgh', '.on_lh.gm_FLAIR_0.25.mgh',
        '.on_lh.gm_FLAIR_0.mgh', '.on_lh.wm_FLAIR_0.5.mgh', '.on_lh.wm_FLAIR_1.mgh',
        '.on_lh.pial.K_filtered.sm20.mgh'])
    n_vert=163842
    cortex_label=nb.freesurfer.io.read_label(os.path.join(subject_dir,'fsaverage_sym/label/lh.cortex.label'))
    medial_wall = np.delete(np.arange(n_vert),cortex_label)
    failed = save_subject(subject,
                          features = features,
                          medial_wall = medial_wall, 
                          subject_dir = subject_dir,
                          demographic_file=DEMOGRAPHIC_FEATURES_FILE,
                          output_dir=output_dir)
    if failed == True:
        print(get_m(f'Features not saved. Something went wrong', subject, 'ERROR'))
        return False
    else:
        print(get_m(f'All features have been extracted and saved in {output_dir}', subject, 'INFO'))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create feature matrix for all subjects')
    #TODO think about how to best pass a list
    parser.add_argument('-id','--id',
                        help='Subjects ID',
                        required=False,
                        default=None)
    parser.add_argument('-ids','--list_ids',
                        help='Subjects IDs in a text file',
                        required=False,
                        default=None)
    parser.add_argument('-sd','--subjects_dir',
                        help='Subjects directory...',
                        required=True,)
    parser.add_argument('-od','--output_dir',
                        type=str,
                        help='output directory to save hdf5',
                        required=True,)
    args = parser.parse_args()


    #save subjects dir and subject ids. import the text file containing subject ids
    subject_dir=args.subjects_dir
    output_dir=args.output_dir
    if args.list_ids:
        subject_ids=np.array(np.loadtxt(args.list_ids, dtype='str', ndmin=1))
    elif args.id:
        subject_ids=np.array([args.id])
    else:
        print('No ids were provided')
        subject_ids=None

    if subject_ids:
        for subject in subject_ids:
            create_training_data_hdf5(subject, subject_dir, output_dir)
        