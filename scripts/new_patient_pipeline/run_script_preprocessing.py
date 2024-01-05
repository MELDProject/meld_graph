"""
Pipeline to prepare data from new patients : 
1) combat harmonise (make sure you have computed the combat harmonisation parameters for your site prior)
2) inter & intra normalisation
3) Save data in the "combat" hdf5 matrix
"""

## To run : python run_script_preprocessing.py -site <site_code> -ids <text_file_with_subject_ids> 

import os
import sys
import argparse
import pandas as pd
import numpy as np
import tempfile
from os.path import join as opj
from meld_graph.meld_cohort import MeldCohort
from meld_graph.data_preprocessing import Preprocess, Feature
from meld_graph.paths import BASE_PATH, MELD_PARAMS_PATH, MELD_DATA_PATH, NORM_CONTROLS_PARAMS_FILE, COMBAT_PARAMS_FILE, MELD_SITE_CODES
from meld_graph.tools_commands_prints import get_m

def create_dataset_file(subjects_ids, save_file):
    df=pd.DataFrame()
    if  isinstance(subjects_ids, str):
        subjects_ids=[subjects_ids]
    df['subject_id']=subjects_ids
    df['split']=['test' for subject in subjects_ids]
    df.to_csv(save_file)

def which_combat_file(site_code):
    file_site=os.path.join(BASE_PATH, f'MELD_{site_code}', f'{site_code}_combat_parameters.hdf5')
    if site_code=='TEST':
        site_code = 'H4'
    if site_code in MELD_SITE_CODES:
        print(get_m(f'Use combat parameters from MELD cohort', None, 'INFO'))
        return os.path.join(MELD_PARAMS_PATH,COMBAT_PARAMS_FILE)
    elif os.path.isfile(file_site):
        print(get_m(f'Use combat parameters from site', None, 'INFO'))
        return file_site
    else:
        print(get_m(f'Could not find combat parameters for {site_code}', None, 'WARNING'))
        return 'None'

def check_demographic_file(demographic_file, subject_ids):
    #check demographic file has the right columns
    try:
        df = pd.read_csv(demographic_file)
        if not any(ext in ';'.join(df.keys()) for ext in ['ID', 'Sex', 'Age at preoperative']):
            sys.exit(get_m(f'Error with column names', None, 'ERROR'))
    except Exception as e:
        sys.exit(get_m(f'Error with the demographic file provided for the harmonisation\n{e}', None, 'ERROR'))
    #check demographic file has the right subjects
    if set(subject_ids).issubset(set(np.array(df['ID']))):
        return demographic_file
    else:
        sys.exit(get_m(f'Missing subject in the demographic file', None, 'ERROR'))


def run_data_processing_new_subjects(subject_ids, site_code, combat_params_file=None, output_dir=BASE_PATH, withoutflair=False):
 
    # Set features and smoothed values
    if withoutflair:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
			}
    else:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
		'.on_lh.gm_FLAIR_0.25.mgh' : 3,
		'.on_lh.gm_FLAIR_0.5.mgh' : 3,
		'.on_lh.gm_FLAIR_0.75.mgh' :3,
		".on_lh.gm_FLAIR_0.mgh": 3,
		'.on_lh.wm_FLAIR_0.5.mgh' : 3,
		'.on_lh.wm_FLAIR_1.mgh' : 3,
    			}
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    features_combat = [feat.combat_feat(feature) for feature in features_smooth]
    
    ### INITIALISE ###
    #create dataset
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)  

    if combat_params_file==None:
        combat_params_file = which_combat_file(site_code)
    if combat_params_file=='need_harmonisation':
        sys.exit(get_m(f'You need to compute the combat harmonisation parameters for this site before to run combat', None, 'ERROR'))

    ### REGRESS THICKNESS ###
    if (".on_lh.thickness" in "".join(features_smooth)) and (".on_lh.curv" in "".join(features_smooth)):
        print(get_m(f'Regress thickness with curvature', subject_ids, 'STEP'))
        #create cohort for the new subject
        c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', dataset=tmp.name)
        #create object combat
        regress =Preprocess(c_smooth,
                        write_output_file='{site_code}_{group}_featurematrix_smoothed.hdf5',
                        data_dir=output_dir)
        #features names
        feature = [feat for feat in features_smooth if ".on_lh.thickness" in feat][0]
        curv_feature = [feat for feat in features_smooth if ".on_lh.curv" in feat][0]
        regress.curvature_regress(feature, curv_feature=curv_feature)

        #add features to list
        feat_regress = feat.regress_feat(feature)
        print(f'Add feature {feat_regress} in features')
        features_smooth = features_smooth + [feat_regress]
        features_combat = [feat.combat_feat(feature) for feature in features_smooth]
    
    
    ### COMBAT DATA ###
    #-----------------------------------------------------------------------------------------------
    print(get_m(f'Combat harmonise subjects', subject_ids, 'STEP'))
    #create cohort for the new subject
    c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', dataset=tmp.name)
    #create object combat
    combat =Preprocess(c_smooth,
                       write_output_file='{site_code}_{group}_featurematrix_combat.hdf5',
                       data_dir=output_dir)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.combat_new_subject(feature, combat_params_file)
    

    ###  INTRA, INTER & ASYMETRY ###
    #-----------------------------------------------------------------------------------------------
    print(get_m(f'Intra-inter normalisation & asymmetry subjects', subject_ids, 'STEP'))
    #create cohort to normalise
    c_combat = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5', dataset=tmp.name)
    # provide mean and std parameter for normalisation by controls
    param_norms_file = os.path.join(MELD_PARAMS_PATH, NORM_CONTROLS_PARAMS_FILE)
    # create object normalisation
    norm = Preprocess(c_combat,
                        write_output_file='{site_code}_{group}_featurematrix_combat.hdf5',
                        data_dir=output_dir)
    # call functions to normalise data
    for feature in features_combat:
        print(feature)
        norm.intra_inter_subject(feature, params_norm = param_norms_file)
        norm.asymmetry_subject(feature, params_norm = param_norms_file )

    tmp.close()


def new_site_harmonisation(subject_ids, site_code, demographic_file, output_dir=BASE_PATH, withoutflair=False):

    # Set features and smoothed values
    if withoutflair:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
			}
    else:
        features = {
		".on_lh.thickness.mgh": 3,
		".on_lh.w-g.pct.mgh" : 3,
		".on_lh.pial.K_filtered.sm20.mgh": None,
		'.on_lh.sulc.mgh' : 3,
		'.on_lh.curv.mgh' : 3,
		'.on_lh.gm_FLAIR_0.25.mgh' : 3,
		'.on_lh.gm_FLAIR_0.5.mgh' : 3,
		'.on_lh.gm_FLAIR_0.75.mgh' : 3,
		".on_lh.gm_FLAIR_0.mgh": 3,
		'.on_lh.wm_FLAIR_0.5.mgh' : 3,
		'.on_lh.wm_FLAIR_1.mgh' : 3,
    			}
    feat = Feature()
    features_smooth = [feat.smooth_feat(feature, features[feature]) for feature in features]
    
    ### INITIALISE ###
    #check enough subjects for harmonisation
    if len(np.unique(subject_ids))<20:
        print(get_m(f'We recommend to use at least 20 subjects for an accurate harmonisation of the data. Here you are using only {len(np.unique(subject_ids))}', None, 'WARNING'))

    #create dataset
    tmp = tempfile.NamedTemporaryFile(mode="w")
    create_dataset_file(subject_ids, tmp.name)

    check_demographic_file(demographic_file, subject_ids)
    
    ### REGRESS THICKNESS ###
    if (".on_lh.thickness" in "".join(features_smooth)) and (".on_lh.curv" in "".join(features_smooth)):
        print(get_m(f'Regress thickness with curvature', subject_ids, 'STEP'))
        #create cohort for the new subject
        c_smooth = MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', dataset=tmp.name)
        #create object combat
        regress =Preprocess(c_smooth,
                        write_output_file='{site_code}_{group}_featurematrix_smoothed.hdf5',
                        data_dir=output_dir)
        #features names
        feature = [feat for feat in features_smooth if ".on_lh.thickness" in feat][0]
        curv_feature = [feat for feat in features_smooth if ".on_lh.curv" in feat][0]
        regress.curvature_regress(feature, curv_feature=curv_feature)

        #add features to list
        feat_regress = feat.regress_feat(feature)
        print(f'Add feature {feat_regress} in features')
        features_smooth = features_smooth + [feat_regress]
    
    ### COMBAT DISTRIBUTED DATA ###
    #-----------------------------------------------------------------------------------------------
    print(get_m(f'Compute combat harmonisation parameters for new site', None, 'STEP'))
        
    #create cohort for the new subject
    c_smooth= MeldCohort(hdf5_file_root='{site_code}_{group}_featurematrix_smoothed.hdf5', 
                       dataset=tmp.name)
    #create object combat
    combat =Preprocess(c_smooth,
                           site_codes=[site_code],
                           write_output_file="MELD_{site_code}/{site_code}_combat_parameters.hdf5",
                           data_dir=output_dir)
    #features names
    for feature in features_smooth:
        print(feature)
        combat.get_combat_new_site_parameters(feature, demographic_file)

    tmp.close()

def run_script_preprocessing(site_code, list_ids=None, sub_id=None, output_dir=BASE_PATH, demographic_file=None, harmonisation_only=False, withoutflair=False, verbose=False):
    site_code = str(site_code)
    subject_id=None
    subject_ids=None
    if list_ids != None:
        list_ids=opj(MELD_DATA_PATH, list_ids)
        try:
            sub_list_df=pd.read_csv(list_ids)
            subject_ids=np.array(sub_list_df.ID.values)
        except:
            subject_ids=np.array(np.loadtxt(list_ids, dtype='str', ndmin=1)) 
        else:
                sys.exit(get_m(f'Could not open {subject_ids}', None, 'ERROR'))             
    elif sub_id != None:
        subject_id=sub_id
        subject_ids=np.array([sub_id])
    else:
        print(get_m(f'No ids were provided', None, 'ERROR'))
        print(get_m(f'Please specify both subject(s) and site_code ...', None, 'ERROR'))
        sys.exit(-1) 
       
    #check that combat parameters exist for this site or compute it
    combat_params_file = which_combat_file(site_code)
    if combat_params_file=='None':
        print(get_m(f'Compute combat parameters for {site_code} with subjects {subject_ids}', None, 'INFO'))
        if demographic_file == None:
            sys.exit(get_m(f'Please provide a demographic file using the flag "-demos" to harmonise your data', None, 'ERROR'))    
        else:
            #check that demographic file exist and is adequate
            demographic_file = os.path.join(MELD_DATA_PATH, demographic_file) 
            if os.path.isfile(demographic_file):
                print(get_m(f'Use demographic file {demographic_file}', None, 'INFO'))
                demographic_file = check_demographic_file(demographic_file, subject_ids) 
            else:
                sys.exit(get_m(f'Could not find demographic file provided {demographic_file}', None, 'ERROR'))
        #compute the combat parameters for a new site
        new_site_harmonisation(subject_ids, site_code=site_code, demographic_file=demographic_file, output_dir=output_dir, withoutflair=withoutflair)

    if not harmonisation_only:
        run_data_processing_new_subjects(subject_ids, site_code=site_code, output_dir=output_dir, withoutflair=withoutflair)

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='data-processing on new subject')
    #TODO think about how to best pass a list
    parser.add_argument("-id","--id",
                        help="Subject ID.",
                        default=None,
                        required=False,
                        )
    parser.add_argument("-ids","--list_ids",
                        default=None,
                        help="File containing list of ids. Can be txt or csv with 'ID' column",
                        required=False,
                        )
    parser.add_argument("-site",
                        "--site_code",
                        help="Site code",
                        required=True,
                        )
    parser.add_argument('-demos', '--demographic_file', 
                        type=str, 
                        help='provide the demographic files for the harmonisation',
                        required=False,
                        default=None,
                        )
    parser.add_argument('--harmo_only', 
                        action="store_true", 
                        help='only compute the harmonisation combat parameters, no further process',
                        required=False,
                        default=False,
                        )
    parser.add_argument("--withoutflair",
                        action="store_true",
                        default=False,
                        help="do not use flair information",
                        )
    parser.add_argument("--debug_mode", 
                        help="mode to debug error", 
                        required=False,
                        default=False,
                        action="store_true",
                        )

    
    args = parser.parse_args()
    print(args)

    run_script_preprocessing(
                    site_code=args.site_code,
                    list_ids=args.list_ids,
                    sub_id=args.id,
                    demographic_file=args.demographic_file,
                    harmonisation_only = args.harmo_only,
                    withoutflair=args.withoutflair,
                    verbose = args.debug_mode,
                    )