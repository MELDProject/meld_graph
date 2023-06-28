import numpy as np
<<<<<<< HEAD
#from scipy.spatial.distance import mahalanobis
from meld_classifier.meld_cohort import MeldCohort,MeldSubject
from scipy.stats import linregress
import pandas as pd
import os
=======

# from scipy.spatial.distance import mahalanobis
from meld_classifier.meld_cohort import MeldCohort, MeldSubject
from scipy.stats import linregress
>>>>>>> de2692f1a214d48b8bacdf071f922206f1229de1


def surface_regression(metric_in, remove):
    remove_means = np.mean(remove)
    remove_data = remove - remove_means
    remove_slope = linregress(remove_data, metric_in).slope
    regress_scaled = remove_data * remove_slope
    metric_out = metric_in - regress_scaled
    return metric_out

<<<<<<< HEAD
def curvature_regress(subs,cohort,
    source_feature = '.on_lh.thickness.sm3.mgh',
    curv_feature = '.on_lh.curv.sm3.mgh',
    target_feature = ".on_lh.thickness_regression.sm3.mgh",
    hdf5_file_root = "{site_code}_{group}_featurematrix_combat_6_kernels.hdf5"):
    for si,sub_id in enumerate(subs):
        if si %100==0:
=======

def curvature_regress(
    subs,
    cohort,
    source_feature=".combat.on_lh.w-g.pct.sm3.mgh",
    curv_feature=".combat.on_lh.curv.sm3.mgh",
    target_feature=".combat.on_lh.w-g.pct_regression.sm3.mgh",
    hdf5_file_root="{site_code}_{group}_featurematrix_combat_6_kernels.hdf5",
):
    for si, sub_id in enumerate(subs):
        if si % 100 == 0:
>>>>>>> de2692f1a214d48b8bacdf071f922206f1229de1
            print(si)
        subj = MeldSubject(sub_id, cohort=cohort)
        try:
            thickness_lh = subj.load_feature_values(source_feature, hemi="lh")
            curvature_lh = subj.load_feature_values(curv_feature, hemi="lh")

<<<<<<< HEAD
            thickness_rh = subj.load_feature_values(source_feature, hemi="rh")
            curvature_rh = subj.load_feature_values(curv_feature, hemi="rh")
            lh_reg= surface_regression(thickness_lh[cohort.cortex_mask],
                                curvature_lh[cohort.cortex_mask]
                    )
            rh_reg = surface_regression(thickness_rh[cohort.cortex_mask],
                                        curvature_rh[cohort.cortex_mask]
                            )
            vals_reg = np.concatenate([lh_reg,rh_reg])
            subj.write_feature_values(target_feature, 
                                    vals_reg, hemis=['lh','rh'],
                                    hdf5_file_root=hdf5_file_root)
        except KeyError:
            print(sub_id)
    return


if __name__ == '__main__':
    hdf5_file_root = "{site_code}_{group}_featurematrix_smoothed_6_kernels.hdf5"
    dataset='MELD_dataset_V6.csv'
    cohort = MeldCohort(
        hdf5_file_root=hdf5_file_root, dataset=dataset
    )
    subs_orig=cohort.get_subject_ids(group='both')
    df = pd.read_csv(os.path.join(cohort.data_dir,cohort.dataset))
    subs_all = df['subject_id'].values
    subs = np.setdiff1d(subs_all,subs_orig)
    subs = ['MELD_H12_15T_FCD_0006','MELD_H12_3T_FCD_0013']
    print('rerunning on:',len(subs))
    features=['.on_lh.gm_FLAIR_0.75.sm3.mgh',
        '.on_lh.gm_FLAIR_0.5.sm3.mgh',
        '.on_lh.gm_FLAIR_0.25.sm3.mgh',
        '.on_lh.wm_FLAIR_0.5.sm3.mgh',
        '.on_lh.wm_FLAIR_1.sm3.mgh']
    new_features = ['.on_lh.gm_FLAIR_0.75_regression.sm3.mgh',
        '.on_lh.gm_FLAIR_0.5_regression.sm3.mgh',
        '.on_lh.gm_FLAIR_0.25_regression.sm3.mgh',
        '.on_lh.wm_FLAIR_0.5_regression.sm3.mgh',
        '.on_lh.wm_FLAIR_1_regression.sm3.mgh',]
    features = ['.on_lh.thickness.sm3.mgh']
    new_features = ['.on_lh.thickness_regression.sm3.mgh']
    for fi, feature in enumerate(features):
        curvature_regress(subs,cohort,
        source_feature = feature,
        curv_feature = '.on_lh.curv.sm3.mgh',
        target_feature = new_features[fi],
        hdf5_file_root = hdf5_file_root
        )
=======
        thickness_lh = subj.load_feature_values(source_feature, hemi="lh")
        curvature_lh = subj.load_feature_values(curv_feature, hemi="lh")

        thickness_rh = subj.load_feature_values(source_feature, hemi="rh")
        curvature_rh = subj.load_feature_values(curv_feature, hemi="rh")
        lh_reg = surface_regression(thickness_lh[cohort.cortex_mask], curvature_lh[cohort.cortex_mask])
        rh_reg = surface_regression(thickness_rh[cohort.cortex_mask], curvature_rh[cohort.cortex_mask])
        vals_reg = np.concatenate([lh_reg, rh_reg])
        subj.write_feature_values(target_feature, vals_reg, hemis=["lh", "rh"], hdf5_file_root=hdf5_file_root)
    return


if __name__ == "__main__":
    hdf5_file_root = "{site_code}_{group}_featurematrix_combat_6_kernels_noCombat.hdf5",
    dataset = "MELD_dataset_V6.csv"
    cohort = MeldCohort(hdf5_file_root=hdf5_file_root, dataset=dataset)
    subs = cohort.get_subject_ids(group="both")
    features = [
        ".combat.on_lh.thickness.sm3.mgh",
    ]
    new_features = [
        ".combat.on_lh.thickness_regression.sm3.mgh",
    ]
    for fi, feature in enumerate(features):
        curvature_regress(
            subs,
            cohort,
            source_feature=feature,
            curv_feature=".combat.on_lh.curv.sm3.mgh",
            target_feature=new_features[fi],
            hdf5_file_root="{site_code}_{group}_featurematrix_combat_6_kernels_noCombat.hdf5",
        )
>>>>>>> de2692f1a214d48b8bacdf071f922206f1229de1
