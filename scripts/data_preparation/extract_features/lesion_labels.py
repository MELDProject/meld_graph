
import os
import argparse
import subprocess
from subprocess import Popen
from meld_graph.tools_pipeline import  get_m


def lesion_labels(subject_id, subjects_dir, verbose=False):
    if os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/lh.lesion_linked.mgh"):
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh"):
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject_id}/surf_meld/lh.lesion_linked.mgh --trg {subject_id}/xhemi/surf_meld/lh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/surf/lh.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False

    elif os.path.isfile(f"{subjects_dir}/{subject_id}/surf_meld/rh.lesion_linked.mgh"):
        if not os.path.isfile(f"{subjects_dir}/{subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh"):
            command = f"SUBJECTS_DIR={subjects_dir} mris_apply_reg --src {subject_id}/surf_meld/rh.lesion_linked.mgh --trg {subject_id}/xhemi/surf_meld/rh.on_lh.lesion.mgh --streg {subjects_dir}/{subject_id}/xhemi/surf/lh.fsaverage_sym.sphere.reg {subjects_dir}/fsaverage_sym/surf/lh.sphere.reg"
            proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            stdout, stderr= proc.communicate()
            if verbose:
                print(stdout)
            if proc.returncode!=0:
                print(get_m(f'COMMAND failing : {command} with error {stderr}', subject_id, 'ERROR'))
                return False


if __name__ == "__main__":
    #parse commandline arguments pointing to subject_dir etc
    parser = argparse.ArgumentParser(description='create lesion labels')
    parser.add_argument('subject_id', type=str,
                        help='subject_id')
    parser.add_argument('subjects_dir', type=str,
                        help='freesurfer subject directory ')
    args = parser.parse_args()
    #save subjects dir and subject ids. import the text file containing subject ids
    subject_id=args.subject_id
    subjects_dir=args.subject_id
    lesion_labels(subject_id, subjects_dir)