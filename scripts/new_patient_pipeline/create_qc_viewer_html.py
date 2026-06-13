## This script creates an html viewer with T1, FLAIR, MRI images, MELD predictions and surfaces for quality check of segmentation


## To run : python create_qc_viewer_html.py -id <sub_id>


import os
import sys
import argparse
import glob
from meld_graph.freebrowse_viewer import _generate_html
import webbrowser
from pathlib import Path

            
def return_file(path, file_name):
    files = glob.glob(path)
    if len(files)>1 :
        print(f'ERROR: Find too much volumes for {file_name}. Check and remove the additional volumes with same key name') 
        return None
    elif not files:
        print(f'ERROR: Could not find {file_name} volume. Check if name follow the right nomenclature')
        return None
    else:
        return Path(files[0])

if __name__ == '__main__':

    #parse commandline arguments 
    parser = argparse.ArgumentParser(description='perform cortical parcellation using recon-all from freesurfer')
    parser.add_argument('-id','--id_subj',
                        help='Subject ID.',
                        required=True,)
    parser.add_argument('-meld_data','--meld_data',
                        help='MELD data folder.',
                        required=True,)
    parser.add_argument("-output",   
                        default='viewer.html',  
                        help=f"Output HTML file (default: viewer.html)")
    args = parser.parse_args()
    subject=str(args.id_subj)
    meld_data_path=args.meld_data
    
    # get subject fs folder 
    pred_dir = os.path.join(meld_data_path,'output', 'predictions_reports', subject)
    subject_fs_folder = os.path.join(meld_data_path, 'output', 'fs_outputs', subject)
    
    # Find T1 and FLAIR FS outputs if exists
    if not os.path.isdir(subject_fs_folder):
        print(f'Freesurfer outputs does not exist for this subject. Unable to perform qc')
        sys.exit()

    #select inputs files T1 and FLAIR
    t1w = return_file(os.path.join(subject_fs_folder, 'mri', 'orig.mgz'), 'orig.mgz')
    flair = return_file(os.path.join(subject_fs_folder, 'mri', 'FLAIR.prenorm.mgz'), 'FLAIR.prenorm.mgz')
    
    # select surface files
    pial_lh = return_file(os.path.join(subject_fs_folder, 'surf', 'lh.pial'), 'lh.pial')
    pial_rh = return_file(os.path.join(subject_fs_folder, 'surf', 'rh.pial'), 'rh.pial')
    white_lh = return_file(os.path.join(subject_fs_folder, 'surf', 'lh.white'), 'lh.white')
    white_rh = return_file(os.path.join(subject_fs_folder, 'surf', 'rh.white'), 'rh.white')

    #select predictions files
    lesion = return_file(os.path.join(pred_dir, 'predictions', 'prediction.nii*'), 'prediction')

    named = {
        "--t1w": t1w, "--flair": flair,
        "--pial-lh": pial_lh, "--white-lh": white_lh,
        "--pial-rh": pial_rh, "--white-rh": white_rh,
        "--lesion": lesion,
    }

    missing = [f"{flag}: {p}" for flag, p in named.items()
               if p is not None and not p.exists()]
    if missing:
        print("Error — file(s) not found:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    if all(p is None for p in named.values()):
        print("Error — no files provided. Use --t1w, --flair, --pial-lh, etc.")
        sys.exit(1)

    # tell name, path, colormap, opacity, vmin, vmax
    volumes = [
        ("FLAIR",  flair,  "gray",   1.0, None, None),
        ("T1w",    t1w,    "gray",   1.0, None, None),
        ("lesion", lesion, "random", 0.8, 0, 1),
    ]

    meshes = [
        ("lh.pial",  pial_lh,  [255, 255, 0, 255]),
        ("lh.white", white_lh, [255, 165, 0, 255]),
        ("rh.pial",  pial_rh,  [255, 255, 0, 255]),
        ("rh.white", white_rh, [255, 165, 0, 255]),
    ]

    # call freebrowse viewer
    output = Path(args.output)
    print(f"Generating {output}…")
    html = _generate_html(volumes, meshes)
    output.write_text(html, encoding="utf-8")
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Done — {output}  ({size_mb:.1f} MB)")
    webbrowser.open(output.resolve().as_uri())

    
    

    
