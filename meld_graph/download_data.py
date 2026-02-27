import urllib.request
import os
import numpy as np
from meld_graph.paths import MELD_DATA_PATH, DEFAULT_HDF5_FILE_ROOT, BASE_PATH
import sys
import shutil
import tempfile

def get_test_data(force_download=False):
    test_data_dir = os.path.join(BASE_PATH, "MELD_TEST")
    exists_patient = os.path.exists(os.path.join(test_data_dir, DEFAULT_HDF5_FILE_ROOT.format(site_code='TEST', group='patient')))
    exists_control = os.path.exists(os.path.join(test_data_dir, DEFAULT_HDF5_FILE_ROOT.format(site_code='TEST', group='control')))
    test_input_dir = os.path.join(MELD_DATA_PATH, "input")
    exists_test_input = os.path.exists(os.path.join(test_input_dir,'sub-test001'))
    if exists_patient and exists_control and exists_test_input:
        print("Test data exists. Specify --force-download to overwrite.")
        return test_data_dir
    else:
        print("Test data does not exists. Please run the step to prepare the classifier")

def check_data(force_download=False):
    for folder in ['input','output','model','meld_params']:
        exit = False
        if os.path.exists(os.path.join(MELD_DATA_PATH, folder)):
            print(f'The folder {folder} already exists at {MELD_DATA_PATH}.')
            exit = True
    if force_download:
        print('Data to download already (partially) exists. \nData will be overwritten.') 
    if exit and (force_download==False):
        print('Data to download already (partially) exists. \nDownload aborted. Please delete folders or provide a new path.')
        sys.exit()
        


def _fetch_url(url, fname):
    def dlProgress(count, blockSize, totalSize):
        percent = int(count*blockSize*100/totalSize)
        if not "SILENT" in os.environ:
            sys.stdout.write("\r" + url + "...%d%%" % percent)
            sys.stdout.flush()
    return urllib.request.urlretrieve(url, fname, reporthook=dlProgress)

def download_meld_graph_data(meld_data_path=MELD_DATA_PATH):
    """
    download meld graph data from GitHub release: model, parameters and test data 
    """
    url = "https://github.com/MELDProject/meld_graph/releases/download/meld_graph_data/meld_graph_data.zip"
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "meld_graph_data.zip"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "meld_graph_data.zip"), meld_data_path)
    print(f"\ndownloaded meld graph data to {meld_data_path}")
