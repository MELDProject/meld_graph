import urllib.request
import os
import numpy as np
from meld_graph.paths import BASE_PATH, DEFAULT_HDF5_FILE_ROOT, EXPERIMENT_PATH, MODEL_NAME, MODEL_PATH, MELD_DATA_PATH
import sys
import shutil
import tempfile

# --- download data from figshare ---
def _fetch_url(url, fname):
    def dlProgress(count, blockSize, totalSize):
        percent = int(count*blockSize*100/totalSize)
        sys.stdout.write("\r" + url + "...%d%%" % percent)
        sys.stdout.flush()
    return urllib.request.urlretrieve(url, fname, reporthook=dlProgress)


def download_test_data():
    """
    Download test data from figshare
    """
    url = "https://figshare.com/ndownloader/files/38045313?private_link=256a8a952e2d1fd5a6fc"
    test_data_dir = MELD_DATA_PATH
    os.makedirs(test_data_dir, exist_ok=True)
    print('downloading test data to '+ test_data_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "test_data.tar.gz"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "test_data.tar.gz"), test_data_dir)
    print(f"\nunpacked data to {test_data_dir}")
    return test_data_dir

def download_meld_params():
    """
    Download meld parameters file from figshare
    """
    url = "https://figshare.com/ndownloader/files/36659463?private_link=dd245fc6b3c7642ed9e3"
    #print()
    meld_params_dir = MELD_DATA_PATH
    os.makedirs(meld_params_dir, exist_ok=True)
    print('downloading meld parameters to '+ meld_params_dir)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "meld_params.tar.gz"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "meld_params.tar.gz"), meld_params_dir)
    print(f"\nunpacked data to {meld_params_dir}")
    return meld_params_dir

def download_models():
    """
    download pretrained ensemble models and return experiment_name and experiment_dir
    """
    url = "https://figshare.com/ndownloader/files/31618988?private_link=2ed2d8cddbfdda5f00ae"
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "models.tar.gz"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "models.tar.gz"), os.path.dirname(EXPERIMENT_PATH))
    print(f"\ndownloaded models to {EXPERIMENT_PATH}")

# --- return path to data (and optionally download) --
def get_test_data(force_download=False):
    test_data_dir = os.path.join(BASE_PATH, "MELD_TEST")
    exists_patient = os.path.exists(os.path.join(test_data_dir, DEFAULT_HDF5_FILE_ROOT.format(site_code='TEST', group='patient')))
    exists_control = os.path.exists(os.path.join(test_data_dir, DEFAULT_HDF5_FILE_ROOT.format(site_code='TEST', group='control')))
    test_input_dir = os.path.join(MELD_DATA_PATH, "input")
    exists_test_input = os.path.exists(os.path.join(test_input_dir,'MELD_TEST_3T_FCD_0011'))
    if exists_patient and exists_control and exists_test_input:
        if force_download:
            print("Overwriting existing test data.")
            return download_test_data()
        else:
            print("Test data exists. Specify --force-download to overwrite.")
            return test_data_dir
    else:
        return download_test_data()

def get_model(force_download=False):
    # test if exists and do not download then
    if not os.path.exists(os.path.join(EXPERIMENT_PATH, MODEL_PATH)):
        download_models()
    else:
        if force_download:
            print("Overwriting existing model.")
            download_models()
        else:
            print("Model exists. Specify --force-download to overwrite.")
    return MODEL_PATH, MODEL_NAME

def get_meld_params(force_download=False):
    # test if exists and do not download then
    if not os.path.exists(os.path.join(MELD_DATA_PATH, 'meld_params')):
        download_meld_params()
    else:
        if force_download:
            print("Overwriting existing model.")
            download_meld_params()
        else:
            print("Model exists. Specify --force-download to overwrite.")
    return MELD_DATA_PATH, 'meld_params'