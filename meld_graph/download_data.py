import urllib.request
import os
import numpy as np
from meld_graph.paths import MELD_DATA_PATH
import sys
import shutil
import tempfile


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

def download_meld_graph_data():
    """
    download meld graph data from GitHub release: model, parameters and test data 
    """
    url = "https://github.com/MELDProject/meld_graph/releases/download/meld_graph_data/meld_graph_data.zip"
    with tempfile.TemporaryDirectory() as tmpdirname:
        # download to tmpdir
        _fetch_url(url, os.path.join(tmpdirname, "meld_graph_data.zip"))
        # unpack
        shutil.unpack_archive(os.path.join(tmpdirname, "meld_graph_data.zip"), MELD_DATA_PATH)
    print(f"\ndownloaded meld graph data to {MELD_DATA_PATH}")
