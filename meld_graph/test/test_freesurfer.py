#### tests for Freesurfer ####
# it test  : 
# - if Freesurfer is activated 
# - check Freesurfer version? 

import subprocess
from subprocess import Popen
import os
import pytest
from meld_graph.paths import MELD_DATA_PATH

@pytest.mark.slow
def test_freesurfer():
    #initialise variables
    fs_folder=os.path.join(MELD_DATA_PATH,'output','fs_outputs')
    id_test = 'MELD_TEST_3T_FCD_0011'

    # check freesurfer activated
    command = format(f"$FREESURFER_HOME/bin/recon-all -sd {fs_folder} -s {id_test} -all -dontrun")
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    assert proc.returncode<=1 , "Oopsie ! Freesurfer failing with error {}".format(stdout)

    #check version
    command = format("$FREESURFER_HOME/bin/recon-all -version")
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    stdout, stderr= proc.communicate()
    version_ok = stdout.count("7.3.")
    assert version_ok==0 , "Oopsie ! Your Freesurfer version is 7.3 and is not compatible with MELD pipeline. Please downgrade to 7.2"
    version_ok = stdout.count("7.4.")
    assert version_ok==0 , "Oopsie ! Your Freesurfer version is 7.4 and is not compatible with MELD pipeline. Please downgrade to 7.2"
 
