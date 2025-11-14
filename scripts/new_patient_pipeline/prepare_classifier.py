import argparse
import os
from configparser import ConfigParser, NoOptionError
import sys
import shutil
import subprocess

def test_license():
    import re
    from pathlib import Path

    # Get the meld license variable
    meld_license_file = os.getenv("MELD_LICENSE", None)

    if meld_license_file is None: 
        print('ERROR: Could not find a MELD_LICENSE environment variable. Please ensure you have exported the MELD_LICENSE environment following the MELD Graph installation guidelines')
        sys.exit()
    if not os.path.isfile(meld_license_file): 
        print(f'ERROR: The file {meld_license_file} does not exist.\nPlease ensure you got the meld license file by filling the registration form provided in the MELD Graph installation guidelines and provided the right path to the file')
        sys.exit()

    # check that the license is correct
    text = Path(meld_license_file).read_text()
    m = re.search(r"License\s*ID[:\s]*([0-9]+)", text, re.IGNORECASE)
    if m:
        license_id = m.group(1)
        if not len(license_id) == 6:
            print("ERROR: The license ID provided does not seem correct.\nPlease ensure you got the correct meld license file by filling the registration form provided in the MELD Graph installation guidelines and provided the right path to the file")
            sys.exit()
    else:
        print(f"ERROR: The license file {meld_license_file} does not seem correct.\nPlease ensure you got the correct meld license file by filling the registration form provided in the MELD Graph installation guidelines and provided the right path to the file")
        sys.exit()

def prepare_meld_config():
    # get scripts dir (parent dir of dir that this file is in)
    SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # read config file from scripts_dir
    config_fname = os.path.join(SCRIPTS_DIR, 'meld_config.ini')

    def copy_config():
        print("Creating new meld_config.ini from meld_config.ini.example")
        shutil.copy(os.path.join(SCRIPTS_DIR, "meld_config.ini.example"), config_fname)

    def get_yn_input():
        r = input()
        while r.lower() not in ('y', 'n'):
            r = input("Unknown value. Either input y or n:\n")
        if r.lower() == 'y':
            return True
        return False

    def get_path_input():
        p = input("Please enter the full path to your MELD data folder:\n")
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            print(f"{p} is not a valid directory")
            return get_path_input()
        return p
    
    # create config file
    if os.path.isfile(config_fname):
        config = ConfigParser()
        config.read(config_fname)
        print(f"Found existing meld_config.ini in {config_fname}")
        try:
            # check that all relevant items are defined in config
            config.get("DEFAULT", "meld_data_path")
            config.get("develop", "base_path")
            config.get("develop", "experiment_path")
        except NoOptionError as e:
            print("Existing meld_config.ini is not in the right format. Would you like to recreate it now? (y/n)")
            if get_yn_input():
                copy_config()
            else:
                print("Exiting without setting up meld_config.ini.")
                sys.exit()
        current_data_path = config.get("DEFAULT", "meld_data_path")
        #default path /data doesn't work on mac, so ensure they reset if it doesn't exist.
        if not os.path.isdir(current_data_path) and not ("KEEP_DATA_PATH" in os.environ):
            print(f'The current MELD data folder path, {config.get("DEFAULT", "meld_data_path")} does not exist. Please reset it now.')
        else:
            print(f'The current MELD data folder path is {config.get("DEFAULT", "meld_data_path")}. Would you like to change it? (y/n)')
            if ("KEEP_DATA_PATH" in os.environ) or (not get_yn_input()):
                print("Leaving MELD data folder unchanged.")
                return
    else:
        copy_config()

    # fill in meld_data_path
    meld_data_path = get_path_input()
    config = ConfigParser()
    config.read(config_fname)
    config.set("DEFAULT", "meld_data_path", meld_data_path)
    with open(config_fname, "w") as configfile:
        config.write(configfile)
    print(f"Successfully changed MELD data folder to {meld_data_path}")   
    
if __name__ == '__main__':
    # ensure that all data is downloaded
    parser = argparse.ArgumentParser(description="Setup the classifier: Create meld_config.ini and download test data and pre-trained models")
    parser.add_argument('--skip-config', action="store_true", help="do not create meld_config.ini" )
    parser.add_argument('--skip-download-data', action = "store_true", help="do not attempt to download test data")
    parser.add_argument("--force-download", action="store_true", help="download data even if exists already")
    parser.add_argument("--update_test", action="store_true", help="only update the test data")
    args = parser.parse_args()

    from meld_graph.download_data import get_test_data, get_model, get_meld_params
    
    test_license()

    if args.update_test:
        get_test_data(force_download=True)
        print('Test data updated')
        sys.exit()
        
    # create and populate meld_config.ini
    if not args.skip_config:
        prepare_meld_config()

    # need to do this import here, because above we are setting up the meld_config.ini
    # which is read when using meld_classifier.paths
    if not args.skip_download_data:
        print("Downloading test data")
        get_test_data(args.force_download)
    print("Downloading meld parameters input")
    get_meld_params(args.force_download)
    print("Downloading model")
    get_model(args.force_download)
    print("Done.")
