import os
import sys
import re
from pathlib import Path

def test_license():

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

# call the test
test_license()  