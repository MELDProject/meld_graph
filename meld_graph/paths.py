import os
import pwd

username = pwd.getpwuid(os.getuid())[0]

for EXPERIMENT_PATH in [
    '/Users/hannah.spitzer/projects/MELD/experiments_graph',  # Hannah's local experiment folder
    "/rds/project/kw350/rds-kw350-meld/experiments_graph/{}".format(username),  # user-specific rds experiment folder
]:
    if os.path.exists(EXPERIMENT_PATH):
        print("Setting EXPERIMENT_PATH to " + EXPERIMENT_PATH)
        break
if not os.path.exists(EXPERIMENT_PATH):
    print('WARNING: EXPERIMENT_PATH not found, setting to "", need to add it to paths.py')
    EXPERIMENT_PATH = ""