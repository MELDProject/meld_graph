import os
import pwd

username = pwd.getpwuid(os.getuid())[0]

# get scripts dir (parent dir of dir that this file is in)
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for EXPERIMENT_PATH in [
    '/Users/hannah.spitzer/projects/MELD/experiments_graph',  # Hannah's local experiment folder
    "/rds/project/kw350/rds-kw350-meld/experiments_graph/{}".format(username),  # user-specific rds experiment folder
    '/home/kw350/software/gdl/meld_classifier_GDL/scripts/'
]:
    if os.path.exists(EXPERIMENT_PATH):
        print("Setting EXPERIMENT_PATH to " + EXPERIMENT_PATH)
        break
if not os.path.exists(EXPERIMENT_PATH):
    print('WARNING: EXPERIMENT_PATH not found, setting to "", need to add it to paths.py')
    EXPERIMENT_PATH = ""

def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config