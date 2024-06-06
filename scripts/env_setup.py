import os,sys,json,subprocess

def setup():
    from meld_graph.paths import MELD_DATA_PATH
    if not "FREESURFER_HOME" in os.environ:
        if sys.platform == "darwin":
            if os.path.exists("/Applications/freesurfer/7.2.0"):
                os.environ["FREESURFER_HOME"] = "/Applications/freesurfer/7.2.0"
        
    freesurfercheck = subprocess.run(['/bin/bash', '-c', "which freeview"], capture_output=True)
    if freesurfercheck.returncode > 0:
        source = 'source /Applications/freesurfer/7.2.0/SetUpFreeSurfer.sh'
        dump = 'python -c "import os,json;print(json.dumps(dict(os.environ)))"'
        penv = subprocess.run(["/bin/bash", "-c", f"{source} && {dump}"], stdout=subprocess.PIPE)
        env = json.loads(penv.stdout)
        os.environ.update(env)
    if not "FS_LICENSE" in os.environ:
        if os.path.exists(f"{MELD_DATA_PATH}/license.txt"):
            print("setting license" + f"{MELD_DATA_PATH}/license.txt")
            os.environ["FS_LICENSE"] = f"{MELD_DATA_PATH}/license.txt"
        elif os.path.exists(f"{os.getcwd()}/license.txt"):
            print("setting license" + f"{os.getcwd()}/license.txt")
            os.environ["FS_LICENSE"] = f"{os.getcwd()}/license.txt"
        else:
            print("Couldn't find Freesurfer license file. Please copy license.txt to the meld folder or set FS_LICENSE manually")
            sys.exit(-1)
