import os,sys,json,subprocess

def setup():
    from meld_graph.paths import MELD_DATA_PATH
    # Should only be run on mac
    if not "FREESURFER_HOME" in os.environ:
        if sys.platform == "darwin":
            if os.path.exists("/Applications/freesurfer/7.2.0"):
                os.environ["FREESURFER_HOME"] = "/Applications/freesurfer/7.2.0"
        
    freesurfercheck = subprocess.run(['/bin/bash', '-c', "type freeview"], capture_output=True)
    if freesurfercheck.returncode > 0:
        # A command that sources freesurfer
        source = 'source /Applications/freesurfer/7.2.0/SetUpFreeSurfer.sh'
        # Grab all of the environment variables 
        dump = 'python -c "import os,json;print(json.dumps(dict(os.environ)))"'
        # Source freesufer then grab all of the environment variables and store in penv
        penv = subprocess.run(["/bin/bash", "-c", f"{source} && {dump}"], stdout=subprocess.PIPE)
        # Load the environment variables in this process
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
