import logging as log
import subprocess
from subprocess import Popen, DEVNULL, STDOUT, check_call

def get_m(message, subject=None, type_message='INFO'):
    try:
        if not isinstance(subject, str):
            subject = ' '.join(subject)
        return f'{type_message} - {subject}: {message}'
    except:
        subject = None
        return f'{type_message}: {message}'

def run_command(command, verbose=False):
    # if verbose:
    #     print(get_m(command, None, 'COMMAND'))
    proc = Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8' )
    # if proc.stderr:
    #     raise subprocess.CalledProcessError(
    #             returncode = proc.returncode,
    #             cmd = proc.args,
    #             stderr = proc.stderr
    #             )
    # if (proc.stdout) and (verbose):
    #     print(get_m("Result: {}".format(proc.stdout.decode('utf-8')), None, 'COMMAND'))
    return proc