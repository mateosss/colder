import os
import sys
from contextlib import contextmanager
from subprocess import Popen, PIPE, STDOUT

DEPTHS_DIR = "depths"
IMAGES_DIR = "images"


@contextmanager
def stdout_redirected(to=os.devnull):
    """Redirect stdout to the given file or file-like object.
    Thanks: https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w", encoding="utf-8") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def sh(command, silence=False):
    if not silence:
        print(command)

    # We use Popen here instead of subprocess.run to have live stdout/err/in but it works weird sometimes
    cmdrun = Popen(command, shell=True, stdout=PIPE, stderr=STDOUT, text=True) # tqdm has no \r
    # cmdrun = Popen(command, shell=True, stdout=PIPE, text=True) # ctrl+c stop doesnt work
    # cmdrun = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True) # no tqdm output

    output = []
    for line in cmdrun.stdout:
        if not silence:
            print(line, end="")
        output.append(line)
    cmdrun.wait()

    stdout = "".join(output)
    retcode = cmdrun.returncode
    return stdout, retcode


def shret(command, silence=False):
    "Executes command in shell and returns its exit status"
    _, ret = sh(command, silence)
    return ret


def shout(command, silence=False):
    "Executes command in shell and returns its stdout"
    out, _ = sh(command, silence)
    return out
