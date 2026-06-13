import os
import sys
from contextlib import contextmanager
from subprocess import run, PIPE

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


def sh(command):
    "Executes command in shell and returns its exit status"
    return run(command, shell=True, check=False).returncode


def shout(command, silence=False):
    "Executes command in shell and returns its stdout"
    stdout = run(command, shell=True, stdout=PIPE, check=False).stdout.decode("utf-8")
    if not silence:
        print(stdout)
    return stdout
