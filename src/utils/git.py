import subprocess


def get_git_hash() -> str:
    cmd = "git rev-parse --short HEAD"
    hash = subprocess.check_output(cmd.split()).strip().decode("utf-8")
    return hash
