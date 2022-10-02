import argparse
import subprocess
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, type=str)
    args = parser.parse_args()

    print()

    process = subprocess.Popen([
        "tensorboard",
        "dev",
        "upload",
        "--logdir",
        args.logdir
    ], shell=False)
    process.communicate()
