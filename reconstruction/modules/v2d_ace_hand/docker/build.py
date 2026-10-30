"""Build the pinned ACE inference image (dependencies installed with uv)."""
from pathlib import Path
import subprocess

if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    subprocess.run(['docker', 'build', '-t', 'v2d_ace_hand', '-f',
                    str(directory / 'Dockerfile'), str(directory)], check=True)
