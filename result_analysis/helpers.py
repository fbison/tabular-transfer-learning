import os

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)