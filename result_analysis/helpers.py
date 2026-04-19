import os

from matplotlib import pyplot as plt
import pickle
##import tikzplotlib


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_save_fig(path_complete, fig=None):
    plt.savefig(path_complete + ".png", dpi=300, bbox_inches="tight")
    plt.savefig((path_complete + ".pdf"), bbox_inches="tight")
    ##tikzplotlib.save(path_complete + ".tex")
    if fig is not None:
        with open(path_complete + ".fig.pickle", "wb") as f:
            pickle.dump(fig, f)
    plt.close()