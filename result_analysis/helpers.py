import os

from matplotlib import pyplot as plt
import pickle
##import tikzplotlib

NOT_USED = "Not Used"


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

def strategies_order_per_imputation(imputation):
    if imputation == NOT_USED:
        return ['FS']
    else:
        return ['FS','LH-E2E', 'MLP-E2E', 'LH', 'MLP']