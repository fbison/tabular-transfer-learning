import matplotlib.pyplot as plt
from result_analysis import plot_save_fig


def plot_pca_curve(cumulative_variance, save_path):
    plt.figure()
    plt.plot(cumulative_variance)
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance")
    plot_save_fig(save_path)


def plot_box(data, labels, ylabel, save_path):
    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.ylabel(ylabel)
    plot_save_fig(save_path)