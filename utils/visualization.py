import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(x, y, filename):
    x_2d = TSNE().fit_transform(x)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y)