from PIL import Image
import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn
import torch
import torch.nn.functional as F
import numpy as np

def unk_error_type(y_true, y_pred, n_labeled_classes):
    # return percentage of unk recall, unk pred to kwn, unk pred to wrong unk
    cm = confusion_matrix(y_true, y_pred)
    n_unlabeled_classes = cm.shape[0]
    tot_unk = np.sum(cm[n_labeled_classes:])
    unk_right = np.sum([cm[i, i] for i in range(n_labeled_classes, n_unlabeled_classes)])
    unk2kwn = np.sum(cm[n_labeled_classes:, :n_labeled_classes])
    unk2unk_err = np.sum(cm[n_labeled_classes:, n_labeled_classes:]) - unk_right
    if tot_unk == 0:
        print('WARNING: tot unk = 0')
        tot_unk = 1
    return unk_right / tot_unk, unk2kwn / tot_unk, unk2unk_err / tot_unk

def kl_divergence(y_true, y_pred, n_labeled_classes, n_unlabeled_classes):
    # return kl_divergence between pred distribution and true distribution
    cnt_pred, cnt_true = torch.zeros((n_unlabeled_classes)), torch.zeros((n_unlabeled_classes))
    for yt, yp in zip(y_true, y_pred):
        cnt_pred[yp] += 1
        cnt_true[yt] += 1
    tot = sum(cnt_true)
    distrib_pred = cnt_pred / tot
    distrib_true = cnt_true / tot

    distrib_uniform = torch.ones((n_unlabeled_classes,), dtype=float) / n_unlabeled_classes
    return F.kl_div(distrib_pred.log(), distrib_true), F.kl_div(distrib_uniform.log(), distrib_true)
        

def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image and return it
    https://stackoverflow.com/questions/8598673/how-to-save-a-pylab-figure-into-in-memory-file-which-can-be-read-into-pil-image/8598881
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def confusion_matrix_img(y_true, y_pred):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax = plt.gca())
    ret = fig2img(fig)
    plt.clf()
    return ret

def similarity_matrix_img(sim):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    ax = plt.gca()

    cax = seaborn.heatmap(sim, ax=ax, annot=True, fmt='.2f')
    return fig2img(fig)

def pred_gt_distribution(pred, gt, num_unlab):
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    ax = plt.gca()

    ax.hist([pred, gt], color=['r', 'b'], bins = range(num_unlab + 1), alpha=0.5)
    return fig2img(fig)