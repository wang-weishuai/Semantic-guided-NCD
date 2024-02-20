import matplotlib.pyplot as plt
import torch
from icecream import ic
from math import ceil
from tqdm import tqdm
from pytorch_lightning.metrics import Accuracy

def progressive_pseudo_labeling(encoder, model, dataloader,
                                num_labeled_classes, num_unlabeled_classes, step=0.05):
    # only for unlabeled classes, alphabetical split
    ic('progressive pseudo labeling')
    num_iters = ceil(1 / step)
    acc_list = []
    ps_acc_list = []

    for iter_id in range(num_iters):
        y_true, y_pred, conf, features = [], [], [], []

        for i, (images, target) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            image_features = encoder(images)
            preds = model(image_features).to('cpu')
            preds = preds[..., num_labeled_classes:]
            labels = preds.argmax(dim=-1) + num_labeled_classes
            y_true.append(target)
            y_pred.append(labels)
            features.append(image_features.to('cpu'))

            # acc-confidence
            confidence, _ = torch.max(preds, dim=-1)  # TODO: try more confidence types, entropy etc
            conf.append(confidence)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        conf = torch.cat(conf)
        features = torch.cat(features)
        acc = Accuracy()
        acc.update(y_pred, y_true)
        acc_list.append(acc.compute())

        # pseudo labeling
        num_samples = len(y_true)
        rate = (iter_id + 1) * step
        num_selected = min(num_samples, ceil(num_samples * rate))
        sorted_conf, _ = torch.sort(conf, descending=True)
        # ic((num_selected, sorted_conf.shape))
        threshold = sorted_conf[num_selected - 1]
        mask = conf >= threshold
        selected_features = features[mask]
        selected_labels = y_pred[mask]
        # selected_labels = y_true[mask] # DEBUG: oracle test
        model.update_memory(selected_features.to('cuda'), selected_labels.to('cuda'))
        ps_acc = Accuracy()
        ps_acc.update(selected_labels, y_true[mask])
        ps_acc_list.append(ps_acc.compute())
        
        ic((iter_id, num_selected, acc_list[-1], ps_acc_list[-1]))

# conf, y_true, y_pred: torch.Tensor
def acc_confidence_figure(conf, y_true, y_pred, step = 20, figname = 'acc-confidence plot'):
    n = y_pred.shape[0]
    width = n // step
    sorted_conf, _ = torch.sort(conf, descending=True)

    acc_list = []
    for i in range(step):
        pl_size = (i + 1) * width
        threshold = sorted_conf[pl_size - 1]
        mask = conf >= threshold
        tot = torch.sum(mask)
        true = torch.sum(y_true[mask] == y_pred[mask])
        acc = true / tot
        acc_list.append(acc)

        # count = [0] * 100
        # if (i == 0):
        #     for y in y_pred[mask]:
        #         count[y] += 1
        #     ic(count)

    x = [(i + 1) * 100 / step for i in range(step)]
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.plot(x, acc_list)
    plt.title('relationship between pseudo-label accuracy and confidence threshold')
    plt.savefig(figname)
    plt.clf()