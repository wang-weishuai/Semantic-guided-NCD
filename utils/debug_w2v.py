from utils.debug import get_classes
from icecream import ic
import torch.nn.functional as F
import torch
from utils.glove import GloVe
w2v_path = '../data/glove/glove.6B.300d.txt'

class_names = get_classes('CIFAR100', data_path='../data/CIFAR100')
# ic(class_names)
glove = GloVe(w2v_path)
w2v = torch.stack([glove[class_name] for class_name in class_names])
# w2v = F.one_hot(torch.tensor(range(num_labeled_classes + num_unlabeled_classes))).float()   # DEBUG: one-hot target
ic(w2v.shape)
ic(class_names[88])
debug_w2v = F.normalize(w2v, dim=1)
girl = F.normalize(glove['girl'], dim=0)
vals, indices = torch.topk(girl @ debug_w2v.T, 5)
for idx in indices:
    ic((idx, class_names[idx]))

center = F.normalize(torch.sum(w2v, dim=0), dim=0)
center_v2 = F.normalize(torch.sum(debug_w2v, dim=0), dim=0)

sims = debug_w2v @ center
sims_v2 = debug_w2v @ center_v2

ic('5 neighbors to unnormalized center')
vals, indices = torch.topk(sims, 20)
for idx in indices:
    ic((idx, class_names[idx]))


ic('5 neighbors to normalized center')
vals, indices = torch.topk(sims_v2, 20)
for idx in indices:
    ic((idx, class_names[idx]))