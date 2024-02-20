import os
import sys		
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
import os
from gensim.models import KeyedVectors
from tqdm import tqdm
from utils.transforms import get_transforms

from utils.data import DiscoverImageNetDataModule
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.glove import GloVe
from icecream import ic
from pytorch_lightning.metrics import Accuracy
from torch.utils.data import DataLoader

from utils.confusion_matrix import confusion_matrix_img
from utils.debug import get_classes
from utils.hubness import hubness_figure, Inverted_softmax
from utils.pseudo_label import acc_confidence_figure, progressive_pseudo_labeling
from utils.simple_dataset import get_dataset

mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 15: 13, 16: 14, 18: 15, 20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22, 27: 23, 28: 24, 30: 25, 32: 26, 33: 27, 35: 28, 36: 29, 37: 30, 39: 31, 40: 32, 41: 33, 42: 34, 44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 53: 44, 54: 45, 56: 46, 57: 47, 58: 48, 59: 49, 60: 50, 61: 51, 62: 52, 63: 53, 64: 54, 65: 55, 66: 56, 67: 57, 68: 58, 69: 59, 70: 60, 71: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 78: 67, 79: 68, 80: 69, 81: 70, 82: 71, 83: 72, 84: 73, 85: 74, 86: 75, 87: 76, 89: 77, 90: 78, 91: 79, 92: 80, 93: 81, 94: 82, 95: 83, 97: 84, 98: 85, 99: 86, 100: 87, 101: 88, 102: 89, 103: 90, 104: 91, 105: 92, 107: 93, 108: 94, 109: 95, 110: 96, 111: 97, 112: 98, 113: 99, 114: 100, 116: 101, 117: 102, 118: 103, 119: 104, 120: 105, 121: 106, 122: 107, 123: 108, 124: 109, 125: 110, 126: 111, 127: 112, 129: 113, 130: 114, 131: 115, 132: 116, 133: 117, 134: 118, 136: 119, 137: 120, 138: 121, 139: 122, 140: 123, 141: 124, 142: 125, 143: 126, 144: 127, 145: 128, 146: 129, 147: 130, 148: 131, 149: 132, 150: 133, 151: 134, 152: 135, 153: 136, 154: 137, 155: 138, 156: 139, 157: 140, 158: 141, 159: 142, 160: 143, 161: 144, 162: 145, 163: 146, 164: 147, 165: 148, 166: 149, 167: 150, 168: 151, 169: 152, 170: 153, 171: 154, 172: 155, 173: 156, 174: 157, 175: 158, 176: 159, 177: 160, 180: 161, 181: 162, 182: 163, 183: 164, 184: 165, 185: 166, 186: 167, 187: 168, 189: 169, 190: 170, 191: 171, 192: 172, 193: 173, 194: 174, 195: 175, 196: 176, 198: 177, 200: 178, 201: 179, 202: 180, 203: 181, 205: 182, 206: 183, 207: 184, 208: 185, 209: 186, 210: 187, 211: 188, 212: 189, 213: 190, 214: 191, 216: 192, 217: 193, 218: 194, 219: 195, 220: 196, 221: 197, 222: 198, 223: 199, 224: 200, 225: 201, 226: 202, 227: 203, 228: 204, 229: 205, 230: 206, 231: 207, 232: 208, 234: 209, 235: 210, 236: 211, 237: 212, 238: 213, 239: 214, 240: 215, 241: 216, 243: 217, 244: 218, 245: 219, 246: 220, 247: 221, 248: 222, 250: 223, 251: 224, 252: 225, 253: 226, 254: 227, 256: 228, 258: 229, 259: 230, 260: 231, 262: 232, 263: 233, 264: 234, 265: 235, 266: 236, 267: 237, 268: 238, 269: 239, 270: 240, 271: 241, 272: 242, 273: 243, 275: 244, 277: 245, 278: 246, 279: 247, 282: 248, 283: 249, 284: 250, 285: 251, 287: 252, 288: 253, 289: 254, 290: 255, 291: 256, 292: 257, 293: 258, 294: 259, 295: 260, 297: 261, 298: 262, 299: 263, 300: 264, 301: 265, 303: 266, 304: 267, 305: 268, 306: 269, 307: 270, 310: 271, 311: 272, 312: 273, 314: 274, 315: 275, 317: 276, 318: 277, 319: 278, 320: 279, 321: 280, 322: 281, 323: 282, 324: 283, 325: 284, 326: 285, 327: 286, 328: 287, 329: 288, 330: 289, 331: 290, 332: 291, 333: 292, 334: 293, 335: 294, 336: 295, 337: 296, 338: 297, 339: 298, 340: 299, 341: 300, 342: 301, 343: 302, 345: 303, 346: 304, 347: 305, 348: 306, 349: 307, 350: 308, 351: 309, 352: 310, 353: 311, 354: 312, 355: 313, 356: 314, 357: 315, 358: 316, 359: 317, 360: 318, 361: 319, 362: 320, 363: 321, 364: 322, 365: 323, 368: 324, 369: 325, 370: 326, 371: 327, 372: 328, 373: 329, 374: 330, 375: 331, 376: 332, 378: 333, 379: 334, 380: 335, 381: 336, 382: 337, 383: 338, 384: 339, 385: 340, 386: 341, 387: 342, 388: 343, 389: 344, 390: 345, 391: 346, 392: 347, 394: 348, 395: 349, 396: 350, 397: 351, 399: 352, 400: 353, 402: 354, 403: 355, 404: 356, 405: 357, 406: 358, 407: 359, 410: 360, 411: 361, 412: 362, 413: 363, 414: 364, 415: 365, 416: 366, 417: 367, 418: 368, 419: 369, 420: 370, 421: 371, 422: 372, 423: 373, 424: 374, 425: 375, 426: 376, 427: 377, 428: 378, 429: 379, 430: 380, 431: 381, 432: 382, 433: 383, 434: 384, 435: 385, 437: 386, 438: 387, 440: 388, 441: 389, 442: 390, 443: 391, 444: 392, 445: 393, 447: 394, 449: 395, 450: 396, 451: 397, 452: 398, 454: 399, 456: 400, 457: 401, 458: 402, 459: 403, 460: 404, 461: 405, 462: 406, 463: 407, 464: 408, 465: 409, 466: 410, 467: 411, 468: 412, 469: 413, 470: 414, 471: 415, 472: 416, 473: 417, 474: 418, 475: 419, 477: 420, 478: 421, 479: 422, 480: 423, 481: 424, 482: 425, 483: 426, 484: 427, 485: 428, 486: 429, 487: 430, 488: 431, 489: 432, 490: 433, 491: 434, 492: 435, 495: 436, 497: 437, 498: 438, 499: 439, 500: 440, 501: 441, 503: 442, 504: 443, 505: 444, 506: 445, 507: 446, 508: 447, 509: 448, 511: 449, 512: 450, 513: 451, 514: 452, 515: 453, 516: 454, 517: 455, 518: 456, 519: 457, 520: 458, 521: 459, 523: 460, 524: 461, 525: 462, 527: 463, 528: 464, 529: 465, 530: 466, 531: 467, 532: 468, 533: 469, 534: 470, 535: 471, 536: 472, 537: 473, 538: 474, 539: 475, 540: 476, 541: 477, 542: 478, 544: 479, 545: 480, 546: 481, 547: 482, 548: 483, 549: 484, 550: 485, 551: 486, 552: 487, 553: 488, 554: 489, 555: 490, 556: 491, 557: 492, 558: 493, 559: 494, 560: 495, 561: 496, 562: 497, 564: 498, 565: 499, 567: 500, 568: 501, 569: 502, 571: 503, 572: 504, 573: 505, 574: 506, 575: 507, 576: 508, 577: 509, 578: 510, 579: 511, 580: 512, 581: 513, 582: 514, 585: 515, 586: 516, 587: 517, 588: 518, 590: 519, 591: 520, 592: 521, 593: 522, 594: 523, 595: 524, 596: 525, 597: 526, 598: 527, 599: 528, 600: 529, 601: 530, 602: 531, 603: 532, 604: 533, 605: 534, 606: 535, 607: 536, 608: 537, 609: 538, 610: 539, 611: 540, 612: 541, 613: 542, 614: 543, 615: 544, 617: 545, 618: 546, 619: 547, 620: 548, 621: 549, 622: 550, 624: 551, 625: 552, 626: 553, 627: 554, 628: 555, 629: 556, 630: 557, 631: 558, 632: 559, 633: 560, 634: 561, 635: 562, 636: 563, 638: 564, 639: 565, 640: 566, 641: 567, 643: 568, 644: 569, 646: 570, 647: 571, 648: 572, 649: 573, 650: 574, 651: 575, 652: 576, 653: 577, 654: 578, 655: 579, 656: 580, 657: 581, 658: 582, 659: 583, 660: 584, 661: 585, 662: 586, 663: 587, 664: 588, 667: 589, 668: 590, 669: 591, 670: 592, 671: 593, 673: 594, 675: 595, 676: 596, 677: 597, 678: 598, 679: 599, 680: 600, 682: 601, 683: 602, 684: 603, 685: 604, 687: 605, 688: 606, 689: 607, 690: 608, 691: 609, 692: 610, 693: 611, 694: 612, 695: 613, 696: 614, 698: 615, 699: 616, 700: 617, 701: 618, 702: 619, 703: 620, 704: 621, 705: 622, 707: 623, 708: 624, 709: 625, 710: 626, 711: 627, 712: 628, 713: 629, 714: 630, 715: 631, 717: 632, 718: 633, 720: 634, 721: 635, 722: 636, 723: 637, 724: 638, 725: 639, 727: 640, 728: 641, 729: 642, 730: 643, 731: 644, 732: 645, 733: 646, 734: 647, 735: 648, 736: 649, 737: 650, 738: 651, 739: 652, 741: 653, 743: 654, 744: 655, 745: 656, 746: 657, 747: 658, 748: 659, 749: 660, 750: 661, 752: 662, 753: 663, 754: 664, 755: 665, 756: 666, 758: 667, 760: 668, 761: 669, 762: 670, 763: 671, 765: 672, 766: 673, 767: 674, 769: 675, 771: 676, 772: 677, 773: 678, 774: 679, 775: 680, 776: 681, 777: 682, 779: 683, 780: 684, 781: 685, 783: 686, 785: 687, 786: 688, 787: 689, 788: 690, 789: 691, 790: 692, 794: 693, 795: 694, 796: 695, 797: 696, 798: 697, 799: 698, 800: 699, 801: 700, 802: 701, 803: 702, 804: 703, 805: 704, 806: 705, 808: 706, 809: 707, 810: 708, 811: 709, 812: 710, 813: 711, 814: 712, 815: 713, 816: 714, 817: 715, 818: 716, 819: 717, 820: 718, 822: 719, 823: 720, 824: 721, 826: 722, 827: 723, 828: 724, 829: 725, 830: 726, 831: 727, 832: 728, 833: 729, 834: 730, 835: 731, 836: 732, 837: 733, 838: 734, 839: 735, 841: 736, 842: 737, 843: 738, 844: 739, 845: 740, 846: 741, 847: 742, 848: 743, 849: 744, 850: 745, 851: 746, 852: 747, 853: 748, 854: 749, 855: 750, 856: 751, 857: 752, 858: 753, 859: 754, 860: 755, 862: 756, 863: 757, 864: 758, 865: 759, 866: 760, 867: 761, 868: 762, 869: 763, 870: 764, 872: 765, 873: 766, 874: 767, 875: 768, 876: 769, 877: 770, 878: 771, 879: 772, 880: 773, 881: 774, 882: 775, 884: 776, 885: 777, 886: 778, 887: 779, 888: 780, 889: 781, 890: 782, 891: 783, 893: 784, 894: 785, 895: 786, 896: 787, 898: 788, 899: 789, 900: 790, 901: 791, 902: 792, 903: 793, 904: 794, 905: 795, 906: 796, 907: 797, 908: 798, 909: 799, 911: 800, 912: 801, 913: 802, 915: 803, 916: 804, 917: 805, 918: 806, 919: 807, 920: 808, 921: 809, 922: 810, 923: 811, 924: 812, 925: 813, 926: 814, 927: 815, 928: 816, 929: 817, 930: 818, 931: 819, 935: 820, 936: 821, 937: 822, 938: 823, 939: 824, 941: 825, 942: 826, 943: 827, 944: 828, 945: 829, 946: 830, 947: 831, 948: 832, 949: 833, 950: 834, 951: 835, 952: 836, 953: 837, 954: 838, 955: 839, 956: 840, 957: 841, 958: 842, 959: 843, 960: 844, 961: 845, 962: 846, 963: 847, 964: 848, 965: 849, 966: 850, 967: 851, 968: 852, 969: 853, 970: 854, 971: 855, 972: 856, 973: 857, 974: 858, 975: 859, 976: 860, 977: 861, 978: 862, 979: 863, 980: 864, 981: 865, 982: 866, 983: 867, 984: 868, 985: 869, 986: 870, 987: 871, 988: 872, 989: 873, 990: 874, 991: 875, 993: 876, 994: 877, 995: 878, 996: 879, 997: 880, 999: 881, 17: 882, 43: 883, 106: 884, 178: 885, 188: 886, 204: 887, 242: 888, 280: 889, 281: 890, 316: 891, 393: 892, 436: 893, 446: 894, 448: 895, 455: 896, 526: 897, 570: 898, 589: 899, 616: 900, 623: 901, 665: 902, 674: 903, 706: 904, 751: 905, 759: 906, 768: 907, 784: 908, 821: 909, 883: 910, 932: 911}
swapped_dict = {value: key for key, value in mapping.items()}

def mapping_index(inp):
    mapped_tensor = torch.tensor([mapping[t.item()] if t.item() in mapping else 1001 for t in inp])
    return mapped_tensor


# Tip-Adapter's few-shot classifier
class cache(nn.Module):
    # sim__func: str, 'cosine' or 'l2'
    # knn: k nearest neighbors
    def __init__(self, features, targets, w2v,dataset,
                normalize_feature = True, sim_func='cosine_softmax',
                temperature=0.1, alpha=1,
                inverted_softmax = None, knn = 50,
                num_classes = 80, k = 32
                ):
        super().__init__()

        if normalize_feature:
            features = F.normalize(features, dim=1)
        w2v = F.normalize(w2v, dim=1)
        if dataset == "ImageNet":
            w2v = w2v[list(mapping.keys())]
        self.num_classes = w2v.shape[0]
        self.w2v = nn.Parameter(w2v, requires_grad = False)
        self.features = nn.Parameter(features)
        self.targets = targets # print it
        print(f"During cache build, self.targets =\" {self.targets}")
        self.targets_w2v = nn.Parameter(w2v[targets])
        self.normalize_feature = normalize_feature
        self.sim_func = sim_func
        self.temperature = temperature
        self.alpha = alpha
        self.inverted_softmax = inverted_softmax
        self.knn = knn

        self.base_features = self.features
        self.base_targets = targets

        self.queue_head_bias = [0] * num_classes # bias relative to the class's queue
        self.num_labeled_classes = num_classes
        self.k = k

        self.features.requires_grad = False

    def update_memory(self, features, targets):
        if self.normalize_feature:
            features = F.normalize(features)
        features = features.float()
        dequeue_idx = []
        enqueue_idx = []
        for i in range(self.num_labeled_classes):
            idx_i = torch.nonzero(targets == i)
            if idx_i.shape[0] > 0:
                idx = idx_i[0, 0]
                enqueue_idx.append(idx)
                dequeue_idx.append(i * self.k + self.queue_head_bias[i])
                self.queue_head_bias[i] += 1
                if self.queue_head_bias[i] == self.k:
                    self.queue_head_bias[i] = 0
        enqueue_idx = torch.tensor(enqueue_idx)
        updated_features = self.features.clone()
        updated_features[dequeue_idx] = features[enqueue_idx]
        self.features = nn.Parameter(updated_features)


    def renew_memory(self, features, targets): #fix bug
        if self.normalize_feature:
            features = F.normalize(features,dim = 1)

        # 用新的特征张量替换旧的特征张量
        indices = (self.targets == targets[0].item())
        new_features = self.features.clone()
        new_features[indices] = features.to(self.features.dtype)
        self.features = nn.Parameter(new_features)

        # features = features.detach()
        # self.features[indices] = features.to(self.features.dtype)

    # def add_memory(self, features, targets): # add unseen feature to MB    
    #     if self.normalize_feature:
    #         features = F.normalize(features)
    #     features = nn.Parameter(features)
    #     self.features = nn.Parameter(torch.cat((self.features, features)))
    #     # self.features = nn.Parameter(self.features)
    #     self.targets = torch.cat((self.targets, targets))
    #     self.targets_w2v = nn.Parameter(self.w2v[self.targets])

    
    def add_memory(self, features, targets):
        if self.normalize_feature:
            features = F.normalize(features,dim=1)
        features = features.float()

        # # 扩展 Memory Bank 以包含新的类别
        # for cluster, class_name in assigned_classes.items():
        #     # 假设已经确定 cluster 中的样本属于 class_name 类别
        #     # 选择 cluster 中的样本
        #     cluster_indices = torch.nonzero(self.epoch_pred_cluster == cluster).reshape(-1)
        #     selected_indices = torch.randperm(cluster_indices.size(0))[:self.k]

            # 添加这些样本的特征和类别到 Memory Bank
        print("self.features2",self.features.shape)
        print("features2",features.shape)

        self.features = nn.Parameter(torch.cat((self.features, features), dim=0))
        self.targets = torch.cat((self.targets, targets))

        # 更新 targets_w2v
        self.targets_w2v = nn.Parameter(self.w2v[self.targets])


    def topk_visual_neighbor(self, x, k):
        if self.normalize_feature:
            x = F.normalize(x, dim=-1)
        
        sims = x @ self.features.T
        if self.inverted_softmax:
            sims = self.inverted_softmax.update(sims)

        if self.sim_func == 'cosine_softmax':
            sims = torch.softmax(sims / self.temperature, dim=-1)
        elif self.sim_func == 'cosine_exp':
            sims = ((-1) * (self.alpha - self.alpha * sims)).exp()
        
        topk_sim, topk_idx = torch.topk(sims, k)
        ic(topk_idx.shape)
        topk_idices = torch.flatten(topk_idx)
        topk_targets = self.targets[topk_idices]
        return topk_targets

    def count_hubness(self, x):
        if self.normalize_feature:
            x = F.normalize(x, dim=-1)
        
        sims = x @ self.features.T # TODO: distance + -exp
        if self.inverted_softmax:
            sims = self.inverted_softmax.update(sims)

        if self.sim_func == 'cosine_softmax':
            sims = torch.softmax(sims / self.temperature, dim=-1)
        elif self.sim_func == 'cosine_exp':
            sims = ((-1) * (self.alpha - self.alpha * sims)).exp()

        visual_hub_cnt = torch.zeros((self.features.shape[0], ), dtype=torch.long)
        indices = torch.argmax(sims, dim=-1)
        for i in indices:
            visual_hub_cnt[i] += 1

        sync_w2v = sims @ self.targets_w2v
        # sync_w2v = self.targets_w2v[sims.argmax(dim=-1)] # DEBUG, use nearest visual neighbor's word vector
        sync_w2v = F.normalize(sync_w2v)
        
        logits = 100.0 * sync_w2v @ self.w2v.T
        return logits, visual_hub_cnt

    def forward(self, x):
        if self.normalize_feature:
            x = F.normalize(x, dim=-1)
        
        sims = x @ self.features.T  # TODO: -exp (batch,512) (512, 85)
        print("sims,",sims.shape)
        print("x,",x.shape)
        print("self.features,",self.features.shape)

        if self.inverted_softmax:
            sims = self.inverted_softmax.update(sims)

        if self.knn:
            value, _ = torch.kthvalue(-sims, self.knn, keepdim=True)
            value = -value
            sims[sims < value] = -1000

        if self.sim_func == 'cosine_softmax':
            sims = torch.softmax(sims / self.temperature, dim=-1)
        elif self.sim_func == 'cosine_exp':
            sims = ((-1) * (self.alpha - self.alpha * sims)).exp()
        print("self.targets_w2v",self.targets_w2v.shape)
        sync_w2v = sims @ self.targets_w2v # (batch 85) ()
        # sync_w2v = self.targets_w2v[sims.argmax(dim=-1)] # DEBUG, use nearest visual neighbor's word vector
        sync_w2v = F.normalize(sync_w2v)
        print("sync_w2v,",sync_w2v.shape)
        
        logits = sync_w2v @ self.w2v.T
        print("logits,",logits.shape)
        return logits

    def combine(self, x, y, y_factor):
        # combine two distributions
        if y_factor == 0:
            return x
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        x = (x - mean) / std
        std, mean = torch.std_mean(y, dim=-1, keepdim=True)
        y = (y - mean) / std
        return x + y * y_factor

    def oracle(self, labels):
        # debug only
        # output logits: 100 for target, 30 for others
        batch_size = labels.shape[0]
        num_classes = self.w2v.shape[0]
        ret = torch.ones((batch_size, num_classes)) * 30.0
        for i, label in enumerate(labels):
            ret[i, label] = 100.0
        return ret


def build_cache(dataset, data_path, k_shot, encoder, w2v_path,
                num_labeled_classes, num_unlabeled_classes,
                load_path = None, save_path = None, augment_epoch = 10,
                temperature = 0.1, w2v_type = 'glove',
                inverted_softmax = False, probe_size = 5000, inverted_temperature = 0.1,
                knn = 50
            ):
    if load_path:
        return torch.load(load_path)

    # -------------------------- sample k_shot images -------------------
    train_dataset = get_dataset(dataset, data_path, transform='train')
    debug_dataset = get_dataset(dataset, data_path, transform='val')
    # if dataset == 'ImageNet':
    #     print(train_dataset.targets)
    #     train_dataset.targets = mapping_index(train_dataset.targets)
    #     debug_dataset.targets = mapping_index(debug_dataset.targets)

    print("train_dataset.targets",train_dataset.targets)
    kshot_indices = []
    if dataset == "ImageNet":
        print("dataset is imagenet!")
        for i in range(num_labeled_classes):
            indices = np.nonzero(np.array(train_dataset.targets) == swapped_dict[i])[0]
            indices = np.random.choice(indices, size=(k_shot, ), replace=False)
            kshot_indices.append(indices)
    else:
        for i in range(num_labeled_classes):
            indices = np.nonzero(np.array(train_dataset.targets) == i)[0]
            indices = np.random.choice(indices, size=(k_shot, ), replace=False)
            kshot_indices.append(indices)
    kshot_indices = np.concatenate(kshot_indices)
    # assert kshot_indices.shape[0] == k_shot * num_labeled_classes
    
    kshot_subset = torch.utils.data.Subset(train_dataset, kshot_indices)
    train_loader = torch.utils.data.DataLoader(kshot_subset, batch_size=256, num_workers=8, shuffle=False)
    debug_kshot_subset = torch.utils.data.Subset(debug_dataset, kshot_indices)
    debug_loader = torch.utils.data.DataLoader(debug_kshot_subset, batch_size=256, num_workers=8, shuffle=False)

    # ---------------------------- extract features ---------------------
    train_images_targets = []
    train_images_features_agg = []
    encoder = encoder.to('cuda')
    encoder.eval()
    with torch.no_grad():
        for augment_idx in range(augment_epoch):
            train_images_features = []

            print('Augment time: {:} / {:}'.format(augment_idx, augment_epoch))
            for i, (images, target) in enumerate(tqdm(train_loader)):
                images = images.cuda()
                image_features = encoder(images)
                train_images_features.append(image_features)

                if augment_idx == 0:
                    target = target.cuda()
                    if dataset == "ImageNet":
                        train_images_targets.append(mapping_index(target))
                    else:
                        train_images_targets.append(target)

            images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
            train_images_features_agg.append(images_features_cat)

    train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
    train_images_features_agg = F.normalize(train_images_features_agg, dim=-1)
    train_images_targets = torch.cat(train_images_targets)
    ic(train_images_features_agg.shape)
    ic(train_images_targets.shape)

    # -------------------------- load w2v --------------------------------
    class_names = train_dataset.classes
    ic(len(class_names))
    if w2v_type == 'glove':
        glove = GloVe(w2v_path)
        w2v = torch.stack([glove[class_name] for class_name in class_names])
        # w2v = F.one_hot(torch.tensor(range(num_labeled_classes + num_unlabeled_classes))).float()   # DEBUG: one-hot target
        ic(w2v.shape)
    elif w2v_type == 'clip':
        if os.path.exists('./clip_text.pt') and os.path.exists('./clip_text_127.pt'):
            w2v = torch.load('./clip_text.pt')
        else:
            import clip
            clip_model, clip_preprocess = clip.load('RN50', device='cpu')
            for param in clip_model.parameters():
                param.requires_grad = False
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c.replace('_', ' ').replace('+', ' ')}") for c in class_names])
            # Calculate features
            with torch.no_grad():
                w2v = clip_model.encode_text(text_inputs)
            torch.save(w2v, './clip_text.pt')
    elif w2v_type == 'word2vec':
        word2vec = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        def encode_word2vec(model, word):
            try:
                ans = model[word]
            except KeyError:
                ic(('keyerror', word))
                word = word.replace('_', ' ').split(' ')
                ans = np.sum(np.stack([model[w] for w in word]), axis=0) / len(word)
            return ans
        w2v = np.stack([encode_word2vec(word2vec, class_name) for class_name in class_names])
        w2v = torch.tensor(w2v)
    else:
        raise NotImplementedError
    
    # ------------------------- inverted softmax -------------------------
    IS_module = None
    if inverted_softmax:
        probe_loader = DataLoader(debug_dataset, batch_size=256, shuffle=True)
        probe_set = []
        cnt = 0
        with torch.no_grad():
            for images, labels in tqdm(probe_loader):
                images = images.cuda()
                feats = encoder(images)
                if cnt <= probe_size:
                    probe_set.append(feats)
                    cnt += feats.shape[0]
                else:
                    break
        probe_set = torch.cat(probe_set, dim=0)[:probe_size]
        ic(probe_set.shape)
        IS_module = Inverted_softmax(inverted_temperature, probe_set, train_images_features_agg)

    res = cache(train_images_features_agg, train_images_targets, w2v, dataset, temperature = temperature,
                inverted_softmax=IS_module, knn=knn,
                num_classes = num_labeled_classes, k = k_shot
                )
    if save_path:
        torch.save(res, save_path)
    return res, debug_loader

if __name__ == '__main__':
    # test

    # ----------------- config --------------------------
    model_name = 'resnet18'

    debug_neighbors = False
    debug_hub = False
    pseudo_labeling = False

    inverted_softmax = False
    inverted_temperature = 0.05
    temperature = 0.1
    knn = 200

    w2v_type = 'glove'
    w2v_path = '../../MITip-main/glove.6B.300d.txt'
    # w2v_type = 'word2vec'
    # w2v_path = '../data/word2vec/GoogleNews-vectors-negative300.bin'

    dataset_split = 'CIFAR100-20' # CIFAR100-50, Imagenet1k

    if dataset_split == 'CIFAR100-20':
        dataset = 'CIFAR100'
        num_labeled_classes = 80
        num_unlabeled_classes = 20
        # model_path = '/home/zhangzr/wangweishuai/wws/SNCD/checkpoints/last.ckpt'
        model_path = '/home/zhangzr/wangweishuai/MITip-main/pretrain_checkpoints/pretrain-resnet18-CIFAR100-80_20.cp'
        data_path = '/home/zhangzr/wangweishuai/datasets/'
    elif dataset_split == 'CIFAR100-50':
        dataset = 'CIFAR100'
        num_labeled_classes = 50
        num_unlabeled_classes = 50
        model_path = 'pretrain_checkpoints/pretrain-resnet18-CIFAR100-50_50.cp'
        data_path = '../data/CIFAR100/'
    elif dataset_split == 'CIFAR10':
        dataset = 'CIFAR10'
        num_labeled_classes = 5
        num_unlabeled_classes = 5
        model_path = 'pretrain_checkpoints/pretrain-resnet18-CIFAR10.cp'
        data_path = '../data/CIFAR10/'
    else:
        raise NotImplementedError

    # ----------------- config end ----------------------


    from utils.nets import MultiHeadResNet, CustomResNet
    if model_name == 'resnet50':
        encoder = CustomResNet().to('cuda')
    elif model_name == 'resnet18':
        model = MultiHeadResNet(
            arch='resnet18',
            low_res=True,
            num_labeled=num_labeled_classes,
            num_unlabeled=num_unlabeled_classes,
            proj_dim=256,
            hidden_dim=2048,
            overcluster_factor=3,
            num_heads=5,
            num_hidden_layers=1,
        )
        state_dict = torch.load(model_path, map_location='cuda:0')
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(state_dict, strict=False)
        encoder = model.encoder.to('cuda')
    elif model_name == 'clip':  # TODO: fix this
        import clip
        clip_model, preprocess = clip.load('RN50')
        encoder = clip_model.visual.to('cuda')

    tip, debug_loader = build_cache(
        dataset, data_path, 16, encoder, w2v_path,
        num_labeled_classes, num_unlabeled_classes, temperature=temperature,
        inverted_softmax=inverted_softmax, inverted_temperature=inverted_temperature,
        w2v_type=w2v_type, knn=knn
        )
    tip = tip.to('cuda')

    # ----------------------- prepare datasets ----------------------------
    labeled_classes = range(num_labeled_classes)
    unlabeled_classes = range(
        num_labeled_classes, num_labeled_classes + num_unlabeled_classes
    )
    # val datasets
    val_dataset_train = get_dataset(dataset, data_path, train=True, transform='val')
    val_dataset_test = get_dataset(dataset, data_path, train=False, transform='val')

    # unlabeled classes, train set
    val_indices_unlab_train = np.where(
        np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
    )[0]
    val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)
    # unlabeled classes, test set
    val_indices_unlab_test = np.where(
        np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
    )[0]
    val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)
    # labeled classes, test set
    val_indices_lab_test = np.where(
        np.isin(np.array(val_dataset_test.targets), labeled_classes)
    )[0]
    val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)

    val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]
    val_unlab_train_loader = DataLoader(val_subset_unlab_train, 256, num_workers=8)
    val_lab_test_loader = DataLoader(val_subset_lab_test, 256, num_workers=8)

    classes = get_classes(dataset, data_path)

    encoder.eval()
    with torch.no_grad():
        # --------------------------- debug -------------------------------
        if debug_neighbors:
            ic('neighbors of [woman] and [whale]')
            debug_classes = ['woman', 'whale']
            k_list = [5]
            for clz in debug_classes:
                for i, name in enumerate(classes):
                    if name == clz:
                        debug_target = i
                        break
                
                ic(clz)
                debug_images = []
                target_list = []
                for images, target in tqdm(val_unlab_train_loader):
                    debug_images.append(images[target == debug_target])
                    target_list.append(target[target == debug_target])
                debug_images = torch.cat(debug_images, dim=0)
                target_list = torch.cat(target_list, dim=0)
                debug_images = debug_images.to('cuda')
                debug_images = encoder(debug_images)

                #  inspect visual nearest neighbor
                for k in k_list:
                    ic(k)
                    topk_targets = tip.topk_visual_neighbor(debug_images, k).to('cpu')
                    cnt = torch.zeros((num_labeled_classes, ))
                    for i in topk_targets:
                        cnt[i] += 1
                    vals, indices = torch.topk(cnt, 5)
                    for val, idx in zip(vals, indices):
                        ic((classes[idx], val))

                preds = tip(debug_images).to('cpu')
                preds = preds[..., num_labeled_classes:]
                labels = preds.argmax(dim=-1) + num_labeled_classes
                acc = Accuracy()
                acc.update(labels, target_list)
                ic(acc.compute())
                ic('predictions')
                cnt = torch.zeros((num_labeled_classes + num_unlabeled_classes, ))
                for i in labels:
                    cnt[i] += 1
                vals, indices = torch.topk(cnt, 5)
                for val, idx in zip(vals, indices):
                    ic((classes[idx], val))

        if debug_hub:
            ic('hubness')
            visual_hub = []
            w2v_hub = []
            for i, (images, target) in enumerate(tqdm(val_unlab_train_loader)):
                images = images.cuda()
                image_features = encoder(images)
                preds, v_hub = tip.count_hubness(image_features)
                preds = preds.to('cpu')[..., num_labeled_classes:]
                v_hub = v_hub.to('cpu')
                visual_hub.append(v_hub)

                w_hub = torch.zeros((num_unlabeled_classes,), dtype=torch.long)
                indices = torch.argmax(preds, dim=-1)
                for i in indices:
                    w_hub[i] += 1
                w2v_hub.append(w_hub)
            v_hub = torch.zeros_like(visual_hub[0])
            for x in visual_hub:
                v_hub += x
            w_hub = torch.zeros_like(w2v_hub[0])
            for x in w2v_hub:
                w_hub += x
            hubness_figure(v_hub.tolist(), 'visual hub')
            hubness_figure(w_hub.tolist(), 'w2v hub')

            ic('most popular samples')
            vals, indices = torch.topk(v_hub, 5)
            for val, idx in zip(vals, indices):
                ic((val, idx, classes[tip.targets[idx]]))
            ic('least popular samples')
            vals, indices = torch.topk(-v_hub, 5)
            for val, idx in zip(vals, indices):
                ic((val, idx, classes[tip.targets[idx]]))

        if pseudo_labeling:
            progressive_pseudo_labeling(encoder, tip, val_unlab_train_loader,
                                        num_labeled_classes, num_unlabeled_classes)

        ic('unlabeled train set')
        acc = Accuracy()
        y_true, y_pred, conf = [], [], []
        oracle_acc = Accuracy()
        for i, (images, target) in enumerate(tqdm(val_unlab_train_loader)):
            images = images.cuda()
            image_features = encoder(images)
            preds = tip(image_features).to('cpu')
            preds = preds[..., num_labeled_classes:]
            labels = preds.argmax(dim=-1) + num_labeled_classes
            ic(preds.shape)
            ic(labels.shape)
            acc.update(labels, target)
            y_true += target.tolist()
            y_pred += labels.tolist()

            # acc-confidence
            confidence, _ = torch.max(preds, dim=-1)  # TODO: try more confidence types, entropy etc
            conf += confidence.tolist()

            # test oracle
            oracle_acc.update(tip.oracle(target).argmax(dim=-1), target)

        ic(acc.compute())
        # confusion_matrix_img(y_true, y_pred).save('confusion.png')
        # acc_confidence_figure(torch.tensor(conf), torch.tensor(y_true), torch.tensor(y_pred))
        ic(oracle_acc.compute())

        preds_list = []
        labels_list = []
        tergets_list = []
        ic('labeled test set')
        acc = Accuracy()
        for i, (images, target) in enumerate(tqdm(val_lab_test_loader)):
            images = images.cuda()
            image_features = encoder(images)
            preds = tip(image_features).to('cpu')
            preds = preds[..., :num_labeled_classes]
            labels = preds.argmax(dim=-1)
            # if i == 0:
            #     preds_list = preds
            #     labels_list = labels
            #     targets_list = target
            # if i > 0:
            #     preds_list = torch.cat((preds_list,preds),dim=0)
            #     labels_list = torch.cat((labels_list,labels),dim=0)
            #     targets_list = torch.cat((targets_list,target),dim=0)

            acc.update(labels, target)
        # ic(preds_list.shape)
        # ic(labels_list.shape)
        # ic(targets_list.shape)

        ic(acc.compute())

        ic('k-shot set')
        acc = Accuracy()
        for i, (images, target) in enumerate(tqdm(debug_loader)):
            images = images.cuda()
            image_features = encoder(images)
            preds = tip(image_features).to('cpu')
            preds = preds[..., :num_labeled_classes]
            labels = preds.argmax(dim=-1)
            # ic(labels)
            acc.update(labels, target)
        ic(acc.compute())
