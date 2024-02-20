import os		
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.mutual_information import mutual_information_loss
from scipy.optimize import linear_sum_assignment
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from icecream import ic
from utils import cache
import wandb
from scipy.special import comb

parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="SNCDcheck", type=str, help="wandb project")
parser.add_argument("--entity", default="pkuwws", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
parser.add_argument("--mutual_information", default=0.1, type=float, help= "mutual information loss weight")
parser.add_argument("--tip_loss", default=0, type=float, help="tip loss weight")
parser.add_argument("--save_cache", type=str, default=None, help="save cache")
parser.add_argument("--load_cache", type=str, default=None, help="load cache")
parser.add_argument("--freeze_encoder", default=False, action='store_true')
parser.add_argument("--mi_combine", type=float, default=0.1)
parser.add_argument("--concat_combine", default=False, action='store_true')
parser.add_argument("--pseudo_label_rate", type=float, default=0)
parser.add_argument("--oracle", default=False, action='store_true')
parser.add_argument("--zsl_loss", default=0, type=float) # self-labeling
parser.add_argument("--dynamic_cache", default=True, action="store_true")
parser.add_argument("--cluster_wise_ps", default=True)
parser.add_argument("--threshold", type = int, default=0)
parser.add_argument("--cluster_top_k", type = int, default=32)


def calculate_cluster_accuracy(cluster_predictions, true_labels, assigned_classes):
    cluster_stats = {}
    for cluster_id in assigned_classes.keys():  
        cluster_stats[cluster_id] = {'total': 0, 'correct': 0, 'real_class_counts': torch.zeros(20, dtype=torch.int32)}

    for cluster, true_label in zip(cluster_predictions, true_labels):
        if cluster.item() in assigned_classes:
            true_label = true_label.item()
            cluster = cluster.item()
            cluster_stats[cluster]['total'] += 1
            cluster_stats[cluster]['real_class_counts'][true_label] += 1
            if true_label == assigned_classes[cluster]:
                cluster_stats[cluster]['correct'] += 1

    for cluster in cluster_stats:
        
        total = cluster_stats[cluster]['total']
        correct = cluster_stats[cluster]['correct']
        cluster_stats[cluster]['accuracy'] = correct / total if total > 0 else 0

    return cluster_stats


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        if self.hparams.dynamic_cache:
            ic('using dynamic cache')
        else:
            ic('static cache')
        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )
        self.epoch_logit_tip = None
        self.epoch_logit_cluster = None
        self.epoch_pred_tip = None
        self.epoch_pred_cluster = None
        self.epoch_preds = None
        self.epoch_features = None
        self.flag = True
        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )
        

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_tip = torch.nn.ModuleList(
            [
                Accuracy(),
                Accuracy(),
                Accuracy()
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

        self.cache, _ = cache.build_cache(self.hparams.dataset, self.hparams.data_dir, 16, self.model.encoder, 'glove/glove.6B.300d.txt',
                                self.hparams.num_labeled_classes, self.hparams.num_unlabeled_classes,
                                save_path=self.hparams.save_cache, load_path=self.hparams.load_cache)
        if self.hparams.concat_combine:
            self.fusion = nn.Sequential(
                nn.Linear(2 * self.hparams.num_classes, 2 * self.hparams.num_classes),
                nn.ReLU(),
                nn.Linear(2 * self.hparams.num_classes, self.hparams.num_classes)
            )

    
    def configure_optimizers(self):
        if self.hparams.concat_combine:
                optimizer = torch.optim.SGD(
                list(self.model.parameters()) + list(self.fusion.parameters()),
                lr=self.hparams.base_lr,
                momentum=self.hparams.momentum_opt,
                weight_decay=self.hparams.weight_decay_opt,
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.base_lr,
                momentum=self.hparams.momentum_opt,
                weight_decay=self.hparams.weight_decay_opt,
            )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_large_crops):
            for other_view in np.delete(range(self.hparams.num_crops), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_large_crops * (self.hparams.num_crops - 1))

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet":
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
        else:
            views, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return views, labels, mask_lab

    def training_step(self, batch, _):
        views, labels, mask_lab = self.unpack_batch(batch)
        print(labels)
        nlc = self.hparams.num_labeled_classes
        self.model.normalize_prototypes()
        outputs = self.model(views)
        outputs_ = {key: value[0] for key, value in outputs.items()}


        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

        if self.hparams.oracle:
            tip = self.cache.oracle(labels).unsqueeze(0).expand(self.hparams.num_large_crops, -1, -1)
        else:
            tip = self.cache(outputs['feats'])
        logits = self.cache.combine(logits, tip.unsqueeze(1).expand((-1, self.hparams.num_heads, -1, -1)), self.hparams.tip_loss) # TODO: check if dimension matches

        

        # concat combine
        if self.hparams.concat_combine:
            logits = torch.cat([logits, tip.unsqueeze(1).expand((-1, self.hparams.num_heads, -1, -1))], dim=-1)
            logits = self.fusion(logits)

        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        print("logits.shape",logits.shape)
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)

        # loss for the memory bank
        loss_zsl = 0
        if self.hparams.zsl_loss > 0:
            targets_zsl = torch.zeros_like(tip)
            for v in range(self.hparams.num_large_crops):
                targets_zsl[v, mask_lab, :nlc] = targets_lab.type_as(targets_zsl)
                targets_zsl[v, ~mask_lab, nlc:] = self.sk(
                    tip[v, ~mask_lab, nlc:]
                ).type_as(targets_zsl)
            loss_zsl = self.swapped_prediction(tip, targets_zsl)


        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # mutual information loss
        loss_mi = 0
        n_labeled = mask_lab.sum()
        n_unlabeled = (~mask_lab).sum()
        n_min = min(n_labeled, n_unlabeled)


        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                lab_softmax = F.softmax(outputs['logits_lab'][v, h, mask_lab][:n_min] / self.hparams.temperature, dim=-1)
                unlab_softmax = F.softmax(outputs['logits_unlab'][v, h, ~mask_lab][:n_min] / self.hparams.temperature, dim=-1)
                loss_mi_i, _ = mutual_information_loss(lab_softmax, unlab_softmax)
                loss_mi += loss_mi_i
        loss_mi /= (self.hparams.num_large_crops * self.hparams.num_heads)

        # mutual information combine
        mi_combine = 0                                                                                                                               
        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                uno_softmax = F.softmax(logits[v, h] / self.hparams.temperature, dim=-1)
                tip_softmax = F.softmax(tip[v] / self.hparams.temperature, dim=-1)
                loss_mi_i, _ = mutual_information_loss(uno_softmax, tip_softmax)
                mi_combine += loss_mi_i
        mi_combine /= (self.hparams.num_large_crops * self.hparams.num_heads)

        # total loss
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2
        
        if self.hparams.mi_combine > 0:
            loss += self.hparams.mi_combine * mi_combine
        if self.hparams.mutual_information > 0:
            loss += self.hparams.mutual_information * loss_mi
        if self.hparams.zsl_loss > 0:
            loss += self.hparams.zsl_loss * loss_zsl 


        # log
        results = {
            "loss": loss.detach(),
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "mi_combine": mi_combine,
            "loss_zsl": loss_zsl
        }
        
        tip_ = tip[0]
        # logits_ = torch.cat([outputs_["logits_lab"], outputs_["logits_unlab"]], dim=-1)
        # logits_over_ = torch.cat([outputs_["logits_lab"], outputs_["logits_unlab_over"]], dim=-1)


        epoch_tip = tip_[~mask_lab,:].detach()

        # preds_logit = self.cache.combine(outputs_["logits_unlab"][0,~mask_lab],tip_[..., self.hparams.num_labeled_classes:], self.hparams.tip_loss)
        preds_logit = self.cache.combine(outputs_["logits_unlab"][:,~mask_lab,:], epoch_tip[..., self.hparams.num_labeled_classes:], self.hparams.tip_loss) # with len 20

        preds = preds_logit.max(dim=-1)[1]
        preds_tip = epoch_tip[...,self.hparams.num_labeled_classes:].max(dim=-1)[1] + self.hparams.num_labeled_classes
        epoch_preds =  labels[~mask_lab].detach()
        epoch_features = outputs_['feats'][~mask_lab,:].detach()

        if self.flag:
            self.epoch_logit_tip = epoch_tip
            self.epoch_logit_cluster = preds_logit
            self.epoch_pred_tip = preds_tip
            self.epoch_pred_cluster = preds
            self.epoch_preds = epoch_preds # TODO: check shape 
            self.epoch_features = epoch_features
            self.flag = False
        else:
            self.epoch_logit_tip = torch.cat((self.epoch_logit_tip, epoch_tip))
            self.epoch_logit_cluster = torch.cat((self.epoch_logit_cluster, preds_logit), dim = 1)
            self.epoch_pred_tip = torch.cat((self.epoch_pred_tip, preds_tip))
            self.epoch_pred_cluster = torch.cat((self.epoch_pred_cluster, preds), dim = 1)
            self.epoch_preds = torch.cat((self.epoch_preds, epoch_preds))
            self.epoch_features = torch.cat((self.epoch_features, epoch_features))

        if self.hparams.dynamic_cache:
            f = outputs['feats'][0, mask_lab].detach()
            t = labels[mask_lab] # may cause bug
            self.cache.update_memory(f, t)

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self,outputs):
        self.epoch_logit_tip = torch.softmax(self.epoch_logit_tip / self.hparams.temperature, dim = 1)
        self.flag = True
        if self.hparams.cluster_wise_ps and self.current_epoch > self.hparams.threshold:
            cluster_id = {}
            M = torch.zeros(self.hparams.num_unlabeled_classes,self.hparams.num_unlabeled_classes,dtype = torch.int32)
            for i in range(self.hparams.num_labeled_classes,self.hparams.num_labeled_classes + self.hparams.num_unlabeled_classes):
                select_indices = self.epoch_pred_tip == i
                if select_indices.sum() < 32:
                    continue
                maxnum, selected_top_indices = torch.topk(self.epoch_logit_tip[select_indices][:,i], self.hparams.cluster_top_k)
                selected_rows = self.epoch_logit_tip * select_indices.reshape(-1, 1) # maintain the original indice

                max_logits, indices = torch.topk(selected_rows[:,i], self.hparams.cluster_top_k) # select top-k confidence indices for each class   
                cid = self.epoch_pred_cluster[0][indices] # Here I wanna to find that these high_tip_conf features are sent to which cluster 
                # ic(cid.shape)
                # find the cluster id
                cluster_id[i] = cid # i-th class images were sent to cluster_id[i] #第x类的去了第y个簇
                for cluster in cid:
                    M[cluster][i - self.hparams.num_labeled_classes] += 1  # 更新矩阵 M
                # First, find all images corespondding to a cluster id
                # ic(cid) # from 1 to 19, a tensor with size(16, )
            # ic(cluster_id)
            # cluster_id: {80: tensor([15, 15, 15, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],                                     │·················································································································
            #                device='cuda:0'),                                                                                          │·················································································································
            #          81: tensor([15, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],                                     │·················································································································
            #                device='cuda:0'), ... # an example
            print("M:\n",M)
            all_label_proto = []
            assigned_classes = {}

            for i in range(M.shape[0]): 
                max_value = torch.max(M[i]) 
                if max_value > self.hparams.cluster_top_k / 2:
                    max_index = torch.argmax(M[i])  
                    assigned_classes[i] = max_index.item()
         
            # cluster_stats = calculate_cluster_accuracy(cluster_predictions, converted_epoch_ground_truth, assigned_classes)

            # for cluster, stats in cluster_stats.items():
            #     if cluster in assigned_classes:
            #         print(f"Cluster {cluster}: Total Samples = {stats['total']}, Correct Samples = {stats['correct']}, "
            #             f"Real Class Counts = {stats['real_class_counts']}, Accuracy = {stats['accuracy']:.2f}")



            for cluster_id, class_id in assigned_classes.items():
                cluster_indices = torch.nonzero(self.epoch_pred_cluster[0] == cluster_id)
                cluster_indices = cluster_indices.squeeze()
                selected_indices = torch.randperm(len(cluster_indices))[:16]
                selected_features = self.epoch_features[cluster_indices[selected_indices]]
                selected_targets = torch.full((16,), class_id + self.hparams.num_labeled_classes, dtype=torch.long)
                selected_targets = selected_targets.to('cuda')

                if not (class_id + self.hparams.num_labeled_classes == self.cache.targets).any():
                    self.cache.add_memory(selected_features, selected_targets)
                else:
                    self.cache.renew_memory(selected_features,selected_targets)



    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch #label is a tensor

        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self(images)
        if self.hparams.oracle:
            tip = self.cache.oracle(labels).unsqueeze(0).expand(self.hparams.num_large_crops, -1, -1)
        else:
            tip = self.cache(outputs['feats']) # tip is logits got from cache
        unlab_flag = False
        if "unlab" in tag:  # use clustering head
            unlab_flag = True
            preds_logit = self.cache.combine(outputs["logits_unlab"], tip[..., self.hparams.num_labeled_classes:], self.hparams.tip_loss) # with len 20
            # ic(preds.shape)
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
            
            preds_inc = self.cache.combine(preds_inc, tip, self.hparams.tip_loss)
            preds_tip = tip[..., self.hparams.num_labeled_classes:].max(dim=-1)[1] + self.hparams.num_labeled_classes
        else:  # use supervised classifier
            unlab_flag = False
            preds_logit = self.cache.combine(outputs["logits_lab"], tip[..., :self.hparams.num_labeled_classes], self.hparams.tip_loss)
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
            preds_inc = self.cache.combine(preds_inc, tip, self.hparams.tip_loss)
            preds_tip = tip[..., :self.hparams.num_labeled_classes].max(dim=-1)[1]
        # ic(preds_logit.shape)
        preds = preds_logit.max(dim=-1)[1] # which cluster id?
        # ic(preds.shape)
        preds_inc = preds_inc.max(dim=-1)[1]


        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)
        self.metrics_tip[dl_idx].update(preds_tip, labels)

    def validation_epoch_end(self, _):
     
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        results_tip = [m.compute() for m in self.metrics_tip]
        # log metrics
        for dl_idx, (result, result_inc, results_tip) in enumerate(zip(results, results_inc, results_tip)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            prefix_tip = 'tip/' + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
                    
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)
            self.log(prefix_tip + '/acc', results_tip)
        self.flag = True
        

def main(args):
    wandb.init(project=args.project, entity=args.entity, config=args.__dict__, 
               mode='offline' if args.offline else 'online', 
               settings=wandb.Settings(start_method='thread'))

    dm = get_datamodule(args, "discover")
    run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )
    model = Discoverer(**args.__dict__)
    model.hparams.cluster_wise_ps = True
    # model = Discoverer.load_from_checkpoint('/home/wangws/SNCD/logs/SNCDcheck/daztnx1f/checkpoints/epoch=251-step=24442.ckpt')
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_sanity_val_steps = 0
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)
