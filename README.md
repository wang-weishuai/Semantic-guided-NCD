# Semantic-guided-NCD

Official implementation for paper: Semantic-Guided Novel Category Discovery.

By Weishuai Wang, Ting Lei, Qingchao Chen and Yang Liu.

The paper has been accepted by IEEE/CVF Association for the Advancement of Artificial Intelligence (AAAI), 2024

[Paper](https://semantic-guided-ncd.github.io/img/SNCDpaper.pdf)  [Homepage](https://semantic-guided-ncd.github.io/)
## Introduction
The Novel Category Discovery problem aims to cluster an unlabeled set with the help of a labeled set consisting of disjoint but related classes. However, many real-world applications require recognition as well as clustering for novel categories. We propose a new setting named **Semantic-guided Novel Category Discovery (SNCD)**, which extends NCD to enable recognition by introducing semantic labels of the unlabeled categories which is easy and cheap to get in form of word vectors of category names, and we demonstrate the recognition task and the clustering task can benefit from each other and jointly optimize. We convert zero-shot recognition to a cross-modal retrieval task by constructing a dynamic multi-modal Memory Bank to project visual features to the label space. Besides, we adopt mutual information maximization to transfer information between two tasks. Experiments on multiple datasets demonstrate the effectiveness of our approach.


##  Installation
Our implementation is based on PyTorch and PyTorch Lightning. Logging is performed using Wandb. You can create an virtual environment as follows.
```conda create --name SNCD python=3.8
conda activate SNCD
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=XX.X -c pytorch
pip install pytorch-lightning==1.1.3 lightning-bolts==0.3.0 wandb scikit-learn
mkdir -p logs/wandb checkpoints
```
Select the appropriate `cudatoolkit` version according to your system. Optionally, you can also replace `pillow` with [`pillow-simd`](https://github.com/uploadcare/pillow-simd) (if your machine supports it) for faster data loading:

Above is all the same as [UNO](https://github.com/DonkeyShot21/UNO).Besides, you should also install
```
pip install icecream gensim matplotlib seaborn
```

## Commands

### Pretraining
Running pretraining on CIFAR10 (5 labeled classes):
 ```
python main_pretrain.py --dataset CIFAR10 --gpus 1  --precision 16 --max_epochs 200 --batch_size 256 --num_labeled_classes 5 --num_unlabeled_classes 5 --comment 5_5`
```
Running pretraining on CIFAR100-80 (80 labeled classes):
```
python main_pretrain.py --dataset CIFAR100 --gpus 1 --precision 16 --max_epochs 200 --batch_size 256 --num_labeled_classes 80 --num_unlabeled_classes 20 --comment 80_20
```
Running pretraining on CIFAR100-50 (50 labeled classes):
```
python main_pretrain.py --dataset CIFAR100 --gpus 1 --precision 16 --max_epochs 200 --batch_size 256 --num_labeled_classes 50 --num_unlabeled_classes 50 --comment 50_50
```

Running pretraining on ImageNet (882 labeled classes):

```
python main_pretrain.py --gpus 2 --num_workers 8 --distributed_backend ddp --sync_batchnorm --precision 16 --dataset ImageNet --data_dir PATH/TO/IMAGENET --max_epochs 100 --warmup_epochs 5 --batch_size 256 --num_labeled_classes 882 --num_unlabeled_classes 30 --comment 882_30
```
You can get a checkpoint like pretrain-resnet18-CIFAR100-20_20.cp, which will use in next step.
You can also get coresponding checkpoint in [Google Drive folder](https://drive.google.com/drive/folders/1lhoBhT3a--TyvdB2eL-mM2n6I4Kg_qrB?usp=sharing), choose the checkpoint you want to download, do right click and select `Get link > Copy link`. For instance, for CIFAR10 the link will look something like this:
```
https://drive.google.com/file/d/1Pa3qgHwK_1JkA-k492gAjWPM5AW76-rl/view?usp=sharing
```



### Discovering
Running discovery on CIFAR10 (5 labeled classes, 5 unlabeled classes):
```
python main_discover.py --dataset CIFAR10 --gpus 1 --max_epochs 500 --batch_size 512 --num_labeled_classes 5 --num_unlabeled_classes 5 --pretrained PATH/TO/CHECKPOINTS/pretrain-resnet18-CIFAR10-5_5.cp --num_heads 4 --precision 16 --data_dir PATH/TO/DATASETS --multicrop --overcluster_factor 5 --data_dir PATH/TO/CIFAR10 --mutual_information 0.1 --mi_combine 0.1 --comment 5_5_mi0.1_dynamic_cache --threshold 250 --cluster_top_k 32 --dynamic_cache
```
Running discovery on CIFAR100-20 (80 labeled classes, 20 unlabeled classes):
```
python main_discover.py --dataset CIFAR100 --gpus 1 --max_epochs 500 --batch_size 512 --num_labeled_classes 80 --num_unlabeled_classes 20 --pretrained PATH/TO/CHECKPOINTS/pretrain-resnet18-CIFAR100-80_20.cp --num_heads 4 --precision 16 --data_dir PATH/TO/DATASETS --multicrop --overcluster_factor 5 --data_dir PATH/TO/CIFAR100 --mutual_information 0.1 --mi_combine 0.1 --comment 80_20_mi0.1_dynamic_cache --threshold 250 --cluster_top_k 32 --dynamic_cache
```
Running discovery on CIFAR100-50 (50 labeled classes, 50 unlabeled classes):
```
python main_discover.py --dataset CIFAR100 --gpus 1 --max_epochs 500 --batch_size 512 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained PATH/TO/CHECKPOINTS/pretrain-resnet18-CIFAR100-50_50.cp --num_heads 4 --precision 16 --multicrop --overcluster_factor 5 --data_dir PATH/TO/CIFAR100 --mutual_information 0.1 --mi_combine 0.1 --comment 50_50_mi0.1_dynamic_cache --threshold 0 --cluster_top_k 32 --dynamic_cache
```

```
python main_discover.py --dataset ImageNet --gpus 2 --distributed_backend ddp --sync_batchnorm --precision 16 --data_dir PATH/TO/DATASETS --max_epochs 60 --base_lr 0.2 --warmup_epochs 5 --batch_size 256 --num_labeled_classes 882 --num_unlabeled_classes 30 --num_heads 4 --pretrained PATH/TO/CHECKPOINTS/pretrained_checkpoints/pretrain-resnet18-ImageNet.cp --imagenet_split A --comment 882_30-A --overcluster_factor 4 --multicrop --threshold 70 --cluster_top_k 32
```
Please note that you should download [glove.6B.300d](https://nlp.stanford.edu/projects/glove/) and put it at your own workspace.

## Logging
Logging is performed with [Wandb](https://wandb.ai/site). Please create an account and specify your `--entity YOUR_ENTITY` and `--project YOUR_PROJECT`. For debugging, or if you do not want all the perks of Wandb, you can disable logging by passing `--offline`.

## Result
All results can check in wandb. For task-aware setting, the classification result is in ```tip/unlab/test```, the clustering result is in ```unlab/test/acc.```
For task-agnostic setting, Lab and Unlab are seperately in ```incremental/unlab/test/acc``` and ```
incremental/lab/test/acc```.

| Metric       | CIFAR100-20 | CIFAR100-50 | CIFAR10
|--------------|:-----------:|:-----------:|:-----------:|
| Classfication|    57.7    |     23.4    |   40.1
| Clustering   |    93.1    |     62.2    | 94.8


