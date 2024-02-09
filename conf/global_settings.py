""" configurations for this project
"""
import os
from datetime import datetime
import math
import torch.nn as nn

dataset_name = 'MNIST' # MNIST, FashionMNIST, CIFAR10

if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
    num_output_classes = 10
    num_input_channels = 1
    mean = 0.1307
    std = 0.3081

elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
    num_output_classes = int(dataset_name[5:])
    num_input_channels = 3
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# device
device = 'cuda'

# runs
EPOCH = 500
SAVE_EPOCH = 10

# hyperparameters
MILESTONES = [60, 100, 200]
batch_size = 128
test_batch_size = 128
lr = 0.1
top_lr = 0.1
warm = 1
tolerance = 0.001
corrupt_prob = 0.0
weight_decay = 5e-4
rnd_aug = True
loss = 'CE' # 'CE', 'MSE'
bn = True

# architecture
net = 'mlp'
top_layers_type = 'fc'
depth = 5
width = 5
alpha = 1
activation = nn.ReLU()
blockType = "ConvBlock"

# saving params
SAVE_EPOCH = 10
directory = './results_new/'
resume = False
save_nc = True
