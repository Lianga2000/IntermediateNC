import os
import sys
import re
import datetime
import operator
import numpy

from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

def corrupt_labels(dataset, corrupt_prob, num_classes):
    np.random.seed(12345)

    for i in range(len(dataset.targets)):#targets labels
        if np.random.rand(1) < corrupt_prob:
            dataset.targets[i] = np.random.randint(0, num_classes)


def get_network(settings):
    """ return given network
    """

    if settings.net == 'mlp':
        from models.mlp import mlp
        net = mlp(settings)
    elif settings.net == 'convnet':
        from models.convnet import convnet
        net = convnet(settings)
    elif settings.net == 'convresnet':
        from models.convresnet import convresnet
        net = convresnet(settings)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    net = net.to(settings.device)

    return net

def get_training_dataloader(dataset_name, mean, std, batch_size=16, num_workers=2,
                            shuffle=True, rnd_aug=True, corrupt_prob=0.0, num_classes=10):
    """ return training dataloader
    """

    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)

    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        padded_im_size = 32
        transform_train = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]
                                       )
        training_set = dataset(root='./data', train=True, download=True, transform=transform_train)

    elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':

        transformations = []

        if rnd_aug:
            transformations = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomRotation(15),
            ]
        transformations += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        transform_train = transforms.Compose(transformations)
        training_set = dataset(root='./data', train=True, download=True, transform=transform_train)


    elif dataset_name == 'STL10' or dataset_name == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        training_set = dataset(root='./data', split='train', download=True, transform=transform_train)

    corrupt_labels(training_set, corrupt_prob, num_classes)

    training_loader = DataLoader(training_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_test_dataloader(dataset_name, mean, std, batch_size=16, num_workers=2,
                        shuffle=True, corrupt_prob=False, num_classes=10):
    """ return test dataloader
    """
    dataset = operator.attrgetter(dataset_name)(torchvision.datasets)

    if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        im_size = 28
        padded_im_size = 32
        transform_test = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)]
                                       )
        test_set = dataset(root='./data', train=False, download=True, transform=transform_test)

    elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = dataset(root='./data', train=False, download=True, transform=transform_test)

    elif dataset_name == 'STL10' or dataset_name == 'SVHN':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_set = dataset(root='./data', split='test', download=True, transform=transform_test)

    corrupt_labels(test_set, corrupt_prob, num_classes)

    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def save_data(dir_name, graphs):
    ## save the results

    attrbts = [attr for attr in dir(graphs) if not \
        callable(getattr(graphs, attr)) and not attr.startswith("__")]

    for name in attrbts:

        if name in ['cdnv_train', 'cdnv_test',
                    'train_emb_acc_ncc', 'test_emb_acc_ncc',
                    'train_emb_acc', 'test_emb_acc',
                    'train_emb_loss', 'test_emb_loss', 'rank']:

            for i in range(len(operator.attrgetter(name)(graphs))):
                _ = open(dir_name + '/' + name + '_' + str(i) + '.txt', 'w+')
                _.write(str(operator.attrgetter(name)(graphs)[i]))
                _.close()
        else:
            _ = open(dir_name + '/' + name + ".txt", "w+")
            _.write(str(operator.attrgetter(name)(graphs)))
            _.close()


def get_dir_name(directory):

    if not os.path.isdir(directory):
        os.mkdir(directory)

    # results directories
    sub_dirs_ids = [int(dir) for dir in os.listdir(directory)
                    if os.path.isdir(directory + '/' + dir)]

    # experiment id
    xid = max(sub_dirs_ids)
    dir_name = directory + '/' + str(xid)

    # sweeps directories
    sub_dirs_ids = [int(dir) for dir in os.listdir(dir_name)
                    if os.path.isdir(dir_name + '/' + dir)]

    # current sweep
    if len(sub_dirs_ids) == 0: pid = 0
    else: pid = max(sub_dirs_ids) + 1
    dir_name += '/' + str(pid)
    os.mkdir(dir_name)

    return dir_name
