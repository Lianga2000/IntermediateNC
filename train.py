import time
from shutil import copyfile

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import importlib
import utils
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, save_data
import analysis
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Graphs:
    def __init__(self):
        self.train_accuracy = []
        self.test_accuracy = []

        self.train_loss = []
        self.test_loss = []

        # NC1
        self.cdnv_train = []
        self.cdnv_test = []

        # Emb performance

        self.train_emb_acc_ncc = []
        self.test_emb_acc_ncc = []

        self.train_emb_accuracy = []
        self.test_emb_accuracy = []

        self.train_emb_loss = []
        self.test_emb_loss = []


        self.num_embs = 0
        self.num_matrices = 0

    def update_num_embs(self, num_embs, num_matrices):

        self.num_embs = num_embs
        self.num_matrices = num_matrices

        self.cdnv_train = [[] for _ in range(num_embs)]
        self.cdnv_test = [[] for _ in range(num_embs)]

        self.train_emb_acc_ncc = [[] for _ in range(num_embs)]
        self.test_emb_acc_ncc = [[] for _ in range(num_embs)]

        self.train_emb_acc = [[] for _ in range(num_embs)]
        self.test_emb_acc = [[] for _ in range(num_embs)]

        self.train_emb_loss = [[] for _ in range(num_embs)]
        self.test_emb_loss = [[] for _ in range(num_embs)]


    def add_data(self, train_acc, test_acc, train_loss, test_loss,
                        cdnv_train, cdnv_test,
                        train_emb_acc_ncc, test_emb_acc_ncc,
                        train_emb_acc, test_emb_acc,
                        train_emb_loss, test_emb_loss):

        self.train_accuracy += [train_acc]
        self.test_accuracy += [test_acc]
        self.train_loss += [train_loss]
        self.test_loss += [test_loss]

        for i in range(self.num_embs):

            if cdnv_train != None: self.cdnv_train[i] += [cdnv_train[i]]
            if cdnv_test != None: self.cdnv_test[i] += [cdnv_test[i]]

            if train_emb_acc_ncc != None: self.train_emb_acc_ncc[i] += [train_emb_acc_ncc[i]]
            if test_emb_acc_ncc != None: self.test_emb_acc_ncc[i] += [test_emb_acc_ncc[i]]

            if train_emb_acc != None: self.train_emb_acc[i] += [train_emb_acc[i]]
            if test_emb_acc != None: self.test_emb_acc[i] += [test_emb_acc[i]]

            if train_emb_loss != None: self.train_emb_loss[i] += [train_emb_loss[i]]
            if test_emb_loss != None: self.test_emb_loss[i] += [test_emb_loss[i]]


def train_epoch(epoch, net, optimizer, train_loader, settings):

    start = time.time()
    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):

        if settings.device == 'cuda':
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs, _ = net(images)

        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())


        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * settings.batch_size + len(images),
            total_samples=len(train_loader.dataset)
        ))

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch, net, test_loader, settings):

    start = time.time()
    if settings.loss == 'CE':
        loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE':
        loss_function = nn.MSELoss()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if settings.device == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        outputs, _ = net(images)
        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    acc = correct / dataset_size
    test_loss = test_loss / dataset_size

    finish = time.time()
    if settings.device == 'cuda':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss,
        acc,
        finish - start
    ))
    print()

    return acc, test_loss


def main(): #(config=None):
    # Initialize a new wandb run
    # with wandb.init(config=config):

    ## get directory name
    directory = './results_new'
    dir_name = utils.get_dir_name(directory)

    ## save hyperparams
    copyfile('./conf/global_settings.py', dir_name + '/global_settings.py')

    spec = importlib.util.spec_from_file_location("module", dir_name + '/global_settings.py')
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    ## MAKE IMPORT OF settings FROM THIS NEW FILE

    net = get_network(settings)

    graphs = Graphs()
    attrbts = [attr for attr in dir(graphs) if not \
        callable(getattr(graphs, attr)) and not attr.startswith("__")]

    # data preprocessing:
    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.batch_size,
        shuffle=True,
        rnd_aug=settings.rnd_aug,
        corrupt_prob=settings.corrupt_prob,
        num_classes=settings.num_output_classes
    )

    test_loader = get_test_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.test_batch_size,
        shuffle=True,
        corrupt_prob=0.0,
        num_classes=settings.num_output_classes
    )

    num_embs = net.num_embs
    num_matrices = net.num_matrices
    graphs.update_num_embs(num_embs, num_matrices)

    optimizer = optim.SGD(net.parameters(), lr=settings.lr, momentum=0.9, weight_decay=settings.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)  # learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings.warm)

    train_emb_acc_ncc = test_emb_acc_ncc = None
    cdnv_train = cdnv_test = None
    train_emb_acc, train_emb_loss, test_emb_acc, test_emb_loss = \
        None, None, None, None

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > settings.warm:
            train_scheduler.step(epoch)

        if epoch % 10 == 1:

            train_acc, train_loss = eval_training(epoch, net, train_loader, settings)
            test_acc, test_loss = eval_training(epoch, net, test_loader, settings)

            if settings.save_nc:
                cdnv_train = analysis.analysis(net, settings, train_loader)
                cdnv_test = analysis.analysis(net, settings, test_loader)

                train_emb_acc_ncc, test_emb_acc_ncc = \
                    analysis.embedding_performance_nearest_mean_classifier(net, settings, \
                                                                           train_loader, test_loader)

                train_emb_acc, train_emb_loss, test_emb_acc, test_emb_loss = \
                    None, None, None, None
                    #analysis.embedding_performance(net, settings, train_loader, test_loader)

            graphs.add_data(train_acc, test_acc, train_loss, test_loss,
                           cdnv_train, cdnv_test,
                           train_emb_acc_ncc, test_emb_acc_ncc,
                           train_emb_acc, test_emb_acc,
                           train_emb_loss, test_emb_loss)

            save_data(dir_name, graphs)

        train_epoch(epoch, net, optimizer, train_loader, settings)


if __name__ == '__main__':
    main()
