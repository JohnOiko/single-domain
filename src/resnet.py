import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.autograd import Variable

import itertools

from src.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = utils.select_device(args.device_id, args.no_hw_accel)

    # Dataset setup
    transform1 = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    dataset_root = '../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform1)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform1)

    transform2 = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=3), transforms.RandomAdjustSharpness(sharpness_factor=10, p=1), transforms.ToTensor()])

    usps_train = torchvision.datasets.USPS(root=dataset_root, train=True, download=True, transform=transform2)
    usps_test = torchvision.datasets.USPS(root=dataset_root, train=False, download=True, transform=transform2)

    transform3 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    svhn_train = torchvision.datasets.SVHN(root=dataset_root, split='train', download=True, transform=transform3)
    svhn_test = torchvision.datasets.SVHN(root=dataset_root, split='test', download=True, transform=transform3)

    # Params
    batch_size = 64
    epochs = 30
    learning_rate = 0.0001
    weight_decay = 0.0001
    gradient_accumulation_steps = 1
    valid_ratio = 0.2
    random_seed = 0

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if not args.no_hw_accel:
        cuda_kwargs = {'num_workers': 0}
        # 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    two_train_dataset = utils.TwoDataset(svhn_train, mnist_train)
    two_test_dataset = utils.TwoDataset(svhn_test, mnist_test)

    train_loader = DataLoader(two_train_dataset, **train_kwargs)
    test_loader = DataLoader(two_test_dataset, **test_kwargs)

    # Losses
    criterion_classification = nn.CrossEntropyLoss().to(device)

    # Models
    input_shape = (1, 64, 64)

    # Initialize generator and discriminator
    C = models.resnet18(num_classes=10).to(device)

    # Optimizers
    optimizer_C = torch.optim.RMSprop(C.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_source_acc = []
    train_target_acc = []
    train_avg_acc = []
    val_source_acc = []
    val_target_acc = []
    val_avg_acc = []

    gan_loss = []
    dis_loss = []

    for epoch in range(epochs):
        batches_num = len(train_loader)
        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        print(f'\nEpoch {epoch + 1}/{epochs}')
        pbar = tf.keras.utils.Progbar(batches_num)

        mnist_examples = {}
        usps_examples = {}

        for batch_idx, ((data_S, target_S), (data_T, target_T)) in enumerate(train_loader):

            data_S, target_S = data_S.to(device), target_S.to(device)
            data_T, target_T = data_T.to(device), target_T.to(device)

            # ------------------
            #  Train Classifier
            # ------------------

            C.train()

            optimizer_C.zero_grad()

            pred_S = C(data_S)

            # Classification loss
            loss_C = criterion_classification(pred_S, target_S)

            loss_C.backward()
            optimizer_C.step()

            _, predicted_S = torch.max(pred_S.data, 1)
            total_S += target_S.size(0)
            correct_S += (predicted_S == target_S).sum().item()

            pred_T = C(data_T)

            _, predicted_T = torch.max(pred_T.data, 1)
            total_T += target_T.size(0)
            correct_T += (predicted_T == target_T).sum().item()

            pbar.update(batch_idx, values=[("Train-Source acc", correct_S / total_S), ("Train-Target acc", correct_T / total_T), ("Train-Average accuracy", (correct_S + correct_T) / (total_S + total_T))])

        train_source_acc.append(correct_S / total_S)
        train_target_acc.append(correct_T / total_T)
        train_avg_acc.append((correct_S + correct_T) / (total_S + total_T))

        C.eval()

        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        with torch.no_grad():
            for batch_idx, ((data_S, target_S), (data_T, target_T)) in enumerate(test_loader):
                data_S, target_S = data_S.to(device), target_S.to(device)
                data_T, target_T = data_T.to(device), target_T.to(device)

                pred_S = C(data_S)
                pred_T = C(data_T)

                _, predicted_S = torch.max(pred_S.data, 1)
                total_S += target_S.size(0)
                correct_S += (predicted_S == target_S).sum().item()

                _, predicted_T = torch.max(pred_T.data, 1)
                total_T += target_T.size(0)
                correct_T += (predicted_T == target_T).sum().item()

        pbar.update(batches_num, values=[("Val-Source acc", correct_S / total_S), ("Val-Target acc", correct_T / total_T), ("Val-Average accuracy", (correct_S + correct_T) / (total_S + total_T))])

        val_source_acc.append(correct_S / total_S)
        val_target_acc.append(correct_T / total_T)
        val_avg_acc.append((correct_S + correct_T) / (total_S + total_T))

    np.savez('../dump/resnet_svhn_mnist_results.npz', train_source_acc=train_source_acc, train_target_acc=train_target_acc, train_avg_acc=train_avg_acc,
             val_source_acc=val_source_acc, val_target_acc=val_target_acc, val_avg_acc=val_avg_acc, gan_loss=gan_loss, dis_loss=dis_loss)
