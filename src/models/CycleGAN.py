import argparse

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


##############################
# Code provided by PyTorch-GAN
# https://github.com/eriklindernoren/PyTorch-GAN?tab=readme-ov-file#auxiliary-classifier-gan
##############################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Down-sampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Up-sampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns down-sampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


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
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset_root = '../../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    usps_train = torchvision.datasets.USPS(root=dataset_root, train=True, download=True, transform=transform)
    usps_test = torchvision.datasets.USPS(root=dataset_root, train=False, download=True, transform=transform)

    # Params
    batch_size = 512
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 0.0001
    gradient_accumulation_steps = 1
    valid_ratio = 0.2
    random_seed = 0

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if not args.no_hw_accel:
        cuda_kwargs = {'num_workers': 0,
                       'drop_last': True,  # I can't bother with the different datasets sizes
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    mnist_train_loader = DataLoader(mnist_train, **train_kwargs)
    mnist_test_loader = DataLoader(mnist_test, **test_kwargs)

    usps_train_loader = DataLoader(usps_train, **train_kwargs)
    usps_test_loader = DataLoader(usps_test, **test_kwargs)

    # Losses
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_cycle = torch.nn.L1Loss().to(device)
    criterion_identity = torch.nn.L1Loss().to(device)
    criterion_classification = nn.CrossEntropyLoss().to(device)

    # Models
    input_shape = (3, 32, 32)

    # Initialize generator and discriminator
    G_TS = GeneratorResNet(input_shape, 3).to(device)
    G_ST = GeneratorResNet(input_shape, 3).to(device)
    D_S = Discriminator(input_shape).to(device)
    C = models.resnet18(num_classes=10).to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_TS.parameters(), G_ST.parameters()), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D_S = torch.optim.Adam(D_S.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_C = torch.optim.RMSprop(C.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(epochs, 0, 100).step
    )
    lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_S, lr_lambda=LambdaLR(epochs, 0, 100).step
    )

    Tensor = torch.cuda.FloatTensor

    for epoch in range(epochs):
        batches_num = len(usps_train_loader)
        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        print(f'\nEpoch {epoch + 1}/{epochs}')
        pbar = tf.keras.utils.Progbar(batches_num)

        # TODO: Change this training loop (problem with unequal dataset sizes)
        for batch_idx, ((data_S, target_S), (data_T, target_T)) in enumerate(zip(mnist_train_loader, usps_train_loader)):
            data_S, target_S = data_S.to(device), target_S.to(device)
            data_T, target_T = data_T.to(device), target_T.to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((data_S.size(0), *D_S.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((data_T.size(0), *D_S.output_shape))), requires_grad=False)

            # ------------------
            #  Train Classifier
            # ------------------

            C.train()

            optimizer_C.zero_grad()

            pred_S = C(data_S)
            pred_T = C(data_T)

            # Classification loss
            loss_C = criterion_classification(pred_S, target_S)

            loss_C.backward()
            optimizer_C.step()

            # ------------------
            #  Train Generators
            # ------------------

            G_TS.train()
            G_ST.train()

            optimizer_G.zero_grad()

            # GAN loss
            fake_S = G_TS(data_S)
            loss_GAN_ST = criterion_GAN(D_S(fake_S), valid)

            # Cycle loss
            reconstruct_T = G_ST(fake_S)
            loss_cycle_A = criterion_cycle(reconstruct_T, data_T)

            total_loss = loss_GAN_ST + loss_cycle_A

            total_loss.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            # D_S.train()

            optimizer_D_S.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_S(data_S), valid)
            # Fake loss (on batch of previously generated samples)
            loss_fake = criterion_GAN(D_S(fake_S.detach()), fake)
            # Total loss
            loss_D_S = loss_real + loss_fake

            loss_D_S.backward()
            optimizer_D_S.step()

            _, predicted_S = torch.max(pred_S.data, 1)
            total_S += target_S.size(0)
            correct_S += (predicted_S == target_S).sum().item()

            _, predicted_T = torch.max(pred_T.data, 1)
            total_T += target_T.size(0)
            correct_T += (predicted_T == target_T).sum().item()

            pbar.update(batch_idx, values=[("loss", total_loss.item()), ("Source acc", correct_S / total_S), ("Target acc", correct_T / total_T), ("Average accuracy", (correct_S + correct_T) / (total_S + total_T))])
