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
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7, padding=0),
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
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, 7, padding=0),
            nn.Tanh()
        ]

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
            # if normalize:
            #     layers.append(nn.InstanceNorm2d(out_filters))
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


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def plotter(real_targets, fake_sources, recont_targets):
    """Plot images in a grid."""
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):  # Assuming you have 3 images for each type
        axs[i, 0].imshow(real_targets[i], cmap='gray')
        axs[i, 0].set_title("Real Target")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(fake_sources[i], cmap='gray')
        axs[i, 1].set_title("Fake Source")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(recont_targets[i], cmap='gray')
        axs[i, 2].set_title("Reconstructed Target")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


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
    transform1 = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dataset_root = '../../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform1)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform1)

    transform2 = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    usps_train = torchvision.datasets.SVHN(root=dataset_root, split='train', download=True, transform=transform2)
    usps_test = torchvision.datasets.SVHN(root=dataset_root, split='test', download=True, transform=transform2)

    # Params
    batch_size = 64
    epochs = 200
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

    # mnist_train_loader = DataLoader(mnist_train, **train_kwargs)
    # mnist_test_loader = DataLoader(mnist_test, **test_kwargs)
    #
    # usps_train_loader = DataLoader(usps_train, **train_kwargs)
    # usps_test_loader = DataLoader(usps_test, **test_kwargs)

    two_train_dataset = utils.TwoDataset(mnist_train, usps_train)
    two_test_dataset = utils.TwoDataset(mnist_test, usps_test)

    train_loader = DataLoader(two_train_dataset, **train_kwargs)
    test_loader = DataLoader(two_test_dataset, **test_kwargs)

    # Losses
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_cycle = torch.nn.L1Loss().to(device)
    criterion_identity = torch.nn.L1Loss().to(device)
    criterion_classification = nn.CrossEntropyLoss().to(device)

    # Models
    input_shape = (1, 64, 64)

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

    #Tensor = torch.cuda.FloatTensor  # This, probably, will not work with DirectML

    fake_S_buffer = ReplayBuffer()

    for epoch in range(epochs):
        batches_num = len(train_loader)
        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        print(f'\nEpoch {epoch + 1}/{epochs}')
        pbar = tf.keras.utils.Progbar(batches_num)

        for batch_idx, ((data_S, target_S), (data_T, target_T)) in enumerate(train_loader):
            if batch_idx == 0:
                fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                randinx = np.random.choice(data_S.size(0), 4, replace=False)
                axes[0, 0].imshow(data_S[randinx[0]].cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')
                axes[0, 1].imshow(data_S[randinx[1]].cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')
                axes[1, 0].imshow(data_T[randinx[2]].cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')
                axes[1, 1].imshow(data_T[randinx[3]].cpu().detach().numpy().transpose((1, 2, 0)), cmap='gray')
                plt.show()

            data_S, target_S = data_S.to(device), target_S.to(device)
            data_T, target_T = data_T.to(device), target_T.to(device)

            # Adversarial ground truths
            valid = Variable(torch.from_numpy(np.ones((data_S.size(0), *D_S.output_shape))).float().to(device), requires_grad=False)
            fake = Variable(torch.from_numpy(np.zeros((data_T.size(0), *D_S.output_shape))).float().to(device), requires_grad=False)

            # ------------------
            #  Train Classifier
            # ------------------

            C.train()

            optimizer_C.zero_grad()

            pred_S = C(data_S.expand(-1, 3, -1, -1))

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

            # Identity loss
            # indentiy_S = G_TS(data_S)
            # loss_identity = criterion_identity(indentiy_S, data_S)

            # GAN loss
            fake_S = G_TS(data_T)
            loss_GAN_TS = criterion_GAN(D_S(fake_S), valid)

            # Cycle loss
            # reconstruct_T = G_ST(fake_S)
            # loss_cycle_T = criterion_cycle(reconstruct_T, data_T)

            total_G_loss = 10 * loss_GAN_TS

            total_G_loss.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            D_S.train()

            optimizer_D_S.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_S(data_S), valid)
            # Fake loss (on batch of previously generated samples)
            fake_S_ = fake_S_buffer.push_and_pop(fake_S)
            loss_fake = criterion_GAN(D_S(fake_S_.detach()), fake)
            # Total loss
            total_D_loss = loss_real + loss_fake

            total_D_loss.backward()
            optimizer_D_S.step()

            _, predicted_S = torch.max(pred_S.data, 1)
            total_S += target_S.size(0)
            correct_S += (predicted_S == target_S).sum().item()

            pred_T = C(fake_S.expand(-1, 3, -1, -1))

            _, predicted_T = torch.max(pred_T.data, 1)
            total_T += target_T.size(0)
            correct_T += (predicted_T == target_T).sum().item()

            pbar.update(batch_idx, values=[("G loss", total_G_loss.item()), ("D loss", total_D_loss.item()), ("Train-Source acc", correct_S / total_S), ("Train-Target acc", correct_T / total_T), ("Train-Average accuracy", (correct_S + correct_T) / (total_S + total_T))])

        C.eval()
        G_TS.eval()

        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        with torch.no_grad():
            for batch_idx, ((data_S, target_S), (data_T, target_T)) in enumerate(test_loader):
                data_S, target_S = data_S.to(device), target_S.to(device)
                data_T, target_T = data_T.to(device), target_T.to(device)

                pred_S = C(data_S.expand(-1, 3, -1, -1))
                fake_S = G_TS(data_T)
                pred_T = C(fake_S.expand(-1, 3, -1, -1))
                recont_T = G_ST(fake_S)

                _, predicted_S = torch.max(pred_S.data, 1)
                total_S += target_S.size(0)
                correct_S += (predicted_S == target_S).sum().item()

                _, predicted_T = torch.max(pred_T.data, 1)
                total_T += target_T.size(0)
                correct_T += (predicted_T == target_T).sum().item()

        pbar.update(batches_num, values=[("Val-Source acc", correct_S / total_S), ("Val-Target acc", correct_T / total_T), ("Val-Average accuracy", (correct_S + correct_T) / (total_S + total_T))])

        lr_scheduler_G.step()
        lr_scheduler_D_S.step()

        rnd_idx = np.random.choice(data_T.size(0), 3, replace=False)
        real_target = np.clip(data_T[rnd_idx].cpu().detach().numpy().transpose((0, 2, 3, 1)), 0, 1)
        fake_source = np.clip(fake_S[rnd_idx].cpu().detach().numpy().transpose((0, 2, 3, 1)), 0, 1)
        recont_target = np.clip(recont_T[rnd_idx].cpu().detach().numpy().transpose((0, 2, 3, 1)), 0, 1)
        plotter(real_target, fake_source, recont_target)
