import torch
import torch.nn as nn
import numpy as np
import time
import argparse

import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18
from src.utils import utils

def multitask_runner(source_domain: tuple, target_domain: tuple, name: str, save_model=False):

    res = resnet18()
    encoder = torch.nn.Sequential(*(list(res.children())[:-1])).to(device)


    class TaskHead(nn.Module):
        def __init__(self, input_size, classes):
            super(TaskHead, self).__init__()
            self.fc = nn.Linear(input_size, classes)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


    flip_head = TaskHead(512, 1).to(device)
    rotation_head = TaskHead(512, 4).to(device)
    location_head = TaskHead(512, 5).to(device)
    classification_head = TaskHead(512, 10).to(device)

    encoder.train()
    flip_head.train()
    rotation_head.train()
    location_head.train()
    classification_head.train()

    optimizer = torch.optim.Adam(
        list(flip_head.parameters()) + list(rotation_head.parameters()) + list(location_head.parameters()) + list(
            classification_head.parameters()) + list(encoder.parameters()))

    criterion_flip = nn.BCEWithLogitsLoss()
    criterion_rot = nn.CrossEntropyLoss()
    criterion_loc = nn.CrossEntropyLoss()
    criterion_classification = nn.CrossEntropyLoss()

    epochs = 20
    train_source_acc = []
    train_target_acc = []
    train_avg_acc = []
    val_source_acc = []
    val_target_acc = []
    val_avg_acc = []

    for epoch in range(epochs):
        start = time.time()

        loss = 0
        correct_S = 0
        total_S = 0
        for batch_idx, (data, target) in enumerate(tqdm(source_domain[0])):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            prediction = classification_head(encoder(data))
            classification_loss = criterion_classification(prediction, target)

            _, predicted = torch.max(prediction.data, 1)
            correct_S += (predicted == target).sum().item()
            total_S += target.size(0)

            # Flip Task
            flip_flag = float(np.random.randint(0, 2))
            flipped_data = transforms.RandomHorizontalFlip(flip_flag)(data)
            flip_prediction = flip_head(encoder(flipped_data))
            flip_loss = criterion_flip(flip_prediction, torch.full((len(target), 1), flip_flag).to(device))

            # Rotation Task
            rotation_angle = np.random.randint(0, 4)
            rotated_data = transforms.functional.rotate(data, angle=rotation_angle * 90)
            rotation_prediction = rotation_head(encoder(rotated_data.to(device)))
            rotation_loss = criterion_rot(rotation_prediction, torch.full((len(target),), rotation_angle).to(device))

            # Location task
            crops = transforms.functional.five_crop(data, (32, 32))
            location_loss = 0
            for i in range(5):
                location_prediction = location_head(encoder(crops[i]))
                location_loss += criterion_loc(location_prediction, torch.full((len(target),), i).to(device))

            classification_loss.backward()
            flip_loss.backward()
            rotation_loss.backward()
            location_loss.backward()

            optimizer.step()

            loss += classification_loss.item()

        correct_T = 0
        total_T = 0
        for batch_idx, (data, target) in enumerate(tqdm(target_domain[0])):
            data, target = data.to(device), target.to(device)

            prediction = classification_head(encoder(data))
            _, predicted = torch.max(prediction.data, 1)
            correct_T += (predicted == target).sum().item()
            total_T += target.size(0)

            optimizer.zero_grad()
            # Flip Task
            flip_flag = float(np.random.randint(0, 2))
            flipped_data = transforms.RandomHorizontalFlip(flip_flag)(data)
            flip_prediction = flip_head(encoder(flipped_data))
            flip_loss = criterion_flip(flip_prediction, torch.full((len(target), 1), flip_flag).to(device))

            # Rotation Task
            rotation_angle = np.random.randint(0, 4)
            rotated_data = transforms.functional.rotate(data, angle=rotation_angle * 90)
            rotation_prediction = rotation_head(encoder(rotated_data.to(device)))
            rotation_loss = criterion_rot(rotation_prediction, torch.full((len(target),), rotation_angle).to(device))

            # Location task
            crops = transforms.functional.five_crop(data, (64, 64))
            location_loss = 0
            for i in range(5):
                location_prediction = location_head(encoder(crops[i]))
                location_loss += criterion_loc(location_prediction, torch.full((len(target),), i).to(device))

            flip_loss.backward()
            rotation_loss.backward()
            location_loss.backward()

            optimizer.step()

        train_source_acc.append(correct_S / total_S)
        train_target_acc.append(correct_T / total_T)
        train_avg_acc.append((correct_S + correct_T) / (total_S + total_T))

        correct_S = 0
        total_S = 0
        correct_T = 0
        total_T = 0
        with torch.no_grad():
            for (images, labels) in source_domain[1]:
                images, labels = images.to(device), labels.to(device)
                outputs = classification_head(encoder(images))
                _, predicted = torch.max(outputs.data, 1)
                correct_S += (predicted == labels).sum().item()
                total_S += labels.size(0)


            for (images, labels) in target_domain[1]:
                images, labels = images.to(device), labels.to(device)
                outputs = classification_head(encoder(images))
                _, predicted = torch.max(outputs.data, 1)
                correct_T += (predicted == labels).sum().item()
                total_T += labels.size(0)

        val_source_acc.append(correct_S / total_S)
        val_target_acc.append(correct_T / total_T)
        val_avg_acc.append((correct_S + correct_T) / (total_S + total_T))

        print(
            f'Epoch {epoch + 1}/{epochs} classification loss: {loss} | train_source_acc: {train_source_acc[-1]}, train_target_acc: {train_target_acc[-1]}, train_avg_acc: {train_avg_acc[-1]}, '
            f' val_source_acc: {val_source_acc[-1]}, val_target_acc: {val_target_acc[-1]}, val_avg_acc: {val_avg_acc[-1]} | time elapsed: {time.time() - start}')
        if save_model:
            np.savez(f'../dump/{name}', train_source_acc=train_source_acc, train_target_acc=train_target_acc,
                     train_avg_acc=train_avg_acc,
                     val_source_acc=val_source_acc, val_target_acc=val_target_acc, val_avg_acc=val_avg_acc)


def parse_args():
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    return parser.parse_args()


if __name__ == '__main__':
    utils.fix_keyboard_interrupts()
    args = parse_args()
    device = utils.select_device(args.device_id, args.no_hw_accel)

    transform1 = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    dataset_root = '../../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform1)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform1)

    transform2 = transforms.Compose([transforms.Resize((64, 64)), transforms.Grayscale(num_output_channels=3),
                                     transforms.RandomAdjustSharpness(sharpness_factor=10, p=1), transforms.ToTensor()])

    usps_train = torchvision.datasets.USPS(root=dataset_root, train=True, download=True, transform=transform2)
    usps_test = torchvision.datasets.USPS(root=dataset_root, train=False, download=True, transform=transform2)

    source_domain = (DataLoader(mnist_train, batch_size=64, shuffle=True), DataLoader(mnist_test, batch_size=64, shuffle=True))
    target_domain = (DataLoader(usps_train, batch_size=64, shuffle=True), DataLoader(usps_test, batch_size=64, shuffle=True))

    multitask_runner(source_domain, target_domain, "resnet_multitask_mnist_to_usps", save_model=True)

