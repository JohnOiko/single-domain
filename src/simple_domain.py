import argparse

import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from src.models import ResNet50
from src.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    return parser.parse_args()


def main():
    utils.fix_keyboard_interrupts()
    args = parse_args()
    device = utils.select_device(args.device_id, args.no_hw_accel)

    # Dataset setup
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset_root = '../data'
    mnist_train_remote = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True)
    mnist_test_remote = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True)
    num_classes = len(mnist_train_remote.classes)

    # Params
    batch_size = 2024
    epochs = 10
    learning_rate = 0.0001
    weight_decay = 0.0001
    gradient_accumulation_steps = 1
    valid_ratio = 0.2
    random_seed = 0

    path, keys = utils.save_torch_dataset('domain/digits/mnist', mnist_train_remote, mnist_test_remote, overwrite=True)
    mnist_train = utils.NumpyDataset(path, train=True, transform=transform)

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if not args.no_hw_accel:
        cuda_kwargs = {'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_idx, valid_idx, _, _ = train_test_split(range(len(mnist_train)), mnist_train.targets,
                                                         stratify=mnist_train.targets, test_size=valid_ratio,
                                                         random_state=random_seed)

    train_loader = DataLoader(dataset=Subset(mnist_train, train_idx), **train_kwargs)
    val_loader = DataLoader(dataset=Subset(mnist_train, valid_idx), **train_kwargs)
    test_loader = DataLoader(mnist_test_remote, **test_kwargs)\

    # Load the model on the device
    model = models.resnet50(num_classes=num_classes).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss().to(device)
    ResNet50.train(model, device, train_loader, optimizer, loss_function, gradient_accumulation_steps, epochs,
                   val_loader=val_loader)


if __name__ == '__main__':
    main()
