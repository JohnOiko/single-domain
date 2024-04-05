import torchvision
from src.utils import utils
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
from torch import nn
import torch
from src.models import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    device = utils.select_device(args.device_id, args.no_hw_accel)

    # Dataset setup
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_root = '../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)
    num_classes = len(mnist_train.classes)

    # Params
    batch_size = 1024
    epochs = 50
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.00001
    gradient_accumulation_steps = 1
    log_interval = 5

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if not args.no_hw_accel:
        cuda_kwargs = {'num_workers': 0,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_subset, val_subset = torch.utils.data.random_split(mnist_train, [50000, 10000], generator=torch.Generator().manual_seed(1))
    train_loader = DataLoader(dataset=train_subset, **train_kwargs)
    val_loader = DataLoader(dataset=val_subset, **train_kwargs)
    test_loader = DataLoader(mnist_test, **test_kwargs)

    # Load the model on the device
    model = models.resnet50(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss().to(device)
    ResNet50.train(model, device, train_loader, optimizer, loss_function, gradient_accumulation_steps, epochs, val_loader=val_loader)


if __name__ == '__main__':
    main()
