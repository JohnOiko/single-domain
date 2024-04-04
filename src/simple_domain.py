import torchvision
from src.utils import utils
from torchvision import models, transforms
from torch.utils.data import DataLoader
import argparse
import time
from torch import nn
import torch
from torch.autograd import profiler


def main():
    # Command line arguments setup
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    args = parser.parse_args()

    # Device creation based on the available hardware acceleration
    device = utils.select_device(args.device_id, args.no_hw_accel)

    # Dataset setup
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_root = '../data'
    mnist_train = torchvision.datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform)
    num_classes = len(mnist_train.classes)


    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.00001
    trace = False

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if not args.no_hw_accel:
        cuda_kwargs = {'num_workers': 0,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(mnist_train, **train_kwargs)
    test_loader = DataLoader(mnist_test, **test_kwargs)

    print(train_loader)

    model = models.resnet50(num_classes=num_classes).to(device)
    ci_train = False



    # Load the model on the device
    start = time.time()

    print('Finished moving {} to device: {} in {}s.'.format('ResNet50', device, time.time() - start))

    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    highest_accuracy = 0

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        size = len(train_loader.dataset)

        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        optimize_after_batches = 1
        start = time.time()
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            if (trace):
                with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
                    with profiler.record_function("model_inference"):
                        # Compute loss and perform backpropagation
                        pred = model(X)

                        batch_loss = cross_entropy_loss(pred, y)
                        batch_loss.backward()

                        if batch % optimize_after_batches == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
                break
            else:
                # Compute loss and perform backpropagation
                pred = model(X)
                batch_loss = cross_entropy_loss(model(X), y)
                batch_loss.backward()

                if batch % optimize_after_batches == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if (batch + 1) % 100 == 0:
                batch_loss_cpu, current = batch_loss.to('cpu'), (batch + 1) * len(X)
                print(f"loss: {batch_loss_cpu.item():>7f}  [{current:>5d}/{size:>5d}] in {time.time() - start:>5f}s")
                start = time.time()

            if ci_train:
                print(f"train [{len(X):>5d}/{size:>5d}] in {time.time() - start:>5f}s")
                break

    print("Done! with highest_accuracy: ", highest_accuracy)









if __name__ == '__main__':
    main()
