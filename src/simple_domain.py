import torchvision
from src.utils import utils
import argparse


def main():
    # Command line arguments setup
    parser = argparse.ArgumentParser(prog="Simple Domain")
    parser.add_argument('-d', '--device_id', type=int,
                        help='the id of the device to be used for hardware acceleration if available')
    parser.add_argument('-nhw', '--no_hw_accel', action="store_true",
                        help='disable hardware acceleration by running on the cpu')
    args = parser.parse_args()

    # Device creation based on the available hardware acceleration
    device = utils.get_device(args.device_id, args.no_hw_accel)

    # Dataset setup
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)


if __name__ == '__main__':
    main()
