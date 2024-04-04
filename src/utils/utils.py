import torch
import torch_directml


def select_device(device_id=None, no_hw_accel=False):
    if torch.cuda.is_available() and not no_hw_accel:
        device = torch.device("cuda") if device_id is None else torch.device(f'cuda:{device_id}')
        print(f'PyTorch is running on CUDA device {torch.cuda.current_device()}')
    elif torch_directml.is_available() and not no_hw_accel:
        device = torch_directml.device(device_id)
        print(f'PyTorch is running on DirectML device {torch_directml.default_device()}')
    else:
        device = torch.device("cpu")
        print(f'PyTorch is running on CPU')

    return device
