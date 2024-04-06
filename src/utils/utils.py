import torch
try:
    import torch_directml
    available_directml = True
except ImportError:
    available_directml = False


# Automatically selects and returns the best available device based on the available hardware acceleration options.
def select_device(device_id=None, no_hw_accel=False):
    if torch.cuda.is_available() and not no_hw_accel:
        device = torch.device("cuda") if device_id is None else torch.device(f'cuda:{device_id}')
        print(f'PyTorch is running on CUDA device {torch.cuda.current_device()}\n')
    elif available_directml and torch_directml.is_available() and not no_hw_accel:
        device = torch_directml.device(device_id)
        print(f'PyTorch is running on DirectML device {torch_directml.default_device()}\n')
    else:
        device = torch.device("cpu")
        print(f'PyTorch is running on CPU\n')

    return device


# Fixes the error that appears when scipy is installed, and a keyboard interrupt is created, causing python to hang.
# Solution found here and updated to python 3: https://stackoverflow.com/a/15472811
def fix_keyboard_interrupts():
    import os
    import ctypes
    import _thread
    import win32api
    import importlib

    # Load the DLL manually to ensure its handler gets set before our handler.
    basepath = importlib.machinery.PathFinder().find_spec('numpy').submodule_search_locations[0]
    ctypes.CDLL(os.path.join(basepath, '..\..\..\Library\\bin', 'libmmd.dll'))
    ctypes.CDLL(os.path.join(basepath, '..\..\..\Library\\bin', 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event types will chain to the next handler.
    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        # CTRL_C_EVENT
        if dwCtrlType == 0:
            hook_sigint()
            # don't chain to the next handler
            return 1

        # chain to the next handler
        return 0

    win32api.SetConsoleCtrlHandler(handler, 1)
