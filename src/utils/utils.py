import os

import numpy as np
import pandas as pd
import scipy
import torch
import torchvision
from sklearn.preprocessing import LabelBinarizer

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


def save_torch_dataset(path, train_set=None, test_set=None, overwrite=False):
    if train_set is None and test_set is None:
        raise ValueError('either the train_set or test_set must be a torchvision dataset')

    if os.path.isfile(path + '.npz') and not overwrite:
        return save_dataset(path, overwrite=overwrite)

    train_inputs, train_labels, test_inputs, test_labels, target_names = None, None, None, None, None

    if train_set is not None:
        train_inputs = train_set.data.numpy()
        train_labels = train_set.targets.numpy()
        target_names = train_set.classes

    if test_set is not None:
        test_inputs = train_set.data.numpy()
        test_labels = train_set.targets.numpy()
        target_names = test_set.classes if target_names is None else target_names

    return save_dataset(path, train_inputs, train_labels, test_inputs, test_labels, target_names, overwrite=overwrite)


def save_dataset(path, train_inputs=None, train_labels=None, test_inputs=None, test_labels=None, target_names=None, overwrite=False):
    if os.path.isfile(path + '.npz') and not overwrite:
        return save_numpy(path, arrays=[], overwrite=overwrite)

    label_binarizer = LabelBinarizer()
    arrays = []
    names = []

    if train_inputs is not None:
        arrays.append(train_inputs)
        names.append('train_inputs')

    if train_labels is not None:
        arrays.append(label_binarizer.fit_transform(train_labels))
        arrays.append(train_labels)
        names.append('train_targets')
        names.append('train_labels')

    if test_inputs is not None:
        arrays.append(test_inputs)
        names.append('test_inputs')

    if test_labels is not None:
        arrays.extend([label_binarizer.fit_transform(test_labels), test_labels])
        names.extend(['test_targets', 'test_labels'])

    if target_names is not None:
        arrays.append(target_names)
        names.append('target_names')

    if len(names) == 0:
        raise ValueError('at least one of train_inputs, train_labels, test_inputs, test_labels '
                         'or target_names must be given')
    else:
        return save_numpy(path, arrays, names, single=False, overwrite=overwrite)


def save_numpy(path, arrays, names=None, single=False, overwrite=False):
    if names is None or single:
        names = []

    if not path.startswith('../data/'):
        path = '../data/' + path

    if os.path.isfile(path + ('.npy' if single else '.npz')) and not overwrite:
        return path + ('.npy' if single else '.npz'), np.load(path + '.npz', mmap_mode='r').files

    if not isinstance(names, list):
        raise TypeError('arrays must be a list')
    elif not isinstance(arrays, list):
        raise TypeError('names must be a list')
    elif len(arrays) < 1:
        raise ValueError('arrays must contain at least one element')
    elif single and len(arrays) != 1:
        raise ValueError('exactly one array must be passed as a list when the single option is selected')
    elif len(names) > 0 and len(names) != len(arrays):
        raise ValueError('there must be an equal number of names and arrays')

    for i in range(len(arrays)):
        if isinstance(arrays[i], (np.ndarray, scipy.sparse._csr.csr_matrix)):
            pass
        elif isinstance(arrays[i], list):
            arrays[i] = np.array(arrays[i])
        elif getattr(arrays[i], '__module__', None).startswith(torchvision.datasets.__name__):
            arrays[i] = arrays[i].data.numpy()
        elif isinstance(arrays[i], (pd.DataFrame, pd.Series, pd.Index)):
            arrays[i] = arrays[i].to_numpy()
        elif isinstance(arrays[i], str) and arrays[i].endswith('.csv'):
            arrays[i] = np.genfromtxt(arrays[i], dtype=None, delimiter=',', encoding='utf8')
        else:
            raise ValueError('arrays list must contain only numpy arrays, lists, torch tensors, pandas dataframes, '
                             'series, indices or paths to csv files that can be converted to numpy arrays')
        i += 1

    os.makedirs(path.rsplit("/", 1)[0] + '/', exist_ok=True)
    np.save(path, arrays[0]) if single else np.savez(path, **{name: array for name, array in zip(names, arrays)})
    return path + ('.npy' if single else '.npz'), names


# Fixes the error that appears when scipy is installed and a keyboard interrupt is created, causing python to hang.
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
