import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import numpy as np

def split(X, y):
    r"""Splits X and y arrays into training and testing sets

    Args:
        X: Array of features.
        y: Array of label indices.

    """
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y))
    return X_train, X_test, y_train, y_test

def split_convert(X, y, types=('float', 'long')):
    r"""Splits a feature array and a label array to training and test tensors of types t[0] and t[1]

    """
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = pair_to_tensor(X_train, y_train, types)    
    X_test, y_test = pair_to_tensor(X_test, y_test, types)    
    return X_train, X_test, y_train, y_test

def generate_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    r"""Generates dataloaders for training and validation using split datasets.

    Args:
        X_train: Array of features from the training set.
        X_test: Array of features from the testing set.
        y_train: Array of label indices from the training set.
        y_test: Array of label indices from the testing set.
        batch_size: Size of the output batches.
    """
    train = data_utils.TensorDataset(X_train, y_train)
    val = data_utils.TensorDataset(X_test, y_test)

    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def convert_to_tensor(arr, t):
    r"""Converts a numpy array to a tensor of type t
    
    """
    assert type(arr) is np.ndarray
    if t == 'float':
        v = Variable(torch.from_numpy(arr).float())
    if t == 'long':
        v = Variable(torch.from_numpy(arr).long())
    return v

def pair_to_tensor(X, y, types=('float', 'long')):
    r"""Converts two numpy arrays to two tensors of types t[0] and t[1]

    """
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray
    return convert_to_tensor(X, types[0]), convert_to_tensor(y, types[1])