import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

def split(X, y):
    r"""Splits X and y arrays into training and testing sets, then converts them into PyTorch variables.

    Args:
        X: Array of features.
        y: Array of label indices.

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = Variable(torch.from_numpy(X_train).float())
    X_test = Variable(torch.from_numpy(X_test).float())
    y_train = Variable(torch.from_numpy(y_train).long())
    y_test = Variable(torch.from_numpy(y_test).long())
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