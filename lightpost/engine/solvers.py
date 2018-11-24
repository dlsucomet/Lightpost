import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from tqdm import tqdm
import numpy as np
import pandas as pd

def accuracy(y_pred, y_acc):
    r"""Computes the multiclass accuracy of predictions.

    Args:
        y_pred: Tensor containing softmax probabilities of outputs.
        y_acc: Tensor containing indices of correct predictions.
    """
    with torch.no_grad():
        acc = torch.sum(torch.max(y_pred, 1)[1] == y_acc).item() / len(y_acc)
    return acc

def train_batch(model, criterion, optimizer, X, y):
    r"""Trains the model using a single batch.

    Args:
        model: A PyTorch nn.Module based model.
        criterion: A PyTorch nn.Module based loss function.
        optimizer: A PyTorch torch.optim based optimizer.
        X: Tensor containing the features of the batch.
        y: Tensor containing the indices of the correct predictions in the batch.

    """
    model.train()
    
    y_pred = model(X)
    loss = criterion(y_pred, y)
    acc = accuracy(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), acc

def evaluate_batch(model, criterion, X, y):
    r"""Evaluates the model using a single batch.

    Args:
        model: A PyTorch nn.Module based model.
        criterion: A PyTorch nn.Module based loss function.
        X: Tensor containing the features of the batch.
        y: Tensor containing the indices of the correct predictions in the batch.

    """
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
        acc = accuracy(y_pred, y)
    
    return loss.item(), acc

def train(model, criterion, optimizer, train_loader, disable_tqdm=False):
    r"""Trains the model for a single epoch.

    Args:
        model: A PyTorch nn.Module based model.
        criterion: A PyTorch nn.Module based loss function.
        optimizer: A PyTorch torch.optim based optimizer.
        train_loader: A PyTorch dataloader containing all batches in a dataset.
        disable_tqdm: Boolean switch that indicates if the progress bar should be printed.

    """
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(train_loader, disable=disable_tqdm):
        X, y = batch
        loss, acc = train_batch(model, criterion, optimizer, X, y)
        epoch_loss += loss
        epoch_acc += acc
    
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    
    return epoch_loss, epoch_acc

def evaluate(model, criterion, train_loader, disable_tqdm=False):
    r"""Evaluates the model for a single epoch.

    Args:
        model: A PyTorch nn.Module based model.
        criterion: A PyTorch nn.Module based loss function.
        train_loader: A PyTorch dataloader containing all batches in a dataset.
        disable_tqdm: Boolean switch that indicates if the progress bar should be printed.
    """
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(train_loader, disable=disable_tqdm):
            X, y = batch
            loss, acc = evaluate_batch(model, criterion, X, y)
            epoch_loss += loss
            epoch_acc += acc
    
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)
    
    return epoch_loss, epoch_acc