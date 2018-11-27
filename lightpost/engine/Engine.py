import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from tensorboardX import SummaryWriter

from .solvers import train, evaluate, accuracy, evaluate_batch

from tqdm import tqdm
import numpy as np
import pandas as pd

class Engine:
	r"""Dynamic Training Engine. 

	Args:
		pipeline: Data Pipeline
		model: PyTorch nn.Module based Model
		criterion: Name of the loss function to be applied. A custom nn.Module criterion can also be passed.
		optimizer: Name of the optimizer to be used. A custom optimizer could also be used.
		scheduler: Name of the learning rate scheduler. Defaultly set to None.
		lr: Learning rate to be used for the optimizer.
		use_tensorboard: Toggle True to use the Tensorboard Visualization Tool

	"""

	def __init__(self, pipeline, model, criterion='cross_entropy', optimizer='adam', scheduler=None, lr=1e-4, use_tensorboard=False):
		self.model = model
		self.optimizer = self.build_optimizer(self.model, optimizer, lr)
		self.criterion = self.build_criterion(criterion)
		self.pipeline = pipeline
		self.scheduler = self.build_scheduler(scheduler)
		self.tensorboard = None

		if use_tensorboard:
			self.tensorboard = SummaryWriter()

	def fit(self, epochs=1, print_every=1, disable_tqdm=False):
		r"""Trains the engine's model for a number of epochs and logs it.

		Args:
			epochs: Number of epochs to train the model.
			print_every: Prints the logs every N iterations.
			disable_tqdm: Disable progress bar which prints every by-batch training

		"""
		for e in range(1, epochs + 1):
		    train_loss, train_acc = train(self.model, self.criterion, self.optimizer, self.pipeline.train_loader, disable_tqdm=disable_tqdm)
		    val_loss, val_acc = evaluate(self.model, self.criterion, self.pipeline.val_loader, disable_tqdm=disable_tqdm)
		    if self.scheduler is not None:
		    	self.scheduler.step(val_loss)

		    if self.tensorboard is not None:
		    	self.tensorboard.add_scalar('train_loss', train_loss, e)
		    	self.tensorboard.add_scalar('train_acc', train_acc, e)
		    	self.tensorboard.add_scalar('val_loss', val_loss, e)
		    	self.tensorboard.add_scalar('val_acc', val_acc, e)
		    	
		    if e % print_every == 0:
		    	print("Epoch {:5} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f}".format(e, train_loss, train_acc, val_loss, val_acc))

	def predict(self, X):
		r"""Uses the model for inference. Returns tensor after the final layer of the model.

		Args:
			X: Tensor of input data.

		""" 
		return self.model(X)

	def evaluate_model(self, X, y):
		r"""Evaluates the loss and accuracy of the model given a test set.

		Args:
			X: Tensor of testing data features.
			y: Tensor of labels (indices) for each input feature.

		"""
		return evaluate_batch(self.model, self.criterion, X, y)

	def build_optimizer(self, model, optimizer, lr):
		if type(optimizer) is str:
			if optimizer == 'adam':
				return optim.Adam(model.parameters(), lr=lr)

		else:
			return optimizer

	def build_criterion(self, criterion):
		if type(criterion) is str:
			if criterion == 'cross_entropy':
				return nn.CrossEntropyLoss()

		else:
			return criterion

	def build_scheduler(self, scheduler):
		if type(scheduler) is str:
			if scheduler == 'plateau':
				return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

		else:
			return scheduler

	def save_weights(self, path):
		torch.save(self.model, path)
