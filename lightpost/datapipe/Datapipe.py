from ..utils.text import preprocess, split_data, tokenize_array, pad, tokenize, serialize
from ..utils.embeddings import build_embedding
from ..utils.general import generate_loaders, split, split_convert, pair_to_tensor

import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext import vocab
import pandas as pd
import numpy as np

class Datapipe:
	r"""Data pipeline for automatic batching. This datapipe is the basic pipeline that can be used for generic tensors.

	Args:
		X: Input array of features (may be numpy arrays or python lists).
		y: Input array of targets.
		types: Target tensor types for feature and target arrays
		batch_size: Batch size to be used when splitting in batches.
	
	"""
	def __init__(self, X, y, types=('float', 'long'), batch_size=32):
		self.X, self.y = pair_to_tensor(X, y, types)
		self.batch_size = batch_size
		X_train, X_test, y_train, y_test = split_convert(X, y, types)
		self.train_loader, self.val_loader = generate_loaders(X_train, X_test, y_train, y_test, batch_size)


class CSVPipe:
	def __init__(self, path, features, targets, types, batch_size=32):
		pass


class Textpipe:
	r"""Specialized data pipeline for text data.

	Args:
		path: Path to the csv file of the training dataset/corpus.
		text: Column name of the data to be used as features.
		target: Column name of the target to be used as labels.
		maxlen: Maximum length of sequences.
		tokenizer: The tokenizer function to be used. Must return a list of tokens. Defaultly uses Regexp Tokenizer from NLTK.
		pretrained_embeddings: Boolean value that sets if a vector of pretrained embeddings should be used.
		embed_path: Path to the vector file of pretrained embeddings.
		embed_dim: Dimensions of embedding layer for sequences.
		batch_size: Batch size to be used when splitting in batches.

	"""
	def __init__(self, path, text, target, maxlen, tokenizer=tokenize, pretrained_embeddings=False, embed_path='', embed_dim=300, batch_size=32):
		self.maxlen = maxlen
		self.tokenizer = tokenizer
		self.X, self.y, self.vocab_dict, self.rev_vocab_dict, self.vocab_len, self.rev_labels_dict = preprocess(path, text, target, maxlen, tokenizer)
		X_train, X_test, y_train, y_test = split_data(self.X, self.y)
		self.batch_size = batch_size
		self.train_loader, self.val_loader = generate_loaders(X_train, X_test, y_train, y_test, batch_size)
		self.X = Variable(torch.from_numpy(self.X).long())
		self.y = Variable(torch.from_numpy(self.y).long())

		if not pretrained_embeddings:
			self.embedding = nn.Embedding(self.vocab_len, embed_dim)
			self.embed_dim = embed_dim
		else:
			self.embedding, self.embed_dim = self.generate_embedding(embed_path, embed_dim)

	def generate_embedding(self, path, dimensions):
		r"""Generates an embedding layer based on a vector of pretrained embeddings.

		Args:
			path: Path to the vector file of pretrained embeddings.
			dimensions: Dimensions of embedding layer for sequences.
		"""

		v = vocab.Vectors(path)
		emb = build_embedding(self.vocab_dict, self.vocab_len, dimensions, v)
		return emb, dimensions

	def load_pretrained_embeddings(self, path, dimensions):
		r"""Sets the pretrained embeddings to be used within the pipeline.

		Args:
			path: Path to the vector file of pretrained embeddings.
			dimensions: Dimensions of embedding layer for sequences.
		"""

		self.embedding, self.embed_dim = self.generate_embedding(path, dimensions)

	def process_test_data(self, path, text):
		r"""Preprocesses text data from a holdout test set based on the vocabulary of the training set.

		Args:
			path: Path to the testing set file.
			text: Column name of the data to be used as features.

		"""
		X = pd.read_csv(path)
		X = list(X[text])
		X = tokenize_array(X, tokenizer=self.tokenizer)
		X = pad(X, maxlen=self.maxlen)
		X = serialize(X, self.rev_vocab_dict)
		X = Variable(torch.from_numpy(X)).long()
		return X
