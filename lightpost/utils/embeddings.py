import torch
import torch.nn as nn
from torchtext import vocab
from tensorboardX import SummaryWriter

def build_embedding(vocab_dict, vocab_count, embedding_dimension, v):
	"""Loads an embedding layer based on vocabularies and a pretrained word vector.

	Args:
		vocab_dict: The index-to-word vocabulary dictionary of the training set.
		vocab_count: The number of words found in the training set.
		embedding_dimension: The dimensions of the pretrained word vectors.
		v: A TorchText vector object containing the weights of the pretrained word vectors.

	"""

	emb = nn.Embedding(vocab_count, embedding_dimension)
	batch = []
	for i in range(vocab_count):
		batch.append(v[vocab_dict[i]])
	weights = torch.stack(batch)
	emb.weight.data.copy_(weights)
	emb.weight.requires_grad = False
	return emb


def load_then_visualize_embeddings(path):
	"""Visualizes pretrained embeddings into tensorboard.

	Args:
		path: Path to the pretrained vector file.
	"""
	writer = SummaryWriter()
	v = vocab.Vectors(path)
	writer.add_embedding(v.vectors, v.itos) 

def visualize_embeddings(v):
	"""Visualizes loaded vectors from pretrained embeddings into tensorboard.

	Args:
		v: The torchtext.vocab.Vector object that contains weights of the embeddings.
	"""
	writer = SummaryWriter()
	writer.add_embedding(v.vectors, v.itos) 