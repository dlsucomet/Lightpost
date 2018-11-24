import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as data_utils

from torchtext import vocab

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer

def load_data(path, data, target):
    r"""Loads data from a dataset.

    Args:
        path: Path to the csv dataset.
        data: Name of the column to be used as features.
        target: Name of the column to be used as targets.

    """
    df = pd.read_csv(path)
    X = list(df[data])
    y = list(df[target])
    return X, y

def tokenize(sentence):
    r"""Regular Expression based tokenizer. Returns a list of tokens."""

    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence)

def tokenize_array(X, tokenizer=tokenize):
    r"""Tokenizes an array of sequences.

    Args:
        X: Array of sequences.
        tokenizer: Function to tokenize a single sequence.
    """
    tokenized_X = []
    for sentence in tqdm(X):
        tokenized_X.append(tokenizer(sentence))
    return tokenized_X

def pad(X, maxlen=20):
    r"""Pads/Truncates an array of sequences.
    
    Args:
        X: Array of tokenizes sequences.
        maxlen: Maximum length of a sequence.
    """
    batch = []
    for sentence in tqdm(X):
        l = len(sentence)
        if l > maxlen:
            batch.append(sentence[:maxlen])
        elif l == maxlen:
            batch.append(sentence)
        else:
            dif = maxlen - l
            pad = ['<PAD>' for i in range(dif)]
            batch.append(sentence + pad)
    return batch

def build_vocab(X):
    r"""Builds a vocabulary dictionaries from an array of sequences."""

    nX = [" ".join(X[i]) for i in range(len(X))] # Join all words
    nS = ' '.join(nX).split() # Join all sentences
    set_ns = list(set(nS)) # Eliminate redundant words
    set_ns.append('<UNK>') # Add unknown 
    vocab_dict = {i: set_ns[i] for i in range(len(set_ns))} # Build vocab dict
    rev_vocab_dict = {vocab_dict[ix]:ix for ix in vocab_dict.keys()}
    
    return vocab_dict, rev_vocab_dict, len(set_ns)

def serialize(X, rev_vocab_dict):
    r"""Vectorizes each sequences in an array using a dictionary of token-to-index values.

    Args:
        X: Array of tokenized, padded sequences.
        rev_vocab_dict: Reverse vocabulary dictionary.
    """

    batch = []
    for sentence in tqdm(X):
        nsent = [rev_vocab_dict[word] if word in rev_vocab_dict.keys() else rev_vocab_dict['<UNK>'] for word in sentence]
        batch.append(nsent)
    return np.array(batch)

def build_labels(y):
    r"""Builds a dictionary of labels and corresponding indices, then returns a vectorized array of labels."""

    rdi = {}
    seen = []
    i = 0
    for label in y:
        if label not in seen:
            seen.append(label)
            rdi[label] = i
            i += 1
    nlabels = np.array([rdi[label] for label in y])
    return nlabels, rdi

def preprocess(path, text, target, maxlen, tokenizer=tokenize):
    r"""Preprocessing pipeline."""

    X, y = load_data(path, text, target)
    X = tokenize_array(X, tokenizer=tokenize)
    X = pad(X, maxlen=maxlen)
    vocab_dict, rev_vocab_dict, vocab_len = build_vocab(X)
    X = serialize(X, rev_vocab_dict)
    y, rev_labels_dict = build_labels(y)

    return X, y, vocab_dict, rev_vocab_dict, vocab_len, rev_labels_dict

def split_data(X, y):
    r"""Splits text data into training and testing sets, then converts them into PyTorch long tensors."""

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = Variable(torch.from_numpy(X_train).long())
    X_test = Variable(torch.from_numpy(X_test).long())
    y_train = Variable(torch.from_numpy(y_train).long())
    y_test = Variable(torch.from_numpy(y_test).long())
    return X_train, X_test, y_train, y_test
