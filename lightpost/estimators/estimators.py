import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

class LSTMClassifier(nn.Module):
    def __init__(self, pretrained, embedding_dim, hidden_dim, output_dim, bidirectional, recur_layers, recur_dropout, dropout):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = pretrained
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=recur_layers, dropout=recur_dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) if bidirectional else nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_state = None

    # Initializes the hidden state given RNN parameters
    def initialize_hidden(self, bidirectional, num_layers, batch_size, hidden_dim):
        if bidirectional:
            h_0 = torch.zeros(2 * num_layers, batch_size, hidden_dim)
            c_0 = torch.zeros(2 * num_layers, batch_size, hidden_dim)
        else:
            h_0 = torch.zeros(num_layers, batch_size, hidden_dim)
            c_0 = torch.zeros(num_layers, batch_size, hidden_dim)
        return h_0, c_0

    # Forward propagation pass
    def forward(self, X):
        self.hidden_state = self.initialize_hidden(self.rnn.bidirectional, self.rnn.num_layers, len(X), self.rnn.hidden_size)
        out = self.embedding(X).permute(1, 0, 2)
        out, self.hidden_state = self.rnn(out, self.hidden_state)
        out = self.dropout(out[-1, :, :])
        out = self.dropout(torch.relu(self.fc1(out)))
        out = torch.log_softmax(self.fc2(out), dim=1)
        return out

class MLPClassifier(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
		super(MLPClassifier, self).__init__()

		self.inp = nn.Linear(input_dim, hidden_dim)
		self.hidden = [nn.Linear(hidden_dim, hidden_dim) for n in range(num_layers)]
		self.output = nn.Linear(hidden_dim, output_dim)

	def forward(self, X):
		out = torch.relu(self.inp(X))
		for layer in self.hidden:
			out = torch.relu(layer(out))
		out = torch.sigmoid(self.output(out))
		return out




