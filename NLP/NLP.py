import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import string
import matplotlib.pyplot as plt
from collections import Counter


# 1. Load and preprocess dataset
# For simplicity, we'll use a small subset of the Penn Treebank dataset.
# You can replace this with any text dataset.


# 2. Build the RNN from scratch using Linear layers
class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(BasicRNN, self).__init__()

        # Embedding layer to convert word indices to word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Linear layer for hidden to output transformation
        self.input_to_hidden = nn.Linear(embedding_dim, hidden_size)  # Linear layer for input to hidden
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)  # Hidden to hidden layer
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        # Hidden layer size
        self.hidden_size = hidden_size

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embed the input words
        embedded = self.embedding(x)  # Shape: (batch_size, seq_length, embedding_dim)

        # Initialize hidden state
        hidden_state = torch.zeros(x.size(0), self.hidden_size)

        # Process the sequence step-by-step
        for t in range(x.size(1)):
            input_t = self.embedding(x[:, t])  # Get the embedding for the current input word
            hidden_state = self.relu(self.input_to_hidden(input_t) + self.hidden_to_hidden(hidden_state))  # Update hidden state

        output = self.hidden_to_output(hidden_state)  # Output prediction
        return output

class RNNModel_TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel_TorchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Embedding layer to transform input indices to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)

        # RNN layer using nn.RNN
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # Output layer (hidden state to vocab size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding lookup
        x = self.embedding(x)
        # RNN processing
        out, _ = self.rnn(x)  # RNN output: (batch_size, seq_len, hidden_size)

        # Get output from the final hidden state
        output = self.hidden_to_output(out[:, -1, :])  # Use last timestep's hidden state
        return output


text = """
This is a simple example of a basic RNN implemented from scratch using PyTorch.
We will not use nn.RNN or any other high-level API.
Instead, we will construct the RNN manually using only basic linear layers.
This will help to better understand how RNNs work.
"""

# Tokenize the text
text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
words = text.split()

# Create a vocabulary
vocab = list(set(words))
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Convert words to indices
word_indices = [word_to_idx[word] for word in words]

# Prepare input-output pairs for language modeling
seq_length = 5
X = []
y = []
for i in range(len(word_indices) - seq_length):
    X.append(word_indices[i:i + seq_length])
    y.append(word_indices[i + seq_length])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# 3. Initialize the model, loss function, and optimizer
input_size = vocab_size  # Input size is the vocab size (one-hot encoded)
embedding_dim = 128  # Embedding size is the vocab size
hidden_size = 128  # Size of the hidden layer
output_size = vocab_size  # Output size is the vocab size (for prediction)


def train_batch(X, model, num_epochs, criterion, optimizer, plot_name='loss_RNN.png'):
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        # Compute loss
        loss = criterion(output, y)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # save the loss and plot it
        loss_list.append(loss.item())
    return loss_list


def count_parameters(model):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)

    return total_params

def train_with_RNN():
    model = BasicRNN(vocab_size, embedding_dim, hidden_size, output_size)
    total_params = count_parameters(model)
    print(f"Basic RNN Trainable Params: {total_params}")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer, plot_name='loss_RNN.png')
    return model, loss_list


def train_with_RNNModel_TorchRNN():
    model = RNNModel_TorchRNN(input_size, hidden_size, output_size)
    total_params = count_parameters(model)
    print(f"Torch RNN Trainable Params: {total_params}")
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer, plot_name='loss_RNNModel_TorchRNN.png')
    return model, loss_list

_, lossRNN = train_with_RNN()
_, lossRNN_torch = train_with_RNNModel_TorchRNN()
plt.plot(lossRNN)
plt.plot(lossRNN_torch)
plt.legend(['basic RNN', 'nn.RNN'])
plt.savefig('RNN_loss.png')
