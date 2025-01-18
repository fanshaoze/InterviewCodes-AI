import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import string

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Define Hyperparameters
input_size = 256  # Size of embedding (for simplicity)
hidden_size = 512  # Hidden size of RNN
output_size = 256  # Size of output (Chinese characters)
num_layers = 4  # Number of layers in RNN
batch_size = 256
learning_rate = 0.001
epochs = 500
pad_token = "<PAD>"  # Padding token
start_token = "<START>"  # Start token for translation
end_token = "<END>"  # End token for translation

# Simple data for training (substitute with actual translation data)
english_sentences = ["hello", "how are you", "I am fine", "thank you", "you are welcome", "goodbye", "good morning",
                     "good night", "what is your name", "my name is", "nice to meet you", "see you later", "take care",
                     "what time is it", "I love you", "I miss you", "happy birthday", "congratulations",
                     "good luck", "best wishes", "safe travels", "have a nice day",
                     "enjoy your meal", "how much is this", "where is the restroom", "I am hungry",
                     "I am tired", "I am bored", "I am busy", "I am sorry", "never mind",
                     "what do you think", "I agree", "I disagree", "I understand", "I don't understand", ]
chinese_sentences = ["你好", "你好吗", "我很好", "谢谢", "不客气", "再见", "早上好",
                     "晚安", "你叫什么名字", "我叫", "很高兴认识你", "回头见", "保重",
                     "现在几点", "我爱你", "我想你", "生日快乐", "恭喜", "好运", "最好的祝愿", "一路平安",
                     "祝你有美好的一天", "享受你的餐点", "这个多少钱", "洗手间在哪里", "我饿了", "我累了", "我无聊",
                     "我忙",
                     "对不起", "没关系", "你怎么看", "我同意", "我不同意", "我明白", "我不明白"]


# Tokenizer: Build vocab and convert sentences to index sequences
def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence.split())  # Split by space for word-level tokenization
    vocab = {word: idx + 2 for idx, word in enumerate(vocab)}  # Reserve 0 for pad and 1 for start token
    vocab[pad_token] = 0
    vocab[start_token] = 1
    vocab[end_token] = len(vocab)
    return vocab


# Tokenize a sentence into indices
def tokenize(sentence, vocab):
    tokens = sentence.split()
    return [vocab.get(word, vocab[pad_token]) for word in tokens]


# Add padding to sequences
def pad_sequence(seq, max_length, vocab):
    return seq + [vocab[pad_token]] * (max_length - len(seq))


# Pad all sentences to have the same length
max_len = max(max(len(sentence.split()) for sentence in english_sentences),
              max(len(sentence) for sentence in chinese_sentences))
vocab_en = build_vocab(english_sentences)
vocab_zh = build_vocab(chinese_sentences)

# Convert sentences to tokenized indices
tokenized_en = [tokenize(sentence, vocab_en) for sentence in english_sentences]
tokenized_zh = [tokenize(sentence, vocab_zh) for sentence in chinese_sentences]

# Pad sentences
padded_en = [pad_sequence(sentence, max_len, vocab_en) for sentence in tokenized_en]
padded_zh = [pad_sequence(sentence, max_len, vocab_zh) for sentence in tokenized_zh]


# Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, english_sentences, chinese_sentences):
        self.english_sentences = english_sentences
        self.chinese_sentences = chinese_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.english_sentences[idx]), torch.tensor(self.chinese_sentences[idx])


# Prepare Dataset and DataLoader
dataset = TranslationDataset(padded_en, padded_zh)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.embedding = nn.Linear(1, input_size)  # Embedding layer

    def forward(self, x):
        # Initialize hidden state for batch_size
        x = self.embedding(x.unsqueeze(2).float())
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, hidden = self.rnn(x, h0)
        return hidden


# Decoder Network
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(output_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Linear(1, input_size)

    def forward(self, x, hidden):
        x = self.embedding(x.unsqueeze(2).float()) # report bug but do not know why
        output, hidden = self.rnn(x, hidden)
        output = self.fc_out(output)
        return output, hidden


# Seq2Seq Model (Encoder-Decoder)
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers)

    def forward(self, src, trg):
        hidden = self.encoder(src)
        output, _ = self.decoder(trg, hidden)
        return output


# Instantiate the model
model = Seq2Seq(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(
    'cuda')

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []
# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for src, trg in dataloader:
        # Move tensors to GPU
        src, trg = src.to('cuda'), trg.to('cuda')

        optimizer.zero_grad()

        output = model(src, trg)

        # Reshape output and target for cross-entropy
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}')
    loss_list.append(total_loss / len(dataloader))
import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.savefig('translation_loss_plot.png')

# Testing Loop (Example)
def translate(sentence, model):
    model.eval()
    sentence = sentence.lower().split()

    # Tokenize and pad input sentence
    tokenized = tokenize(sentence[0], vocab_en)
    padded = pad_sequence(tokenized, max_len, vocab_en)

    # Add batch dimension (1 batch, sequence length, input size)
    input_tensor = torch.tensor(padded).unsqueeze(0).to('cuda').float()  # Add batch dimension of size 1

    # Start translation
    hidden = model.encoder(input_tensor)  # hidden is of shape (num_layers, 1, hidden_size)

    # Start token for decoder
    input_tensor = torch.zeros(1, 1).to('cuda')  # The input to the decoder is a start token

    output_sentence = []

    for _ in range(len(sentence)):
        output, hidden = model.decoder(input_tensor, hidden)

        # Get the predicted word (index) and convert it back to the word
        predicted_word = output.argmax(2).item()
        output_sentence.append(list(vocab_zh.keys())[list(vocab_zh.values()).index(predicted_word)])  # Convert index to word

        input_tensor = output  # Update input tensor with the decoder's output

    return ' '.join(output_sentence)


# Translate a sample sentence
test_sentence = "hello morning"
print(f"Translated: {translate(test_sentence, model)}")
