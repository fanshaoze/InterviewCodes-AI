import json

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
        hidden_state = hidden_state.to(device)
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


class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(BasicLSTM, self).__init__()

        # Initialize dimensions
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Linear layers for input, forget, and output gates
        self.input_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)  # Input gate
        self.forget_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)  # Forget gate
        self.cell_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)  # Cell gate (candidate)
        self.output_gate = nn.Linear(embedding_dim + hidden_size, hidden_size)  # Output gate

        # Final linear layer to map hidden state to output
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial hidden and cell states (zeros at the beginning)
        h = torch.zeros(x.size(0), self.hidden_size)  # hidden state
        c = torch.zeros(x.size(0), self.hidden_size)  # cell state
        h = h.to(device)
        c = c.to(device)

        # Process each word in the sequence
        for t in range(x.size(1)):
            # Get the embedding for the current word
            embedding = self.embedding(x[:, t])  # Shape: (batch_size, embedding_dim)

            # Concatenate the embedding with the previous hidden state
            combined = torch.cat((embedding, h), dim=1)  # Shape: (batch_size, embedding_dim + hidden_size)

            # Calculate the forget, input, and output gates
            f = torch.sigmoid(self.forget_gate(combined))  # Forget gate
            i = torch.sigmoid(self.input_gate(combined))  # Input gate
            o = torch.sigmoid(self.output_gate(combined))  # Output gate
            c_ = torch.tanh(self.cell_gate(combined))  # Candidate cell state

            # Update the cell state and hidden state
            c = f * c + i * c_  # Cell state
            h = o * torch.tanh(c)  # Hidden state

        # Output layer to get the prediction
        output = self.hidden_to_output(h)  # Shape: (batch_size, output_size)

        return output


class LSTMModel_Torch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel_Torch, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Embedding layer to transform input indices to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Output layer (hidden state to vocab size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding lookup
        x = self.embedding(x)
        # LSTM processing
        out, _ = self.lstm(x)  # LSTM output: (batch_size, seq_len, hidden_size)

        # Get output from the final hidden state
        output = self.hidden_to_output(out[:, -1, :])  # Use last timestep's hidden state
        return output


# text = """
# The cat jumped over the fence. She smiled and waved at her friend. The sun was setting behind the mountains. The dog barked loudly in the distance. He ran quickly towards the finish line. The trees rustled in the wind, casting long shadows. I watched the birds fly high in the sky. They lived happily ever after in a small village. She opened the door to find a surprise waiting. The flowers bloomed in the garden, filling the air with fragrance. The car sped down the highway, leaving dust behind. He wrote a letter to his parents, sharing the good news. The kids played soccer in the park until it got dark. He made a cup of coffee and sat by the window. The phone rang, and she picked it up immediately. The lake was calm and peaceful, with not a ripple in sight. They took a walk along the beach, collecting seashells. The stars twinkled in the clear night sky. He turned the page of the book, eager to see what happened next. The wind howled through the trees as the storm approached. She painted a beautiful landscape, capturing the essence of nature. The ice cream melted quickly in the hot sun. The bookshop had an old wooden floor that creaked when you walked on it. They sat by the fire, sipping hot chocolate on a cold winter's night. The sound of the guitar echoed through the empty room. He forgot his keys on the kitchen counter. The dog wagged its tail when it saw its owner. The train arrived late at the station, causing a delay. She loved the smell of fresh-baked bread in the morning. The plane took off smoothly, soaring into the clouds. They decided to take a road trip across the country. The baby giggled when her mother played peek-a-boo. He wore his favorite jacket, even though it was warm outside. The old man sat on the porch, watching the world go by. She sang her favorite song, hitting every note perfectly. The children gathered around the teacher, eager to learn. The restaurant was busy, but the food was worth the wait. They spent the afternoon exploring the city, taking photos along the way. The book was so good that he couldn't put it down. She made a wish on her birthday and hoped it would come true. The cake was decorated with colorful frosting and sprinkles. The dog chased its tail in circles, looking silly. He tried to fix the broken chair with a hammer and some nails. The mountain trail was steep, but the view from the top was worth it. The rain stopped, and the sun came out, creating a beautiful rainbow. She smiled at the compliment, feeling proud of herself. The soup was too hot to eat right away, so she waited a bit. They played board games all afternoon, laughing and having fun. He turned on the TV to watch his favorite show. The clock ticked steadily as he waited for the bus. The birds chirped in the trees, greeting the new day. She put on her shoes and headed out the door to meet her friend. The coffee shop was cozy, with soft music playing in the background. He tied his shoes tightly before heading out for a jog. The door creaked as he opened it, revealing the dark room inside. The baby slept soundly in her crib, unaware of the noise. They celebrated their anniversary at a fancy restaurant. The wind blew the leaves off the trees, scattering them across the ground. He sat on the couch, watching the game with his friends. The kitten curled up in a ball and fell asleep in the sun. She wore a red dress that sparkled in the light. The rain started pouring down, so they decided to stay inside. He picked up the guitar and strummed a few chords. The cake was delicious, with layers of chocolate and vanilla. They took a picture together, capturing the moment forever. The movie ended with a surprising twist that no one saw coming. He carried the heavy box up the stairs, grunting with effort. The wind swept through the field, making the grass sway. They planned a trip to the mountains for the weekend. The sunset painted the sky with shades of pink and orange. The bus arrived just as they were about to miss it. The sound of footsteps echoed down the hallway. He put his keys in his pocket and locked the door behind him. The children ran around the yard, chasing each other. She opened the window and let the fresh air in. The sound of the rain tapping on the roof was soothing. He was excited to meet his favorite author at the book signing. They sat on the bench by the lake, watching the ducks swim. The cat stretched out on the windowsill, soaking up the sun. She laughed at the joke, feeling lighthearted. The car engine started with a loud roar, and they were on their way. The breeze felt cool against her skin as she walked by the ocean. He adjusted his tie before heading into the meeting. They enjoyed a quiet evening at home, relaxing after a busy day. The baby took her first steps, and the parents cheered with joy. The lights flickered before going out, leaving the room in darkness. The mountains stood tall and majestic in the distance. He glanced at his watch and realized he was running late. The house was quiet except for the sound of the ticking clock. They packed their bags and headed to the airport for their vacation. The popcorn was buttery and warm, perfect for movie night. She found a note in her mailbox that made her smile. The dog followed its owner everywhere, never leaving her side. The music played softly in the background as they dined. The car screeched to a halt just before hitting the curb. He felt a sense of accomplishment after finishing the project. They took a nap in the hammock, listening to the birds sing. The restaurant had a beautiful view of the ocean, making it the perfect place for dinner. He was excited to try the new restaurant in town. The snowflakes drifted down from the sky, covering the ground in a white blanket. The kids put on their jackets and ran outside to play. She loved the smell of the ocean breeze, fresh and salty. The fire crackled as they sat around the campfire, roasting marshmallows. The waiter brought their food to the table with a smile. The phone beeped, indicating a new message. He gazed at the stars, wondering about the mysteries of the universe. They walked along the path, talking about their hopes and dreams. The dog barked at the squirrel running up the tree. She ran to catch the bus, but it was already pulling away. The waves crashed against the shore, sending spray into the air. He was eager to start his new job and make a positive impression. The rain continued to fall, making the streets shine. The teacher wrote the math problem on the board for the class to solve. They rode their bikes through the park, enjoying the sunny weather. The ice cream truck played its familiar tune as it drove by. He sat down at the piano and began to play a song. The bakery had a long line of customers waiting to buy fresh bread. The children picked flowers and made bouquets for their mothers. The sound of the doorbell echoed through the house. He smiled as he opened the door to find his friends waiting outside. The cat meowed loudly, asking for food. They went for a walk in the park, enjoying the warm summer day. The flowers in the garden were blooming, attracting bees and butterflies. The phone rang, and he picked it up quickly. They made a wish as they blew out the candles on the cake. The music played louder as the party grew more lively. She grabbed her coat and left the house, heading out for a walk. The children laughed as they ran around the playground. The sound of the waves crashing against the rocks was calming. The party was a success, and everyone had a great time. He watched the sunset from the balcony, feeling peaceful. They built a sandcastle at the beach, complete with a moat. The smell of fresh coffee filled the kitchen as she brewed a pot. The car broke down on the side of the road, and they had to call for help. The book was so interesting that she read it in one sitting. The puppy chased its tail, rolling around on the floor. The lights twinkled in the distance as the city came to life. The storm passed, leaving behind a rainbow in the sky. He decided to take a break and have a cup of tea. They watched the fireworks light up the night sky. The train moved slowly down the tracks, passing through small towns. The kids built a fort out of blankets and pillows in the living room. She looked at the clock and realized it was time to go. The snow fell gently, covering everything in a blanket of white. They ate dinner outside, enjoying the cool evening air. The airplane flew high above the clouds, offering a view of the landscape below. The dog ran in circles, wagging its tail happily. He played his guitar softly, strumming the strings. They sat by the fire, listening to the crackling of the wood. The streetlights flickered as night fell. She took a deep breath and stepped outside to enjoy the fresh air. The children gathered around the campfire, listening to the story. The ice cream cone melted in the heat of the summer sun. They held hands as they walked down the path, enjoying the quiet moment. The rain stopped, and the clouds parted, revealing a clear sky. He smiled as he saw the familiar sight of his house in the distance. They gathered around the table for a family dinner. The clouds rolled in, signaling the approach of a storm. The garden was filled with flowers of every
# """
file_path = 'HPStone.txt'
with open(file_path, 'r') as file:
    # Read the entire file and store it as a single string
    file_content = file.read()
text = file_content[:1000000]
text.replace('\n', ' ')
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
seq_length = 10
X = []
y = []
for i in range(len(word_indices) - seq_length):
    X.append(word_indices[i:i + seq_length])
    y.append(word_indices[i + seq_length])
device = torch.device('cuda')
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
X = X.to(device)
y = y.to(device)

# 3. Initialize the model, loss function, and optimizer
input_size = vocab_size  # Input size is the vocab size (one-hot encoded)
embedding_dim = 128  # Embedding size is the vocab size
hidden_size = 128  # Size of the hidden layer
output_size = vocab_size  # Output size is the vocab size (for prediction)


def train_batch(X, model, num_epochs, criterion, optimizer):
    # put the model to the cuda
    loss_list = []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        model.to(device)
        X.to(device)
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
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer)
    return model, loss_list


def train_with_RNNModel_TorchRNN():
    model = RNNModel_TorchRNN(input_size, hidden_size, output_size)
    total_params = count_parameters(model)
    print(f"Torch RNN Trainable Params: {total_params}")
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer)
    return model, loss_list

def train_with_LSTM():
    model = BasicLSTM(vocab_size, embedding_dim, hidden_size, output_size)
    total_params = count_parameters(model)
    print(f"TLSTM Params: {total_params}")
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer)
    return model, loss_list

def train_with_LSTMModel_Torch():
    model = LSTMModel_Torch(input_size, hidden_size, output_size)
    total_params = count_parameters(model)
    print(f"Torch LSTM Trainable Params: {total_params}")
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    loss_list = train_batch(X, model, num_epochs, criterion, optimizer)
    return model, loss_list

_, lossRNN = train_with_RNN()
_, lossRNN_torch = train_with_RNNModel_TorchRNN()
_, lossLSTM = train_with_LSTM()
_, lossLSTM_torch = train_with_LSTMModel_Torch()
plt.plot(lossRNN)
plt.plot(lossRNN_torch)
plt.plot(lossLSTM)
plt.plot(lossLSTM_torch)
plt.legend(['basic RNN', 'nn.RNN', 'basic LSTM', 'nn.LSTM'])
plt.savefig('RNN_loss.png')
