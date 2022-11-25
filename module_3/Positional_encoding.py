import math
import torch
from torch import nn, Tensor
from collections import Counter
from itertools import chain
import sys

train_path = "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/train_filtered.tokens"
val_path = "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/val_filtered.tokens"
test_path = "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/test_filtered.tokens"



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # dim: 1 , a column vector of size max_len
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) 
        pe[:, 0, 0::2] = torch.sin(position * div_term) 
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # register a buffer to the module

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DataPreprocessor:
    def __init__(self, train_path, val_path, test_path):
        super(DataPreprocessor, self).__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        train_list = self.read_data(self.train_path)
        val_list = self.read_data(self.val_path)
        test_list = self.read_data(self.test_path)

        self.train_vocab = self.get_vocab(train_list)
        self.val_vocab = self.get_vocab(val_list)
        self.test_vocab = self.get_vocab(test_list)

        self.train_word2index = self.word2index(self.train_vocab)
        self.val_word2index = self.word2index(self.val_vocab)
        self.test_word2index = self.word2index(self.test_vocab)

        self.train_tokens = self.token_int_representation(train_list, self.train_word2index)
        self.val_tokens = self.token_int_representation(val_list, self.val_word2index)
        self.test_tokens = self.token_int_representation(test_list, self.test_word2index)

        self.train_max_length = self.max_length(self.train_tokens)
        self.val_max_length = self.max_length(self.val_tokens)
        self.test_max_length = self.max_length(self.test_tokens)

        self.max_len_tokens = max(self.train_max_length, self.val_max_length, self.test_max_length)

        self.train_tokens = self.pad_sequence(self.train_tokens, self.max_len_tokens)
        self.val_tokens = self.pad_sequence(self.val_tokens, self.max_len_tokens)
        self.test_tokens = self.pad_sequence(self.test_tokens, self.max_len_tokens)

        self.train_tokens = self.convert_to_tensor(self.train_tokens)
        self.val_tokens = self.convert_to_tensor(self.val_tokens)
        self.test_tokens = self.convert_to_tensor(self.test_tokens)

    def read_data(self, file):
        token_list= []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                token_list.append(line.split())
        return token_list

    def get_vocab(self, token_list):
        word_counts = Counter(chain(*token_list))
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        return vocab

    def word2index(self, vocab):
        word2index = {word: i for i, word in enumerate(vocab)}
        return word2index

    def index2word(self, vocab):
        index2word = {i: word for i, word in enumerate(vocab)}
        return index2word

    def token_int_representation(self, tokens, word2index):
        return [[word2index[word] for word in token] for token in tokens]

    def max_length(self, tokens):
        return max([len(token) for token in tokens])

    def pad_sequence(self, tokens, max_len):
        return [token + [0] * (max_len - len(token)) for token in tokens]

    def convert_to_tensor(self, tokens):
        return torch.tensor(tokens)

    def get_tokens(self):
        return self.train_tokens, self.val_tokens, self.test_tokens

"""
tokens = []
with open(train_tokens, "r") as f:
    lines = f.readlines()
    for line in lines:
        tokens.append(line)

tokens = [token.split() for token in tokens]
tokens[0]

# turn strings into integers
# count the number of words
word_counts = Counter(chain(*tokens))
word_counts.elements
word_counts.most_common(10)
word_counts.get("the")

# sort the words by frequency, you sort it for the vocabulary, so that the most frequent words are at the top, not for the tokens.
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
# info about vocabulary
print("Vocabulary size: ", len(vocab))
print("Most common words: ", vocab[:10])
print("Least common words: ", vocab[-10:])

# add padding and unknown tokens
vocab = ['<PAD>', '<UNK>'] + vocab

# create a dictionary of words to integers
word2idx = {word: idx for idx, word in enumerate(vocab)}
word2idx.keys()  # '47LI850A' : 217 , integer representation of the word
word2idx['47LI850A']

# convert the tokens to integers
tokens = [[word2idx[word] for word in token] for token in tokens]
tokens[4000]

# find the length of the longest token
max_len = max([len(token) for token in tokens])
print("Max length of tokens: ", max_len)

# pad the tokens
tokens = [token + [0] * (max_len - len(token)) for token in tokens]
tokens[4000]

# create train and validation sets
#train_tokens = tokens[:4000]
#val_tokens = tokens[4000:]

# convert to tensors
train_tokens = torch.tensor(train_tokens)

# create the dataloaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_tokens, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_tokens, batch_size=batch_size, shuffle=False)

# check the shapes
for batch in train_loader:
    print(batch.shape)
    break

# pass the tokens through the model
d_model = max_len # embedding dimension: 730
pe = PositionalEncoding(d_model, dropout=0.1, max_len=max_len)
for batch in train_loader:
    batch = batch.float()
    batch = pe(batch)
    print(batch.shape)
    break
"""

if __name__ == '__main__':
    data_preprocessor = DataPreprocessor(train_path, val_path, test_path)
    train_tokens, val_tokens, test_tokens = data_preprocessor.get_tokens()

    max_len = data_preprocessor.max_len_tokens

    # ask if you want to continue
    print("Do you want to continue? (y/n)")
    answer = input()
    if answer == "y":
        pass
    else:
        sys.exit()

    pos_en = PositionalEncoding(max_len, dropout=0.1, max_len=max_len)
    train_tokens = pos_en(train_tokens)
    val_tokens = pos_en(val_tokens)
    test_tokens = pos_en(test_tokens)



    # save the tensors of positions
    torch.save(train_tokens, "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/train_tokens.pt")
    torch.save(val_tokens, "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/val_tokens.pt")
    torch.save(test_tokens, "/Users/hamzagorgulu/Desktop/thesis/Waris Final/tupras/data/test_tokens.pt")














