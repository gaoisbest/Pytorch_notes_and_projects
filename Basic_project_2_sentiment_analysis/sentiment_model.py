import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class RNNSelfAttention(nn.Module):
    """A structured self attentive sentence embedding
    https://arxiv.org/pdf/1703.03130.pdf
    """

    def __init__(self, args):
        super(RNNSelfAttention, self).__init__()
        self.bi_rnn = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.num_units, bidirectional=True, batch_first=True)
        self.self_attention = nn.Sequential(OrderedDict([
            ('W_s1', nn.Linear(in_features=2 * args.num_units, out_features=args.atten_units, bias=False)),
            ('tanh', nn.Tanh()),
            ('W_s2', nn.Linear(in_features=args.atten_units, out_features=args.atten_hops, bias=False))
        ]))

    def forward(self, input_embeddings):

        # output_src shape: (batch: B, seq_len: L, 2 * hidden_size: D)
        # h_n shape: (batch, num_layers * 2, hidden_size)
        output_src, (h_n, c_n) = self.bi_rnn(input_embeddings)

        # (B, L, D)
        output_shape = output_src.shape

        # (B, L, D) -> (B*L, D)
        output = output_src.reshape(-1, output_shape[2])

        # (B*L, atten_hops)
        alpha = self.self_attention(output)

        #print('alpha 1 shape: ', alpha.shape)

        # (B, L, atten_hops)
        alpha = alpha.reshape(output_shape[0], output_shape[1], -1)

        alpha = torch.softmax(alpha, dim=1)

        # (B, atten_hops, L)
        alpha = alpha.transpose(1, 2)

        # (B, atten_hops, D)
        m = torch.bmm(alpha, output_src)

        return m, alpha


class SentimentClassifier(nn.Module):
    def __init__(self, args):
        super(SentimentClassifier, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.embedding_dim)
        self.encoder = RNNSelfAttention(args)
        # set the number of hidden units is 10
        self.fc1 = nn.Linear(args.atten_hops * 2 * args.num_units, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, text):
        input_embeddings = self.embeddings(text)
        output, attention = self.encoder(input_embeddings)
        output = output.reshape(output.size(0), -1)
        return self.fc2(self.dropout(F.relu(self.fc1(output))))