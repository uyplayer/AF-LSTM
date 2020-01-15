# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2020/1/14 下午4:05
# Tool ：PyCharm

'''
Implementation model in paper
'''

import os
import sys

Dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Dir)

import numpy as np
import word_embedding
import data_convert
from numpy.fft import fft, ifft

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# LSTM
class LSTM_Layer(nn.Module):

    def __int__(self, input, input_size, hidden_size, batch_size, embedding_dim, word_embedding, num_layers=1,
                bias=True):
        super(LSTM_Layer, self).__init__()
        self.hidden_dim = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.input_dim = embedding_dim
        self.bias = bias
        self.num_layers = num_layers
        self.input = input
        self.word_embedding = word_embedding
        self.embedding = nn.Embedding(self.word_embedding, self.input)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, self.num_layers)

    def forward(self):
        lstm_out, _ = self.lstm(self.embedding)
        return lstm_out


# Aspect_Normalization
class A_N(object):

    def __int__(self, aspect_input, word_embedding, input_dim):
        self.aspect_input = aspect_input
        self.word_embedding = word_embedding
        self.bn = nn.BatchNorm1d(num_features=input_dim)

    def embed(self):
        embedding = nn.Embedding(self.word_embedding, self.aspect_input)
        output = self.bn(embedding)
        normal_aspect = torch.sum(output, 1)
        return normal_aspect


# Hidenstate_Normalization
class H_N(object):

    def __int__(self, input_h, input_dim):
        self.input_h = input_h
        self.bn = nn.BatchNorm1d(num_features=input_dim)

    def embed(self):
        output = self.bn(self.input_h)
        return output


# calculate  associative memory ；circular correlation
class M(object):

    def __int__(self, h, s):
        self.h = h
        self.s = s

    # calculate circular correlation
    def correlation(self):
        return ifft(fft(self.h) * fft(self.s).conj()).real

# attention
class Attention(nn.Module):

    '''
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    '''

    def __int__(self,dimensions, attention_type='general'):
        super(Attention, self).__init__()
        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        self.attention_type = attention_type

        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions,dimensions,bias=False)

        self.linear_out = nn.Linear(dimensions*2, dimensions,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
               Args:
                   query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                       queries to query the context.
                   context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                       overwhich to apply the attention mechanism.

               Returns:
                   :class:`tuple` with `output` and `weights`:
                   * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
                     Tensor containing the attended features.
                   * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
                     Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

            # TODO: Include mask on PADDING_INDEX?

            # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
            # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights



if __name__ == "__main__":
    pass