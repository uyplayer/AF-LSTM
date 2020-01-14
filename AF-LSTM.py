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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# LSTM
class LSTM_Layer(nn.Module):

    def __int__(self):`









