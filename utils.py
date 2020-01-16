# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2020/1/14 下午7:20
# Tool ：PyCharm

'''

some data process

'''

import numpy as np


# batch prosessing
def get_batch_data(x_data,y_data, aspect,batch_size,n_iter=100,is_shuffle=False):
    for index in batch_index(len(y_data), batch_size, n_iter, is_shuffle):
        x = x_data[index]
        y = y_data[index]
        s = aspect[index]
        yield x, y,s


def batch_index(length, batch_size, n_iter=100, is_shuffle=False):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]