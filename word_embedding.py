# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2020/1/11 下午12:32
# Tool ：PyCharm

'''

conver data into word embedding

'''


import numpy as np
from data_convert import ConverData
from nltk.tokenize import word_tokenize

class embedding(object):

    def __init__(self,train_raw_data,test_raw_data):
        print("__int__  embedding ")
        self.raw_train = train_raw_data
        self.raw_test =  test_raw_data

    # parsing sentence into word
    def __participle(self):
        print("parsing sentence into word")
        self.all_words = [] # all word
        self.all_sentence = []
        # train
        for i,v in enumerate(self.raw_train):
            tw = word_tokenize(v['sentence'])
            self.all_sentence.append(tw)
            for t in tw:
                if t in self.all_words:
                    continue
                else:
                    self.all_words.append(t)
            self.raw_train[i]['tokens'] = word_tokenize(v['sentence'])

        # test
        for i,v in enumerate(self.raw_test):
            tt = word_tokenize(v['sentence'])
            self.all_sentence.append(tt)
            for t in tt:
                if t in self.all_words:
                    continue
                else:
                    self.all_words.append(t)
            self.raw_test[i]['tokens'] = tt
        self.all_words = set(self.all_words)

    # load glove and meke embedding
    def __loadGloveModel(self):
        print("Loading Glove Model")
        path_glove = '/home/uyplayer/Github/Glov/glove.6B.300d.txt'
        f = open(path_glove, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        model['unknown'] = np.zeros(300)
        print("Done.", len(model), " words loaded!")
        return model


if __name__=="__main__":
    train, test = ConverData.getData()
    em = embedding(train,test)
    em.loadGloveModel()












