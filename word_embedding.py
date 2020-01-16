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
        self.sentence_len = 0

    # parsing sentence into word
    def __participle(self):
        print("parsing sentence into word")
        # train
        for i,v in enumerate(self.raw_train):
            tw = word_tokenize(v['sentence'])
            if len(tw)>self.sentence_len:
                self.sentence_len = len(tw)
            self.raw_train[i]['tokens'] = word_tokenize(v['sentence'])

        # test
        for i,v in enumerate(self.raw_test):
            tt = word_tokenize(v['sentence'])
            if len(tt)>self.sentence_len:
                self.sentence_len = len(tt)
            self.raw_test[i]['tokens'] = tt

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
        print("Done.", len(model), " words loaded!")
        return model

    # word_embedding
    def word_embedding(self,embedin_dim = 300):
        print("word_embedding")
        model = self.__loadGloveModel()
        embedding = []
        word_dict = dict()
        embedding.append([0.] * embedin_dim)
        cnt = 0
        for key,values in model.items():
            cnt += 1
            embedding.append(values)
            word_dict[key] = cnt
        print("embedding len",len(embedding))
        return embedding,word_dict

    #  all data we need to use
    def all_data(self):
        print("all_data")
        self.__participle()
        embedding, word_dict = self.word_embedding()
        print("embedding len:",len(embedding))
        train_ids,test_ids,train_y,test_y,train_aps_id,test_aps_id= [],[],[],[],[],[]
        # {'sentence':text.lower(), 'term':asp.attrib['term'].lower(), 'polarity': polar,'tokens':[]}
        # {'positive': 2, 'neutral': 1, 'negative': 0}
        for i in self.raw_train:
            if i['polarity'] == 2:
                train_y.append([1, 0, 0])
            elif i['polarity'] == 1:
                train_y.append([0, 1, 0])
            elif i['polarity'] == 0:
                train_y.append([0, 0, 1])
            train_aps_id.append(word_dict.get(i['term'], 0))
            ids=[]
            for t in i['tokens']:
                ids.append(word_dict.get(t, 0))
            train_ids.append(ids + [0] * (self.sentence_len - len(ids)))
        for i in self.raw_test:
            if i['polarity']==2:
                test_y.append([1,0,0])
            elif i['polarity']==1:
                test_y.append([0, 1, 0])
            elif i['polarity']==0:
                test_y.append([0, 0, 1])
            test_aps_id.append(word_dict.get(i['term'], 0))
            ids = []
            for t in i['tokens']:
                ids.append(word_dict.get(t, 0))
            test_ids.append(ids + [0] * (self.sentence_len - len(ids)))
        print("*******all data Model needs are ready*******")
        return np.asarray(train_ids, dtype=np.int32),np.asarray(test_ids, dtype=np.int32),np.asarray(train_y),np.asarray(test_y),np.asarray(train_aps_id, dtype=np.int32),np.asarray(test_aps_id, dtype=np.int32),np.asarray(embedding, dtype=np.int32),word_dict



if __name__=="__main__":
    train, test = ConverData.getData()
    em = embedding(train,test)
    train_ids, test_ids, train_y, test_y, train_aps_id, test_aps_id,embedding,word_dict = em.all_data()
    print("train_ids:",train_ids.shape)
    print("test_ids:",test_ids.shape)
    print("train_y:",train_y.shape)
    print("test_y:",test_y.shape)
    print("train_aps_id:",train_aps_id.shape)
    print("test_aps_id:",test_aps_id.shape)
    print("embedding:",embedding.shape)










