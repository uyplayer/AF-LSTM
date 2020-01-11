# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2020/1/10 下午3:52
# Tool ：PyCharm
'''
this python file is used to convert all data filese in Datasets(dir) into train data and test data
'''

import os
import xml.etree.ElementTree as ET
import os
import numpy as np


class ConverData(object):

    def __init__(self,raw_data_dir):
        print(" __init__   ConverData ")
        self.rawdir = raw_data_dir
        self.polar = {'positive': 2, 'neutral': 1, 'negative': 0}

    # handle the raw data dir
    def __handledie(self):
        print(" loading dir list ")
        return os.listdir(self.rawdir)

    # handle the raw data in each file
    def conver(self):
        print(" start to conver  ")
        self.raw_cases = []
        for i in self.__handledie():
            xml = ET.parse(self.rawdir+"/"+i)
            for sent in xml.findall('sentence'):
                if sent.find('aspectTerms'):
                    text = sent.find('text').text
                    asps = sent.find('aspectTerms').findall('aspectTerm')
                    for asp in asps:
                        if asp.attrib['polarity'] in self.polar:
                            polar = self.polar[asp.attrib['polarity']]
                            self.raw_cases.append({'sentence':text.lower(), 'term':asp.attrib['term'].lower(), 'polarity': polar})
        # shuffle
        np.random.seed(100)
        self.raw_cases = np.random.permutation(self.raw_cases)
        # split data
        train = self.raw_cases[:int(0.8*len(self.raw_cases))]
        test = self.raw_cases[int(0.8*len(self.raw_cases)):]

        return train,test

    @classmethod
    def getData(cls):
        # root path
        root_path = os.getcwd()
        # datasets's dir
        data_dir = root_path + "/Datasets/raw_data/ABSA-SemEval2014"
        raw_data = ConverData(data_dir)
        train, test = raw_data.conver()

        return train,test


if __name__=="__main__":

    # root path
    root_path = os.getcwd()
    # datasets's dir
    data_dir = root_path + "/Datasets/raw_data/ABSA-SemEval2014"
    raw_data = ConverData(data_dir)
    train,test = raw_data.conver()
    # count data
    print(train[5])
    print(test[50])
    print("train:",len(train))
    print("test:",len(test))





