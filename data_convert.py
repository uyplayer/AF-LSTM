# coding: utf-8
# Team :  uyplayer team
# Author： uyplayer 
# Date ：2020/1/10 下午3:52
# Tool ：PyCharm
'''
this python file is used to convert all data filese in Datasets(dir) into united file ;
let these united file into train data and test data
'''

import os
import xml.etree.ElementTree as ET
import os
import numpy as np


class ConverData(object):

    def __init__(self,raw_data_dir):
        self.rawdir = raw_data_dir
        self.polar = {'positive': [2], 'neutral': [1], 'negative': [0]}

    # handle the raw data dir
    def __handledie(self):
        return os.listdir(self.rawdir)

    # handle the raw data in each file
    def conver(self):
        self.raw_cases = []
        for i in self.__handledie():
            xml = ET.parse(self.rawdir+"/"+i)
            for sent in xml.findall('sentence'):
                if sent.find('aspectTerms'):
                    text = sent.find('text').text
                    asps = sent.find('aspectTerms').findall('aspectTerm')
                    for asp in asps:
                        if asp.attrib['polarity'] in self.polar:
                            self.raw_cases.append((text, asp.attrib['term'], asp.attrib['polarity']))
        # shuffle
        np.random.seed(100)
        self.raw_cases = np.random.permutation(self.raw_cases)
        # split data
        train = self.raw_cases[:int(0.7*len(self.raw_cases))]
        test = self.raw_cases[int(0.7*len(self.raw_cases)):]

        return train,test

if __name__=="__main__":

    # root path
    root_path = os.getcwd()
    # datasets's dir
    data_dir = root_path + "/Github/AF-LSTM/Datasets/raw_data"
    raw_data = ConverData(data_dir)
    train,test = raw_data.conver()









