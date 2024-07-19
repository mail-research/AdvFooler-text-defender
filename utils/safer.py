import json
import numpy as np
import torch
import string
import pickle
from typing import Union, Dict
import json

class WordSubstitude:
    def __init__(self, table: Union[str, Dict]):
        '''
        table: str or Dict, when type(table) is string, mean path, using pickle to load.
        '''
        if isinstance(table, str):
            if table.split(".")[-1]=="pkl":
                pkl_file = open(table, 'rb')
                self.table = pickle.load(pkl_file)
                pkl_file.close()
            else:
                self.table = json.load(open(table))
        else:
            self.table = table
        self.table_key = set(list(self.table.keys()))
        self.exclude = set(string.punctuation)

    def get_perturbed_batch(self, sentence: str, rep:int = 1):
        out_batch = []
        sentence_in_list = sentence.split()
        for k in range(rep):
            tmp_sentence = []
            for i in range(len(sentence_in_list)):
                token = sentence_in_list[i]
                if token[-1] in self.exclude:
                    tmp_sentence.append(self.sample_from_table(token[0:-1]) + token[-1])
                else:
                    tmp_sentence.append(self.sample_from_table(token))
            out_batch.append(' '.join(tmp_sentence))
        return out_batch
    
    def sample_from_table(self, word):
        if word in self.table_key:
            tem_words = self.table[word]['set']
            #tem_words = self.table[word]
            num_words = len(tem_words)
            if num_words==0:
                return word
            index = np.random.randint(0, num_words)
            return tem_words[index]
        else:
            return word