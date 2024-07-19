import numpy as np
from transformers import AutoTokenizer

class BertVectorizer():
    def __init__(self,bert_tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        self.tokenizer=bert_tokenizer
        self.token_length = len(self.tokenizer.get_vocab())
    def transform(self, list_input):
        out = np.zeros((len(list_input),self.token_length),dtype=np.int8)
        for i in range(len(list_input)):
            current_out = self.tokenizer.encode(list_input[i])[1:-1]
            out[i][current_out]+=1
        return out