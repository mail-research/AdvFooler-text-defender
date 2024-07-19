from utils.preprocessing import ProcessFuntionIMDBInference
from utils.preprocessing import clean_text_imdb
from tokenizers import Tokenizer
import torch
import torch.nn as nn
import numpy as np
import OpenAttack as oa
import sys
sys.path.append('../')


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()

    def forward(self, text, text_length):

        # [sentence len, batch size] => [sentence len, batch size, embedding size]
        embedded = self.embedding(text)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, text_length.to('cpu'), batch_first=True, enforce_sorted=False)
        # [sentence len, batch size, embedding size] =>
        #  output: [sentence len, batch size, hidden size]
        #  hidden: [1, batch size, hidden size]
        packed_output, hidden = self.rnn(packed)
        x = self.dropout(torch.squeeze(hidden))
        x = self.fc(x)

        x = torch.squeeze(self.sigmoid(x))
        return x


class OpenAttackMNB(oa.Classifier):
    def __init__(self, RNBModel, vectorizer):
        self.model = RNBModel
        self.vectorizer = vectorizer

    def get_pred(self, input_):

        res = self.model.predict(self.vectorizer.transform(input_))
        if len(res.shape) > 1:
            return np.squeeze(np.array(res), axis=-1)
        return res
    # access to the classification probability scores with respect input sentences

    def get_prob(self, input_):
        return self.model.predict_proba(self.vectorizer.transform(input_))


class OpenAttackGRU(oa.Classifier):
    def __init__(self, GRUModel, vectorizer, preprocess, batch_size=64, device=torch.device("cuda")):
        self.model = GRUModel
        self.vectorizer = vectorizer
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.device = device
        self.vectorizer.enable_padding(
            pad_id=1, pad_type_id=1, pad_token='[UNK]')
        self.model = self.model.to(device)

    def get_pred(self, input_):

        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty(len(input_), dtype=torch.int8)
        for i in range(iter_range):
            chunk = input_[i*self.batch_size:(i+1)*self.batch_size]
            chunk = [self.preprocess(x) for x in chunk]
            chunk_output = self.vectorizer.encode_batch(chunk)
            chunk = [i.ids for i in chunk_output]
            chunk = torch.IntTensor(chunk)
            length = torch.tensor([sum(i.attention_mask)
                                  for i in chunk_output])
            chunk = chunk.to(self.device)

            chunk = self.model(chunk, length)
            logits = torch.round(chunk)
            result[i*self.batch_size:(i+1)*self.batch_size] = logits
        return result.detach().numpy()
    # access to the classification probability scores with respect input sentences

    def get_prob(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty(len(input_), dtype=torch.float64)
        for i in range(iter_range):

            chunk = input_[i*self.batch_size:(i+1)*self.batch_size]
            chunk = [self.preprocess(x) for x in chunk]
            chunk_output = self.vectorizer.encode_batch(chunk)
            chunk = [i.ids if len(i.ids) > 0 else [1] for i in chunk_output]
            chunk = torch.IntTensor(chunk)
            length = torch.tensor([sum(i.attention_mask) if sum(
                i.attention_mask) > 0 else 1 for i in chunk_output])
            chunk = chunk.to(self.device)
            logit = self.model(chunk, length)
            result[i*self.batch_size:(i+1)*self.batch_size] = logit
        inverse = torch.ones(len(input_))
        inverse -= result
        res = torch.stack([inverse, result], dim=1)
        return res.detach().numpy()


class OpenAttackGRUMultiClasses(oa.Classifier):
    def __init__(self, GRUModel, vectorizer, preprocess, batch_size=64, device=torch.device("cuda"), num_classes=5):
        self.model = GRUModel
        self.vectorizer = vectorizer
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.device = device
        self.vectorizer.enable_padding(
            pad_id=1, pad_type_id=1, pad_token='[UNK]')
        self.model = self.model.to(device)
        self.num_classes = num_classes

    def get_pred(self, input_):

        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty(len(input_), dtype=torch.int8)
        for i in range(iter_range):
            chunk = input_[i*self.batch_size:(i+1)*self.batch_size]
            chunk = [self.preprocess(x) for x in chunk]
            chunk_output = self.vectorizer.encode_batch(chunk)
            chunk = [i.ids for i in chunk_output]
            chunk = torch.IntTensor(chunk)
            length = torch.tensor([sum(i.attention_mask)
                                  for i in chunk_output])
            chunk = chunk.to(self.device)

            chunk = self.model(chunk, length)
            logits = torch.argmax(chunk, axis=-1)
            result[i*self.batch_size:(i+1)*self.batch_size] = logits
        return result.detach().numpy()
    # access to the classification probability scores with respect input sentences

    def get_prob(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty(
            (len(input_), self.num_classes), dtype=torch.float64)
        for i in range(iter_range):

            chunk = input_[i*self.batch_size:(i+1)*self.batch_size]
            chunk = [self.preprocess(x) for x in chunk]
            chunk_output = self.vectorizer.encode_batch(chunk)
            chunk = [i.ids if len(i.ids) > 0 else [1] for i in chunk_output]
            chunk = torch.IntTensor(chunk)
            length = torch.tensor([sum(i.attention_mask) if sum(
                i.attention_mask) > 0 else 1 for i in chunk_output])
            chunk = chunk.to(self.device)
            logit = self.model(chunk, length)
            result[i*self.batch_size:(i+1)*self.batch_size] = logit

        return result.detach().numpy()


def loadGRUModel(path, classes=2, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    GRUModel = torch.load(f"{path}/GRU.pth", map_location)
    vectorizer = Tokenizer.from_file(f"{path}/tokenizer.json")
    if classes == 2:
        model = OpenAttackGRU(GRUModel, vectorizer,
                              clean_text_imdb, device=map_location)
    else:
        model = OpenAttackGRUMultiClasses(
            GRUModel, vectorizer, clean_text_imdb, num_classes=classes, device=map_location)
    return model


if __name__ == "__main__":
    GRUModel = torch.load(
        "/home/ubuntu/Robustness_Gym/model/weights/GRU/IMDB/GRU.pth")
    vectorizer = Tokenizer.from_file(
        "/home/ubuntu/Robustness_Gym/model/weights/GRU/IMDB/tokenizer.json")
    transform = ProcessFuntionIMDBInference()
    model = OpenAttackGRU(GRUModel, vectorizer, clean_text_imdb)
    print(model.get_pred(["Hello", "How are you"]))
    print(model.get_prob(["Hello", "How are you"]))
