import torch
from transformers import *
import onnxruntime
import OpenAttack as oa
import sys
from transformers import *
import sys
from time import time
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/ubuntu/mt-dnn')
from mt_dnn.matcher import SANBertNetwork

import numpy as np
class OpenAttackSMARTBERT(oa.Classifier):
    def __init__(self, path, batch_size=64, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), num_classes=2):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        state_dict = torch.load(path, map_location=device)
        self.bert = SANBertNetwork(state_dict["config"],state_dict)
        self.bert.load_state_dict(
                state_dict["state"], strict=False
            )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.batch_size = batch_size
        self.num_classes = num_classes
    def get_pred(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty((len(input_)), dtype=torch.float16)
        for i in range(iter_range):
            inputs = self.tokenizer.batch_encode_plus(input_[i*self.batch_size:(i+1)*self.batch_size],padding="max_length",max_length=512,truncation= True)
            output = self.bert.forward(torch.IntTensor(inputs["input_ids"]), torch.IntTensor(inputs["token_type_ids"]), torch.IntTensor(inputs["attention_mask"]))
            result[i*self.batch_size:(i+1)*self.batch_size] = torch.argmax(output,axis=-1)
        return result.detach().numpy()
    def get_prob(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty((len(input_),self.num_classes), dtype=torch.float16)
        for i in range(iter_range):
            inputs = self.tokenizer.batch_encode_plus(input_[i*self.batch_size:(i+1)*self.batch_size],padding="max_length",max_length=512,truncation= True)
            output = self.bert.forward(torch.IntTensor(inputs["input_ids"]), torch.IntTensor(inputs["token_type_ids"]), torch.IntTensor(inputs["attention_mask"]))
            result[i*self.batch_size:(i+1)*self.batch_size] = self.softmax(output)
        return result.detach().numpy()
#################################################################################
class OpenAttackSMARTBERTONNX(oa.Classifier):
    def __init__(self, path, batch_size=128, device=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'], num_classes=2):
        providers = [
            ('CUDAExecutionProvider', {
        'device_id': 3,
    }),         
            'CPUExecutionProvider'
        ]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = onnxruntime.InferenceSession(path,providers=providers)
        print(self.model.get_providers())
        self.softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=-1)[:,None]
        self.batch_size = batch_size
        self.num_classes = num_classes
    def get_pred(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = np.zeros((len(input_)))
        input_dict={}
        for i in range(iter_range):
            inputs = self.tokenizer.batch_encode_plus(input_[i*self.batch_size:(i+1)*self.batch_size],padding="max_length",max_length=512,truncation= True)
            for k in inputs.keys():
                input_dict[k] = np.array(inputs[k])
            output = self.model.run(None, input_dict)[0]
            result[i*self.batch_size:(i+1)*self.batch_size] = np.argmax(output,axis=-1)
        return result
    def get_prob(self, input_):
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = np.zeros((len(input_),self.num_classes))
        input_dict={}
        for i in range(iter_range):
            inputs = self.tokenizer.batch_encode_plus(input_[i*self.batch_size:(i+1)*self.batch_size],padding="max_length",max_length=512,truncation= True)
            for k in inputs.keys():
                input_dict[k] = np.array(inputs[k])
            output = self.model.run(None, input_dict)[0]
            result[i*self.batch_size:(i+1)*self.batch_size] = self.softmax(output)
        return result
if __name__ == "__main__":
    # Loading the tokenizer from the pretrained model.
    """model = OpenAttackSMARTBERT()
    print(model.get_prob(["hello","hello","hello","hello"]))
    print(model.get_pred(["hello","hello","hello","hello"]))
    
    """
    state_dict = torch.load("/home/ubuntu/mt-dnn/checkpoint_imdb/model_4.pt", map_location=torch.device("cpu"))
    bert = SANBertNetwork(state_dict["config"],state_dict)
    bert.load_state_dict(
            state_dict["state"], strict=False
        )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dummy_model_input = tokenizer("This is a sample", return_tensors="pt")
    torch.onnx.export(
        bert, 
        tuple(dummy_model_input.values()),
        f="/home/ubuntu/mt-dnn/checkpoint_imdb/IMDB.onnx",  
        input_names=['input_ids', 'token_type_ids','attention_mask'], 
        output_names=['logits'], 
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                    'token_type_ids': {0: 'batch_size', 1: 'sequence'}, 
                    'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                    'logits': {0: 'batch_size', 1: 'sequence'}}, 
        do_constant_folding=True, 
        opset_version=13, 
    )
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    state_dict = torch.load("/home/ubuntu/mt-dnn/checkpoint_yelp/model_1.pt", map_location=torch.device("cpu"))
    bert1 = SANBertNetwork(state_dict["config"],state_dict)
    bert1.load_state_dict(
            state_dict["state"], strict=False
        )
    dummy_model_input = tokenizer.batch_encode_plus(["test " * 450,"test " * 450,"test " * 450,"test " * 450,"test " * 450],padding="max_length",max_length=512,truncation= True)
    print(dummy_model_input)
    start = time()
    print(bert1.forward(torch.IntTensor(dummy_model_input["input_ids"]), torch.IntTensor(dummy_model_input["token_type_ids"]), torch.IntTensor(dummy_model_input["attention_mask"])))
    end = time()
    print(f"Pytorch CPU: {end - start}")

    
    
    ort_session = onnxruntime.InferenceSession("/home/ubuntu/mt-dnn/checkpoint_yelp/IMDB.onnx",providers = [
           
            'CPUExecutionProvider'
        ])
    
    print(ort_session.get_inputs())
    input_dict = {}
    for i in dummy_model_input.keys():
        input_dict[i] = np.array(dummy_model_input[i])

    start = time()
    print(ort_session.run(None, input_dict))
    end = time()
    print(f"ONNX CPU (Cold): {end - start}")

    start = time()
    print(ort_session.run(None, input_dict))
    end = time()
    print(f"ONNX CPU (Warm): {end - start}")

    print(model.get_prob(["hello","hello","hello","hello"]))
    print(model.get_pred(["hello","hello","hello","hello"]))"""