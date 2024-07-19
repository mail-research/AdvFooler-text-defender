import torch.nn as nn
import torch
from simpletransformers.classification import ClassificationModel
import numpy as np
import OpenAttack as oa
from time import time
class OpenAttackBert(oa.Classifier):
    def __init__(self, bert_model_path, batch=32, num_classes=2):
        """
        `self.model = ClassificationModel('bert', bert_model_path, use_cuda=True if
        torch.cuda.is_available() else False, args={"silent": True})`
        
        This is the main function that loads the pretrained BERT model. The first argument is the model
        type, which is `bert` in this case. The second argument is the path to the pretrained BERT
        model. The third argument is the `use_cuda` flag, which is set to `True` if a GPU is available.
        The fourth argument is the `args` argument, which is a dictionary of arguments that are passed
        to the model. In this case, we set the `silent` argument to `True` to suppress the output of the
        model
        
        :param bert_model_path: The path to the pre-trained BERT model
        :param batch: The batch size to use for training, defaults to 32 (optional)
        :param num_classes: The number of classes in the dataset, defaults to 2 (optional)
        """
        self.model = ClassificationModel(
            'bert',
            bert_model_path,
            use_cuda=True,

            args={"silent": True}
        )
        self.softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=-1)[:,None]
        self.batch_size = batch
        self.num_classes = num_classes

    def get_prob(self, input_):
        """
        It takes the input data, splits it into batches, and then runs the model on each batch
        
        :param input_: the input data
        :return: The probability of each class.
        """
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty((len(input_), self.num_classes), dtype=torch.float16)
        for i in range(iter_range):
            result[i*self.batch_size:(i+1)*self.batch_size] = torch.FloatTensor(self.softmax(
                self.model.predict(input_[i*self.batch_size:(i+1)*self.batch_size])[1]))
        return result.detach().numpy()

    def get_pred(self, input_):
        """
        It takes a list of input_, and returns a list of predictions
        
        :param input_: the input data
        :return: The classes being predicted.
        """
        iter_range = len(input_)//self.batch_size
        if len(input_) % self.batch_size != 0:
            iter_range += 1
        result = torch.empty(len(input_), dtype=torch.int8)
        for i in range(iter_range):
            result[i*self.batch_size:(i+1)*self.batch_size] = torch.IntTensor(
                self.model.predict(input_[i*self.batch_size:(i+1)*self.batch_size])[0])
        return result.detach().numpy()


if __name__ == "__main__":
    
    # Create a TransformerModel
    """model = ClassificationModel(
        "bert",
        "/home/ubuntu/Robustness_Gym/model/weights/BERT/IMDB/checkpoint-4692-epoch-6",
        use_cuda=True,
    )

    start = time()
    print(model.predict(["test " * 450]))
    end = time()
    print(f"Pytorch CPU: {end - start}")


    model = ClassificationModel(
        "bert",
        "/home/ubuntu/Robustness_Gym/model/weights/BERT/IMDB/ONNX",
        use_cuda=True,
    )

    start = time()
    print(model.predict(["test " * 450]))
    end = time()
    print(f"ONNX CPU (Cold): {end - start}")

    start = time()
    print(model.predict(["test " * 450]))
    end = time()
    print(f"ONNX CPU (Warm): {end - start}")
"""
    model = ClassificationModel(
        "bert",
        "/home/ubuntu/Robustness_Gym/model/weights/BERT/IMDB/ONNX",
        use_cuda=True,
    )
    mold =  torch.empty((2, 2), dtype=torch.float16)
    softmax = lambda x: np.exp(x)/np.sum(np.exp(x), axis=-1)[:,None]
    mold[0:2] =torch.FloatTensor(softmax(model.predict(["test " * 450,"test " * 450])[1]))
    print(mold)
