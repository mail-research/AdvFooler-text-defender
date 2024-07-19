import re
import torch
def clean_text_imdb(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text.strip().lower()

class TextPadding(torch.nn.Module):
    def __init__(self,pad_num):
        super().__init__()
        self.pad_num = pad_num
    def forward(self, x):
        max_len = max([len(i) for i in x])

        ret = torch.ones((len(x),max_len))*self.pad_num
        for i in range(len(x)):
            ret[i][:len(x[i])] = torch.LongTensor(x[i])
        ret = ret.type(torch.LongTensor)
        return ret
class ProcessFuntionIMDBInference(torch.nn.Module):
    def __init__(self,tensor_padding=TextPadding(1)):
        super().__init__()
        self.tensor_padding = tensor_padding
    def forward(self, x):
        texts = []
        lengths = []
        for i in x:
            text = i
            texts.append(text)
            lengths.append(len(text))
        texts = self.tensor_padding(texts)
        return texts,torch.FloatTensor(lengths)
 