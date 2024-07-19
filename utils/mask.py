import numpy as np
from typing import List, Dict
import copy
import json


class InputInstance(object):
    '''
        use to store a piece of data
        contains:
            idx:    index of this instance
            text_a: first sentence
            text_b: second sentence (if agnews etc. : default None)
            label:  label (if test set: None)
    '''
    def __init__(self, idx, text_a, text_b=None, label=None):
        self.idx = idx
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def perturbable_sentence(self):
        if self.text_b is None:
            return self.text_a
        else:
            return self.text_b

    def is_nli(self):
        return self.text_b is not None

    def length(self):
        return len(self.perturbable_sentence().split())

    @classmethod
    def create_instance_with_perturbed_sentence(cls, instance: "InputInstance", perturb_sent: str):
        idx = instance.idx
        label = instance.label
        if instance.text_b is None:
            text_a = perturb_sent
            text_b = None
        else:
            text_a = instance.text_a
            text_b = perturb_sent
        return cls(idx, text_a, text_b, label)

def mask_forbidden_index(sentence: str, forbidden_words: List[str]) -> List[int]:
    sentence_in_list = sentence.split()
    forbidden_indexes = []
    for index, word in enumerate(sentence_in_list):
        if word in forbidden_words:
            forbidden_indexes.append(index)
    if len(forbidden_indexes) == 0:
        return None
    else:
        return forbidden_indexes

def sampling_index_loop_nums(length: int,
                             mask_numbers: int, 
                             nums: int, 
                             sampling_probs: List[float] = None) -> List[int]:
    if sampling_probs is not None:
        assert length == len(sampling_probs)
        if sum(sampling_probs) != 1.0:
            sampling_probs = sampling_probs / sum(sampling_probs)
    mask_indexes = []
    for _ in range(nums):
        mask_indexes.append(np.random.choice(list(range(length)), mask_numbers, replace=False, p=sampling_probs).tolist()) 
    return mask_indexes

def mask_instance(instance: InputInstance, 
                  rate: float, 
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden_indexes: List[int] = None, 
                  random_probs: List[float] = None) -> List[InputInstance]:
    sentence = instance.perturbable_sentence()
    results = mask_sentence(sentence, rate, token, nums, return_indexes, forbidden_indexes, random_probs)
    if return_indexes:
        mask_sentences_list = results[0]
    else:
        mask_sentences_list = results
    tmp_instances = [InputInstance.create_instance_with_perturbed_sentence(instance, sent) for sent in mask_sentences_list]
    if return_indexes:
        return tmp_instances, results[1]
    else:
        return tmp_instances


def mask_sentence(sentence: str, 
                  rate: float, 
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden: List[int] = None,
                  random_probs: List[float] = None, 
                  min_keep: int = 2) -> List[str]:
    # str --> List[str]
    sentence_in_list = sentence.split()
    length = len(sentence_in_list)

    mask_numbers = round(length * rate)
    if length - mask_numbers < min_keep:
        mask_numbers = length - min_keep if length - min_keep >= 0 else 0

    mask_indexes = sampling_index_loop_nums(length, mask_numbers, nums, random_probs)
    tmp_sentences = []
    for indexes in mask_indexes:
        tmp_sentence = mask_sentence_by_indexes(sentence_in_list, indexes, token, forbidden)
        tmp_sentences.append(tmp_sentence)
    if return_indexes:
        return tmp_sentences, mask_indexes
    else:
        return tmp_sentences


def mask_sentence_by_indexes(sentence: List[str], indexes: np.ndarray, token: str, forbidden: List[str]=None) -> str:
    tmp_sentence = sentence.copy()
    for index in indexes:
        tmp_sentence[index] = token
    if forbidden is not None:
        for index in forbidden:
            tmp_sentence[index] = sentence[index]
    return ' '.join(tmp_sentence)