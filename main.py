import os

os.environ['CURL_CA_BUNDLE'] = ''
from textattack.datasets import HuggingFaceDataset
from typing import List, Dict
import random
from textattack.attack_recipes import (
    TextFoolerJin2019,
    BERTAttackLi2020,
    HotFlipEbrahimi2017,
    DeepWordBugGao2018,
    TextBuggerLi2018,
    HardLabelMaheshwary2021
)
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import (
    UniversalSentenceEncoder,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification

from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, WordSwapMaskedLM
from utils.dataloader import load_train_test_imdb_data

from model.BERTNoiseDefend import BertForSequenceClassification

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import (
    ModelWrapper,
    PyTorchModelWrapper,
    SklearnModelWrapper,
    HuggingFaceModelWrapper,
)
import torch
import argparse
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
import datasets
import numpy as np
import os

MAX_TOKEN_DICT = {
    'ag_news': 128,
    'imdb': 256
}

from textattack.transformations import CompositeTransformation

class RandomCompositeTransformation(CompositeTransformation):
    """A transformation which applies each of a list of transformations,
    returning a set of all optoins.

    Args:
        transformations: The list of ``Transformation`` to apply.
    """

    def __init__(self, transformations, total_count=20):
        super().__init__(transformations)
        self.total_count = total_count

    def __call__(self, *args, **kwargs):
        new_attacked_texts = set()
        transformation_num = len(self.transformations)
        if transformation_num <= 0:
            raise ValueError
        index = np.random.choice(transformation_num, self.total_count, replace=True)

        for i in index:
            new_attacked_texts.update(self.transformations[i](*args, **kwargs))
        return list(new_attacked_texts)

class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self, model, tokenizer):
        super(CustomModelWrapper, self).__init__(model, tokenizer)

    def __call__(self, text_input_list):
        inputs_dict = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs, tuple):
            return outputs[-1]  # model-h,model-bh

        if isinstance(outputs, torch.Tensor):
            return outputs  # baseline

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

def build_attacker_from_textdefender(model: HuggingFaceModelWrapper,args) -> Attack:
    if args["attack_method"] == 'hotflip':
        attacker = HotFlipEbrahimi2017.build(model)
    if args["attack_method"] == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args["attack_method"] == 'bertattack':
        attacker = BERTAttackLi2020.build(model)
        attacker.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=args["k_neighbor"])
    elif args["attack_method"] == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args["attack_method"] == 'hardlabel':
        attacker = HardLabelMaheshwary2021.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)

    if args["attack_method"] in ['textfooler', 'pwws', 'textbugger', 'pso']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args["k_neighbor"])
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance) or isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)
                
    attacker.constraints.append(MaxWordsPerturbed(max_percent=args["modify_ratio"]))
    use_constraint = UniversalSentenceEncoder(
        threshold=args["similarity"],
        metric="cosine"
    )
    attacker.constraints.append(use_constraint)
    print(attacker.constraints)
    attacker.goal_function = UntargetedClassification(model, query_budget=args["k_neighbor"])
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)

def build_attacker(model, args):
    if args["attack_method"] == "textfooler":
        attacker = TextFoolerJin2019.build(model)
    elif args["attack_method"] == "textbugger":
        attacker = TextBuggerLi2018.build(model)
    elif args["attack_method"] == "deepwordbug":
        attacker = DeepWordBugGao2018.build(model)
    elif args["attack_method"] == "bertattack":
        attacker = BERTAttackLi2020.build(model)
    
    else:
        attacker = TextFoolerJin2019.build(model)
    
    if args["modify_ratio"] != 0:
        print(args["modify_ratio"])
        attacker.constraints.append(MaxWordsPerturbed(max_percent=args["modify_ratio"]))
    if args["similarity"] != 0:
        print("AGHHHHHHHHHHHHHH")
        attacker.constraints.append(
            UniversalSentenceEncoder(threshold=args["similarity"], metric="cosine")
        )
    return attacker


def str2bool(strIn):
    if strIn.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif strIn.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True


def attack(args, wrapper, name, dataset):
    attackMethod = args.attack_method
    attack_args_dict = {
        "attack_method": attackMethod,
        "attack_times": 1,
        "attack_examples": int(args.attack_examples),
        "modify_ratio": float(args.modify_ratio),
        "similarity": float(args.similarity),
        "k_neighbor": int(args.k_neighbor),
        "log_path": "{}/{}/{}_{}-{}.txt".format(
            args.load_path, attackMethod, args.attack_dataset, args.modify_ratio, name
        ),
    }
    attack = build_attacker_from_textdefender(wrapper, attack_args_dict)
    attack_args = AttackArgs(
        num_examples=attack_args_dict["attack_examples"],
        log_to_txt=attack_args_dict["log_path"],
        csv_coloring_style="file",
        num_workers_per_device=int(args.num_workers_per_device),
        parallel=args.parallel
    )
    attacker = Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()


def gen_dataset(instances):
    test_instances = instances
    test_dataset = []
    for instance in range(len(test_instances)):
        test_dataset.append(
            (test_instances["text"][instance], int(test_instances["label"][instance]))
        )
    dataset = Dataset(test_dataset, shuffle=True)
    return dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument(
        "-da", "--dataset", default="agnews"
    ) 
    parser.add_argument("-a", "--attack", type=str2bool, nargs="?", const=False)
    parser.add_argument("-lp", "--load_path", default="textattack/bert-base-uncased-ag-news")
    parser.add_argument("-am", "--attack_method", default="textfooler")
    parser.add_argument(
        "-ad", "--attack_dataset", default="test"
    )  # attack dataset & accuracy dataset
    parser.add_argument("-ae", "--attack_examples", default=1000)
    parser.add_argument("-dp", "--def_position", default="post_att_cls")
    parser.add_argument("-ni", "--noise_intensity", default=0.8)
    parser.add_argument("-mr", "--modify_ratio", default=0.3)
    parser.add_argument("-sm", "--similarity", default=0.84)
    parser.add_argument("-kn", "--k_neighbor", default=50)
    parser.add_argument("-nd", "--num_workers_per_device", default=2)
    parser.add_argument('-pr', '--parallel',action='store_true')
    parser.add_argument("-en", "--ensemble_num", default=100, type=int)
    parser.add_argument("-eb", "--ensemble_batch_size", default=32)
    parser.add_argument("-rmr", "--random_mask_rate", default=0.9)
    parser.add_argument("-spf", "--safer_pertubation_file", default="/home/ubuntu/TextDefender/dataset/ag_news/perturbation_constraint_pca0.8_100.pkl")
    
    args = parser.parse_args()
    sst2_dataset = datasets.load_dataset(args.dataset)
    test_data = sst2_dataset[args.attack_dataset]

    tokenizer = AutoTokenizer.from_pretrained(
        args.load_path, use_fast=True
    )
    tokenizer.model_max_length=MAX_TOKEN_DICT[args.dataset]

    
    config = AutoConfig.from_pretrained(args.load_path)
    model = BertForSequenceClassification(config)
    state = AutoModelForSequenceClassification.from_pretrained(
        args.load_path
    )
    model.load_state_dict(state.state_dict(), strict=False)
    model.to("cuda")
    model.eval()
    BERT = HuggingFaceModelWrapper(model, tokenizer)
    
   
    
    with torch.no_grad():

        noise_pos = {"post_att_cls": [0.8]}                
        set_seed(1)
        dataset = gen_dataset(test_data)
        args.load_path = (
            f"noise_defense_attack_result/test_3/{args.dataset.upper()}/{2}/"
        )
        model.change_defense(defense_cls="random_noise",def_position=args.def_position,noise_sigma=args.noise_intensity,defense=True)
        attack(args, BERT, f"BERT_{args.def_position}_{args.noise_intensity}", dataset)
                