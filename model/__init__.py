from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaPreTrainedModel, RobertaModel
from utils.dne_utils import DecayAlphaHull,WeightedEmbedding,get_bert_vocab,get_roberta_vocab
from utils.certified import ibp_utils
from utils.luna import batch_pad
from .generative_models import *
from .language_models import *
import argparse
MODEL_CLASSES = {
    # Note: there may be some bug in `dcnn` modeling, if you want to pretraining.
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    'electra': (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
}
DNE_SETUP = {
    'agnews':{
            "dir_alpha":1.0,
            "dir_decay":0.5,
            "nbr_num":50
            },
    'imdb':{
            "dir_alpha":0.1,
            "dir_decay":0.1,
            "nbr_num":50
            }
}
DATASET_LABEL_NUM = {
    'sst2': 2,
    'agnews': 4,
    'imdb': 2,
    'mr': 2,
    'onlineshopping': 2,
    'snli': 3,
    'yelp':2
}



LABEL_MAP = {
    'nli': {'entailment': 0, 'contradiction': 1, 'neutral': 2},
    'agnews': {'0': 0, '1': 1, '2': 2, '3': 3},
    'binary': {'0': 0, '1': 1}
}

GLOVE_CONFIGS = {
    '6B.50d': {'size': 50, 'lines': 400000},
    '840B.300d': {'size': 300, 'lines': 2196017}
}


import json
from collections import defaultdict



def string_to_bool(string_val):
    return True if string_val.lower() == 'true' else False



class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.mode = 'attack'  # in ['train', 'attack', 'evaluate', 'augment', 'textattack_augment', 'dev_augment']
        self.model_type = 'bert'
        self.dataset_name = 'agnews'
        self.keep_sentiment_word = 'False'
        self.model_name_or_path = 'bert-base-uncased'
        self.evaluation_data_type = 'test'
        self.training_type = 'freelb'

        # attack parameters
        self.attack_method = 'bae'
        self.attack_times = 1
        self.attack_numbers = 1000
        # attack constraint args defined by us
        self.modify_ratio = 0.3
        self.neighbour_vocab_size = 50
        self.sentence_similarity = 0.840845057
        self.query_budget_size = self.neighbour_vocab_size

        # path parameters
        self.workspace = '/root/TextDefender'
        self.dataset_path = self.workspace + '/dataset/' + self.dataset_name
        # self.log_path = self.workspace + '/log/' + self.dataset_name + '_' + self.model_type
        self.cache_path = self.workspace + '/cache'
        # self.saved_path = self.workspace + '/saved_models/' + self.dataset_name
        self.sentiment_path = self.workspace + '/dataset/sentiment_word/sentiment-words.txt'
        self.log_path = self.workspace + "/log"
        self.tensorboard = None

        # augment parameters
        self.use_aug = 'False'
        self.aug_ratio = 0.5
        self.aug_attacker = 'pwws'

        self.dev_aug_ratio = 0.5
        self.dev_aug_attacker = 'textfooler'
        self.use_dev_aug = 'False'

        # text_attack augment parameters
        self.split_num = 3
        self.start_idx = 0

        # model ensemble num in predicting (if needed)
        self.ensemble = 'False'
        self.ensemble_num = 100
        self.ensemble_method = 'logits' # in ['logits', 'votes']

        # base training hyper-parameters, if need other, define in subclass
        self.epochs = 10  # training epochs
        if string_to_bool(self.use_aug) and self.aug_ratio == 0.5:
            self.batch_size = 24
        else:
            self.batch_size = 32  # batch size
        # self.gradient_accumulation_steps = 1  # Number of updates steps to accumulate before performing a backward/update pass.
        # self.learning_rate = 5e-5  # The initial learning rate for Adam.
        # self.weight_decay = 1e-6  # weight decay
        # self.adam_epsilon = 1e-8  # epsilon for Adam optimizer
        # self.max_grad_norm = 1.0  # max gradient norm
        # self.learning_rate_decay = 0.1  # Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training

        # read dataset parameter
        if self.dataset_name != 'imdb':
            self.max_seq_len = 128
        else:
            self.max_seq_len = 256
        self.shuffle = 'True'

        # unchanged args
        self.type_accept_instance_as_input = ['mask', 'safer']
        # self.imdb_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/aclImdb'
        # self.imdb_lm_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/lm_scores/imdb_all.txt'
        # self.counter_fitted_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/counter-fitted-vectors.txt'
        # self.snli_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/snli'
        # self.snli_lm_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/lm_scores/snli_all.txt'
        # self.neighbor_file = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/counterfitted_neighbors.json'
        self.nbr_file = '/root/TextDefender/counterfitted_neighbors.json' 
        # self.glove_dir = '/disks/sdb/lzy/adversarialBenchmark/IBP_data/glove'
        self.do_lower_case = 'True'
        # for lstm
        self.hidden_size = 100
        self.glove_name = '840B.300d'
        self.use_lm = 'False'

        # saving args
        self.saving_step = 1
        self.saving_last_epoch = 'False'
        self.compare_key = '+accuracy'
        self.file_name = None
        self.seed = 42
        self.remove_attack_constrainst = 'False'

        # tmd args
        self.gm = "infogan"
        self.gm_path = "/root/manifold_defense/outputs/infogan_bert_imdb/manifold-defense/yutbyyz5/checkpoints/epoch=99-step=2199.ckpt"
        self.tmd_layer = -1
        self.start_index = 0
        self.end_index = None
        self.method = 3
        self.threshold = 1.0
        self.step_size = 0.01
        #self.num_steps = 10
        self.k = 30

infogan_args = ProgramArgs()

class LSTMModel(nn.Module):
    """LSTM text classification model.

    Here is the overall architecture:
      1) Rotate word vectors
      2) Feed to bi-LSTM
      3) Max/mean pool across all time
      4) Predict with MLP

    """

    def __init__(self, word_vec_size, hidden_size, word_mat, device, num_labels=2, pool='mean',
                 dropout=0.2, no_wordvec_layer=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.pool = pool
        self.no_wordvec_layer = no_wordvec_layer
        self.device = device
        self.embs = ibp_utils.Embedding.from_pretrained(word_mat)
        if no_wordvec_layer:
            self.lstm = ibp_utils.LSTM(word_vec_size, hidden_size, bidirectional=True)
        else:
            self.linear_input = ibp_utils.Linear(word_vec_size, hidden_size)
            self.lstm = ibp_utils.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.dropout = ibp_utils.Dropout(dropout)
        self.fc_hidden = ibp_utils.Linear(2 * hidden_size, hidden_size)
        self.fc_output = ibp_utils.Linear(hidden_size, num_labels)

    def forward(self, batch, compute_bounds=True, cert_eps=1.0):
        """
        Args:
          batch: A batch dict from a TextClassificationDataset with the following keys:
            - x: tensor of word vector indices, size (B, n, 1)
            - mask: binary mask over words (1 for real, 0 for pad), size (B, n)
            - lengths: lengths of sequences, size (B,)
          compute_bounds: If True compute the interval bounds and reutrn an IntervalBoundedTensor as logits. Otherwise just use the values
          cert_eps: Scaling factor for interval bounds of the input
        """
        if type(batch) != tuple:
            B = len(batch)
            x = batch.view(B, -1, 1)
            mask = batch != 1
            mask = mask.long()
            lengths = mask.sum(dim=1)

        else:
            if compute_bounds:
                x = batch[0]
            else:
                x = batch[0].val
            mask = batch[1]
            lengths = batch[2]
            B = x.shape[0]

        x_vecs = self.embs(x)  # B, n, d
        if not self.no_wordvec_layer:
            x_vecs = self.linear_input(x_vecs)  # B, n, h
        if isinstance(x_vecs, ibp_utils.DiscreteChoiceTensor):
            x_vecs = x_vecs.to_interval_bounded(eps=cert_eps)
        if self.no_wordvec_layer:
            z = x_vecs
        else:
            z = ibp_utils.activation(F.relu, x_vecs)  # B, n, h
        h0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        c0 = torch.zeros((B, 2 * self.hidden_size), device=self.device)  # B, 2*h
        h_mat, c_mat = self.lstm(z, (h0, c0), mask=mask)  # B, n, 2*h each
        h_masked = h_mat * mask.unsqueeze(2)
        if self.pool == 'mean':
            fc_in = ibp_utils.sum(h_masked / lengths.to(dtype=torch.float).view(-1, 1, 1), 1)  # B, 2*h
        else:
            raise NotImplementedError()
        fc_in = self.dropout(fc_in)
        fc_hidden = ibp_utils.activation(F.relu, self.fc_hidden(fc_in))  # B, h
        fc_hidden = self.dropout(fc_hidden)
        output = self.fc_output(fc_hidden)  # B, num_labels

        return output


## if using BERT, just switch to BertModel4Mix in the MixText model
## and change name to self.bert

class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None,
                hidden_states2=None, attention_mask2=None,
                l=None, mix_layer=1000, head_mask=None):
        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None

        # Perform mix till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1 - l) * hidden_states2

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i <= mix_layer:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            elif i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1 - l) * hidden_states2
                    attention_mask = attention_mask.long() | attention_mask2.long()
                    ## sentMix: (bsz, len, hid)
                    # hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1-l)*hidden_states2[:, 0, :]
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs


class BertModel4Mix(BertPreTrainedModel, nn.Module):

    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    # def _resize_token_embeddings(self, new_num_tokens):
    #     old_embeddings = self.embeddings.word_embeddings
    #     new_embeddings = self._get_resized_embeddings(
    #         old_embeddings, new_num_tokens)
    #     self.embeddings.word_embeddings = new_embeddings
    #     return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000, head_mask=None):

        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0
            embedding_output2 = self.embeddings(input_ids2, token_type_ids=token_type_ids2)
            encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                                           embedding_output2, extended_attention_mask2,
                                           l, mix_layer, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, embedding_output) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class MixText(BertPreTrainedModel, nn.Module):
    def __init__(self, config):
        super(MixText, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel4Mix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_ids2=None, attention_mask2=None, token_type_ids2=None,
                l=None, mix_layer=1000):

        if input_ids2 is not None:
            outputs = self.bert(input_ids, attention_mask, token_type_ids,
                                input_ids2, attention_mask2, token_type_ids2,
                                l, mix_layer)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(input_ids, attention_mask, token_type_ids)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        return logits, outputs
    
class ASCCRobertaModel(RobertaPreTrainedModel, nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def build_nbrs(self, nbr_file, vocab, alpha, num_steps,device):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        loaded = json.load(open(nbr_file))
        filtered = defaultdict(lambda: [], {})
        for k in loaded:
            if k in t2i:
                for v in loaded[k]:
                    if v in t2i:
                        filtered[k].append(v)
        nbrs = dict(filtered)

        nbr_matrix = []
        vocab_size = vocab.get_vocab_size("tokens")
        for idx in range(vocab_size):
            token = vocab.get_token_from_index(idx)
            nbr = [idx]
            if token in nbrs.keys():
                words = nbrs[token]
                for w in words:
                    assert w in t2i
                    nbr.append(t2i[w])
            nbr_matrix.append(nbr)
        nbr_matrix = batch_pad(nbr_matrix)
        self.nbrs = torch.tensor(nbr_matrix).to(device)
        self.max_nbr_num = self.nbrs.size()[-1]
        # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
        #                                     requires_grad=True).cuda()
        self.weighting_mask = self.nbrs != 0
        self.criterion_kl = nn.KLDivLoss(reduction="sum")
        self.alpha = alpha
        self.num_steps = num_steps

    def forward(self, input_ids, attention_mask):
        clean_outputs = self.roberta(input_ids, attention_mask)
        clean_sequence_output = clean_outputs[0]
        clean_logits = self.classifier(clean_sequence_output)
        return clean_logits, clean_logits

        # 0 initialize w for neighbor weightings
        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, self.max_nbr_num, 1).to(self.device).to(torch.float)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        optimizer_w = torch.optim.Adam([w], lr=1, weight_decay=2e-5)

        # 1 forward and backward to calculate adv_examples
        input_nbr_embed = self.get_input_embeddings()(self.nbrs[input_ids])
        weighting_mask = self.weighting_mask[input_ids]
        # here we need to calculate clean logits with no grad, to find adv examples
        with torch.no_grad():
            clean_outputs = self.roberta(input_ids, attention_mask)
            clean_sequence_output = clean_outputs[0]
            clean_logits = self.classifier(clean_sequence_output)

        for _ in range(self.num_steps):
            optimizer_w.zero_grad()
            with torch.enable_grad():
                w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
                embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

                adv_outputs = self.roberta(attention_mask=attention_mask, inputs_embeds=embed_adv)
                adv_sequence_output = adv_outputs[0]
                adv_logits = self.classifier(adv_sequence_output)

                adv_loss = - self.criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(clean_logits.detach(), dim=1))
                loss_sparse = (-F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1) * F.log_softmax(w_after_mask, -2)).sum(-2).mean()
                loss = adv_loss + self.alpha * loss_sparse

            loss.backward(retain_graph=True)
            optimizer_w.step()

        optimizer_w.zero_grad()
        self.zero_grad()

        # 2 calculate clean data logits
        clean_outputs = self.roberta(input_ids, attention_mask)
        clean_sequence_output = clean_outputs[0]
        clean_logits = self.classifier(clean_sequence_output)

        # 3 calculate convex hull of each embedding
        w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
        embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

        # 4 calculate adv logits
        adv_outputs = self.roberta(attention_mask=attention_mask, inputs_embeds=embed_adv)
        adv_sequence_output = adv_outputs[0]
        adv_logits = self.classifier(adv_sequence_output)

        return clean_logits, adv_logits

class ASCCModel(BertPreTrainedModel, nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_embeddings = self.get_input_embeddings().num_embeddings

        self.init_weights()

    def build_nbrs(self, nbr_file, vocab, alpha, num_steps,device="gpu"):
        t2i = vocab.get_token_to_index_vocabulary("tokens")
        loaded = json.load(open(nbr_file))
        filtered = defaultdict(lambda: [], {})
        for k in loaded:
            if k in t2i:
                for v in loaded[k]:
                    if v in t2i:
                        filtered[k].append(v)
        nbrs = dict(filtered)

        nbr_matrix = []
        vocab_size = vocab.get_vocab_size("tokens")
        for idx in range(vocab_size):
            token = vocab.get_token_from_index(idx)
            nbr = [idx]
            if token in nbrs.keys():
                words = nbrs[token]
                for w in words:
                    assert w in t2i
                    nbr.append(t2i[w])
            nbr_matrix.append(nbr)
        nbr_matrix = batch_pad(nbr_matrix)
        self.nbrs = torch.tensor(nbr_matrix).to(device)
        self.max_nbr_num = self.nbrs.size()[-1]
        # self.weighting_param = nn.Parameter(torch.empty([self.num_embeddings, self.max_nbr_num], dtype=torch.float32),
        #                                     requires_grad=True).cuda()
        self.weighting_mask = self.nbrs != 0
        self.criterion_kl = nn.KLDivLoss(reduction="sum")
        self.alpha = alpha
        self.num_steps = num_steps

    def forward(self, input_ids, attention_mask, token_type_ids):
        clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_clean_output = self.dropout(clean_outputs[1])
        clean_logits = self.classifier(pooled_clean_output)
        return clean_logits, clean_logits

        # 0 initialize w for neighbor weightings
        batch_size, text_len = input_ids.shape
        w = torch.empty(batch_size, text_len, self.max_nbr_num, 1).to(self.device).to(torch.float)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        optimizer_w = torch.optim.Adam([w], lr=1, weight_decay=2e-5)

        # 1 forward and backward to calculate adv_examples
        input_nbr_embed = self.get_input_embeddings()(self.nbrs[input_ids])
        weighting_mask = self.weighting_mask[input_ids]
        # here we need to calculate clean logits with no grad, to find adv examples
        with torch.no_grad():
            clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
            pooled_clean_output = self.dropout(clean_outputs[1])
            clean_logits = self.classifier(pooled_clean_output)

        for _ in range(self.num_steps):
            optimizer_w.zero_grad()
            with torch.enable_grad():
                w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
                embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)
                adv_outputs = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                        inputs_embeds=embed_adv)
                pooled_adv_output = self.dropout(adv_outputs[1])
                adv_logits = self.classifier(pooled_adv_output)
                adv_loss = - self.criterion_kl(F.log_softmax(adv_logits, dim=1),
                                           F.softmax(clean_logits.detach(), dim=1))
                loss_sparse = (-F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1) * F.log_softmax(w_after_mask, -2)).sum(-2).mean()
                loss = adv_loss + self.alpha * loss_sparse

            loss.backward(retain_graph=True)
            optimizer_w.step()

        optimizer_w.zero_grad()
        self.zero_grad()

        # 2 calculate clean data logits
        clean_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_clean_output = self.dropout(clean_outputs[1])
        clean_logits = self.classifier(pooled_clean_output)

        # 3 calculate convex hull of each embedding
        w_after_mask = weighting_mask.unsqueeze(-1) * w + ~weighting_mask.unsqueeze(-1) * -999
        embed_adv = torch.sum(input_nbr_embed * F.softmax(w_after_mask, -2) * weighting_mask.unsqueeze(-1), dim=2)

        # 4 calculate adv logits
        adv_outputs = self.bert(attention_mask=attention_mask, token_type_ids=token_type_ids,
                                inputs_embeds=embed_adv)
        pooled_adv_output = self.dropout(adv_outputs[1])
        adv_logits = self.classifier(pooled_adv_output)


        return clean_logits, adv_logits
ASCC_MODEL = {
    "bert":ASCCModel,
    "roberta":ASCCRobertaModel,
}
VOCAB = {
    "bert": get_bert_vocab,
    "roberta": get_roberta_vocab,

}
def TextDefense_model_builder(model_type,model_name_or_path,training_type,device="cpu",dataset_name="imdb",glove_name=None,hidden_size=None,gm_path = None,):
    
    if training_type == 'mixup':
        config_class, _, _ = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
            model_name_or_path,
            num_labels=DATASET_LABEL_NUM[dataset_name],
            finetuning_task=dataset_name,
            output_hidden_states=True,
        )
        model = MixText.from_pretrained(
            model_name_or_path,
            from_tf=bool('ckpt' in model_name_or_path),
            config=config
        ).to(device)
    elif training_type == 'dne' and model_type == 'bert':
        model_args = DNE_SETUP[dataset_name]
        config_class, model_class, _ = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
            model_name_or_path,
            num_labels=DATASET_LABEL_NUM[dataset_name],
            finetuning_task=dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool('ckpt' in model_name_or_path),
            config=config
        ).to(device)
        bert_vocab = get_bert_vocab()
        hull = DecayAlphaHull.build(
            alpha=model_args["dir_alpha"],
            decay=model_args["dir_decay"],
            nbr_file="model/weights/nbr_file.json",
            vocab=bert_vocab,
            nbr_num=model_args["nbr_num"],
            second_order=True,
            device=device
        )
        model.bert.embeddings.word_embeddings = WeightedEmbedding(
            num_embeddings=bert_vocab.get_vocab_size('tokens'),
            embedding_dim=768,
            padding_idx=model.bert.embeddings.word_embeddings.padding_idx,
            _weight=model.bert.embeddings.word_embeddings.weight,
            hull=hull,
            sparse=False)
    elif training_type == 'dne' and model_type == 'roberta':
        model_args = DNE_SETUP[dataset_name]
        config_class, model_class, _ = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
            model_name_or_path,
            num_labels=DATASET_LABEL_NUM[dataset_name],
            finetuning_task=dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool('ckpt' in model_name_or_path),
            config=config
        ).to(device)
        roberta_vocab = get_roberta_vocab()
        hull = DecayAlphaHull.build(
            alpha=model_args["dir_alpha"],
            decay=model_args["dir_decay"],
            nbr_file="model/weights/nbr_file.json",
            vocab=roberta_vocab,
            nbr_num=model_args["nbr_num"],
            second_order=True,
            device=device
        )
        model.roberta.embeddings.word_embeddings = WeightedEmbedding(
            num_embeddings=roberta_vocab.get_vocab_size('tokens'),
            embedding_dim=768,
            padding_idx=model.roberta.embeddings.word_embeddings.padding_idx,
            _weight=model.roberta.embeddings.word_embeddings.weight,
            hull=hull,
            sparse=False)

    elif training_type == 'ascc':
        config_class, _, _ = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
            model_name_or_path,
            num_labels=DATASET_LABEL_NUM[dataset_name],
            finetuning_task=dataset_name,
            output_hidden_states=True,
        )
        ascc_model = ASCC_MODEL[model_type]
        model = ascc_model.from_pretrained(
            model_name_or_path,
            from_tf=bool('ckpt' in model_name_or_path),
            config=config
        ).to(device)
        bert_vocab = VOCAB[model_type]()
        model.build_nbrs("model/weights/nbr_file.json", bert_vocab, 10.0, 5,device)
    elif training_type == 'tmd':
        print(f"Loading LM: {model_name_or_path}")
        model = AutoLanguageModel.get_class_name(model_type).from_pretrained(model_name_or_path, load_tokenizer=False)
        model.eval()
        model.to(device)
        # Load generative model
        print(f"Loading GM: {'infogan'}")
        gm = AutoGenerativeModel.get_class_name("infogan").load_from_checkpoint(gm_path)
        gm.eval()
        gm.to(device)
        model.set_reconstructor(gm, **vars(infogan_args))
    else:
        config_class, model_class, _ = MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(
            model_name_or_path,
            num_labels=DATASET_LABEL_NUM[dataset_name],
            finetuning_task=dataset_name,
            output_hidden_states=True,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool('ckpt' in model_name_or_path),
            config=config
        ).to(device)
    return model