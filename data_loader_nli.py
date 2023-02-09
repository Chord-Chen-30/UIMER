from ctypes import alignment
import sys
from pickletools import optimize
from socket import TIPC_MEDIUM_IMPORTANCE
from turtle import setposition
from unittest import TestCase
import pandas as pd
from difflib import Differ
import copy
from regex import B
from transformers import BertForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model import CrossEncoderForWNLI
from transformers import BertTokenizer
from utils import print_cmd, set_seed
from random import shuffle
from tqdm import tqdm
import random
import csv

from Logger import MyLogger
import pdb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class InputExample():
    def __init__(self, index, sentence1, sentence2, label):
        self.index = index
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.rationale = [] # Used for wnli
        self.rationale_word_indices = [] # Used for esnli

        self.data_clean()

    def data_clean(self):
        # Remove "." at the end of sentence
        if self.sentence1.endswith(".") or self.sentence1.endswith(","):
            self.sentence1 = self.sentence1[:-1]
        
        if self.sentence2.endswith(".") or self.sentence2.endswith(","):
            self.sentence2 = self.sentence2[:-1]
        
        self.sentence1 = self.sentence1.split()
        self.sentence2 = self.sentence2.split()

        self.sentence1 = [i.lower() for i in self.sentence1]
        self.sentence2 = [i.lower() for i in self.sentence2]

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, label, weight=-1, augmented_flag=0, annotation_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.weight = weight # Used for augmented samples. Change later
        self.augmented_flag = augmented_flag

        # The 0-1 list indicating rationales w.r.t. input_ids
        self.annotation_ids = annotation_ids



def read_wnli_to_examples(path):
    data = pd.read_csv(path, sep='\t')

    examples = []

    for i, row in data.iterrows():
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        try:
            label = int(row['label'])
        except KeyError: # Label not given in test set
            label = -1

        examples.append(InputExample(i, sentence1, sentence2, label))

    return examples

def read_esnli_to_examples(path=['./data/eSNLI/esnli_train_1.csv','./data/eSNLI/esnli_train_2.csv']):
    data = []
    examples = []

    for _path in path:
        print(_path)

        with open(_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            # print(type(reader))

            for i, row in enumerate(reader):
                if i != 0:
                    data.append(row)

    for i in range(1, len(data)):
        for ind, content in enumerate(data[i]):
            if ind == 1:
                if content == 'entailment':
                    label = 0
                elif content == 'neutral':
                    label = 1
                elif content == 'contradiction':
                    label = 2
                else:
                    print("Unseen Label: ", content)
                    print(i)
                    print(data[i])
                    # exit(-1)
            elif ind == 6:
                sentence1 = content
            elif ind == 7:
                sentence2 = content
        
        examples.append(InputExample(i-1, sentence1, sentence2, label))

    return examples

def find_rationale_wnli(all_examples):
    print("All examples:", len(all_examples))

    # First step, sort w.r.t. the alphabetic order of sentence
    all_examples.sort(key = lambda x: x.sentence1)
    # for i in all_examples:
    #     print(i.sentence1)
    
    # Second step, remove repeat sentence
    example_set = []
    sentence_set = []
    for example in all_examples:
        if example.sentence1 not in sentence_set:
            sentence_set.append(example.sentence1)
            example_set.append(example)

    print("Example set Length:", len(example_set))
    # for i in example_set:
    #     print(i.sentence1)

    # Third step, only keep the pairs remaining
    # Tricky, try finding pairs by looking at the first 10 letters of sentence
    FIRST_LETTER = 13
    pairs_found_debug = []

    pair_flag = 1
    for example in example_set:
        sentence_1_str = ' '.join(example.sentence1)
        sentence_2_str = ' '.join(example.sentence2)
        if pair_flag == 1:
            fisrt_10 = copy.deepcopy(sentence_1_str[:FIRST_LETTER])
            saved_sentence_1 = copy.deepcopy(sentence_1_str)
            saved_sentence_2 = copy.deepcopy(sentence_2_str)
            pair_flag = 2

        elif pair_flag == 2:
            another_first_10 = sentence_1_str[:FIRST_LETTER]
            if another_first_10 == fisrt_10:
                # We found pairs!
                pairs_found_debug.append((saved_sentence_1, saved_sentence_2))
                pairs_found_debug.append((sentence_1_str, sentence_2_str))
                pair_flag = 1

            else:
                # Not a pair
                fisrt_10 = copy.deepcopy(sentence_1_str[:FIRST_LETTER])
                saved_sentence_1 = copy.deepcopy(sentence_1_str)
                saved_sentence_2 = copy.deepcopy(sentence_2_str)
                pair_flag = 2


    # Fourth step, finish sentence_to_rationale
    sentence_to_rationale = {}
    differ = Differ()

    for i, (premise, hypothesis) in enumerate(pairs_found_debug):
        if i%2 == 0:
            premise_saved = premise
            hypothesis_saved = hypothesis

        elif i%2 == 1:
            premise_1_word_list = premise_saved.split()
            premise_2_word_list = premise.split()

            hypothesis_1_word_list = hypothesis_saved.split()
            hypothesis_2_word_list = hypothesis.split()

            rationale_1 = []
            rationale_2 = []
            
            diff = differ.compare(premise_1_word_list, premise_2_word_list)
            for i in diff:
                if i[:2] == "- ":
                    rationale_1.append(i[2:])
                elif i[:2] == "+ ":
                    rationale_2.append(i[2:])

            diff = differ.compare(hypothesis_1_word_list, hypothesis_2_word_list)
            for i in diff:
                if i[:2] == "- ":
                    rationale_1.append(i[2:])
                elif i[:2] == "+ ":
                    rationale_2.append(i[2:])

            rationale_1 = list(set(rationale_1))
            rationale_2 = list(set(rationale_2))

            sentence_to_rationale[premise_saved] = rationale_1
            sentence_to_rationale[premise] = rationale_2


    _keys = list(sentence_to_rationale.keys())
    _keys.sort()
    for i, k in enumerate(_keys):
        print(k)
        print(sentence_to_rationale[k])
        if i==50:
            break
    

    # return sentence_to_rationale
    cnt_example_with_rationale = 0
    for example in all_examples:
        try:
            if sentence_to_rationale[' '.join(example.sentence1)] != []:
                example.rationale = sentence_to_rationale[' '.join(example.sentence1)]
                cnt_example_with_rationale += 1
        except KeyError:
            pass

    print("Rationale cnt: {0}/{1}".format(cnt_example_with_rationale, len(all_examples)))


def find_rationale_esnli(examples):
    print("All examples:", len(examples))

    for example in examples:
        for i, word in enumerate(example.sentence1):
            if "*" in word:
                word_clean = word.replace("*", "").replace(".", "")
                example.rationale.append(word.replace("*", ""))
                example.rationale_word_indices.append(i+1) # +1 for CLS token
                example.sentence1[i] = word_clean
        
        sentence1_len = len(example.sentence1)
        for i, word in enumerate(example.sentence2):
            if "*" in word:
                word_clean = word.replace("*", "").replace(".", "")
                example.rationale.append(word.replace("*", ""))
                example.rationale_word_indices.append(i+2+sentence1_len) # +1 for CLS and SEP token
                example.sentence2[i] = word_clean


def convert_examples_to_features(args, examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    pbar = tqdm(examples, desc='convert examples to features ')
    for (ex_index, example) in enumerate(pbar):

        # Tokenize sentence 1
        tokens = [cls_token]
        for word in example.sentence1:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Tokenizer sentence 2
        sentence2_sub_token_len = 0
        for word in example.sentence2:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            sentence2_sub_token_len += len(word_tokens)

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids += [sequence_b_segment_id] * (sentence2_sub_token_len+1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # print(len(input_ids))
        # print(len(token_type_ids))
        # exit()

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=example.label,
                          ))

    return features

def gen_dataset(features, fold):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    all_aug_flag = torch.tensor([f.augmented_flag for f in features], dtype=torch.long)

    if fold ==  'train':
        max_seq_len = all_input_ids.size(1)
        for f in features:
            f.annotation_ids.extend([0 for _ in range(max_seq_len-len(f.annotation_ids))])   
        all_annotation_ids = torch.tensor([f.annotation_ids for f in features], dtype=torch.long)
    else:
        all_annotation_ids = torch.zeros_like(all_input_ids)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label, all_aug_flag, all_annotation_ids)

    return dataset

def gen_dataset_aug(features):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([[f.input_ids for f in _list] for _list in features], dtype=torch.long)
    all_attention_mask = torch.tensor([[f.attention_mask for f in _list] for _list in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([[f.token_type_ids for f in _list] for _list in features], dtype=torch.long)
    all_label = torch.tensor([[f.label for f in _list] for _list in features], dtype=torch.long)
    all_weight = torch.tensor([[f.weight for f in _list] for _list in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label, all_weight)

    return dataset


def find_token_frequency(examples, tokenizer):
    import itertools
    import numpy

    VOCAB_SIZE = tokenizer.vocab_size
    NUM_TOKENS = 0

    frequency = [0.0 for _ in range(VOCAB_SIZE)]
    all_tokens_in_train = [e.sentence1 for e in examples]
    all_tokens_in_train = list(itertools.chain(*all_tokens_in_train))

    for token in all_tokens_in_train:
        input_ids = tokenizer(token)['input_ids'][1:-1]
        for i in input_ids:
            NUM_TOKENS += 1
            frequency[i] += 1
        
    frequency = (numpy.array(frequency) / NUM_TOKENS).tolist()

    return frequency

def _sample(frequency, repeat):

        """
        return K samples based on input probability
        :param frequency: list of probabilities, sum(FREQUENCY) = 1
        :param k: repeat sampling w.r.t. FREQUENCY K times
        :return: the sampled index based on probability from FREQUENCY
        """

        token_indices_ret = []
        token_frequency_ret = []
        for repeat in range(repeat):
            x = random.uniform(0, 1)
            cumulative_probability = 0.0
            for token_index, item_probability in enumerate(frequency):
                cumulative_probability += item_probability
                if x < cumulative_probability:
                    token_indices_ret.append(token_index)
                    token_frequency_ret.append(item_probability)
                    break

        return torch.tensor(token_indices_ret), torch.tensor(token_frequency_ret) / sum(token_frequency_ret)

def modify_aug_flag_and_annotation_ids(features, aug_flags, annotation_ids):
    for i in range(len(aug_flags)):
        features[i].augmented_flag = aug_flags[i]
        features[i].annotation_ids = annotation_ids[i]
        

@torch.no_grad()
def augment_examples_and_conver_to_features(examples, max_seq_len, tokenizer, replace, args):
    pad_token_id = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size
    max_seq_cnt = -1

    # assert replace in ['mask', 'random', 'frequency', 'bert']
    if replace == 'none':
        return None

    aug_features = []
    aug_flags = [0 for _ in range(len(examples))]

    # annotation_ids = find_annotation_id_in_tokenizer_vocab(examples, tokenizer)
    annotation_ids = find_annotation_id_in_tokenizer_vocab_wnli(examples, tokenizer, args)

    assert len(annotation_ids) == len(examples), "Different length of annotation_ids and examples"

    bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir='./bert-base-uncased/') if replace == 'bert' else None
    bert_decider = bert_decider.cuda() if replace == 'bert' else None
    frequency = find_token_frequency(examples, tokenizer) if replace == 'frequency' else None

    pbar = tqdm(examples)
    print("Augmenting data by {}".format(args.replace))
    for example_id, example in enumerate(pbar):
        
        concat = ['[CLS]'] + example.sentence1 + ['[SEP]'] + example.sentence2 + ['[SEP]']

        attention_mask = []
        token_type_id = []

        cur_token_type_id = 0
        for word in concat:
            word_tokens = tokenizer.tokenize(word)
            attention_mask.extend([1 for _ in range(len(word_tokens))])
            
            if word == '[SEP]':
                cur_token_type_id = 1
            token_type_id.extend([cur_token_type_id for _ in range(len(word_tokens))])

        pad_len = max_seq_len - len(attention_mask)
        attention_mask.extend([0 for _ in range(pad_len)])
        token_type_id.extend([0 for _ in range(pad_len)])

        if len(attention_mask) > max_seq_cnt:
            max_seq_cnt = len(attention_mask)

        label = int(example.label)

        # Complete input_ids
        origin_inputs = tokenizer(' '.join(concat), return_tensors='pt')
        origin_inputs['input_ids'] = origin_inputs['input_ids'][0][1:-1].unsqueeze(dim=0)
        origin_inputs['token_type_ids'] = origin_inputs['token_type_ids'][0][1:-1].unsqueeze(dim=0)
        origin_inputs['attention_mask'] = origin_inputs['attention_mask'][0][1:-1].unsqueeze(dim=0)

        # print(origin_inputs['input_ids'].shape)
        # print(origin_inputs['token_type_ids'].shape)
        # print(origin_inputs['attention_mask'].shape)

        id_list = origin_inputs['input_ids'].tolist()[0]
        sentence_len = len(id_list)

        # pdb.set_trace()
        # print(id_list)
        # print(tokenizer.convert_ids_to_tokens(id_list))
        # print(annotation_ids[example_id])
        # print(len(annotation_ids[example_id]))
        assert sentence_len == len(annotation_ids[example_id]), "Different length of sentence_len and annotation_len"

        aug_each_sample = []
        
        if replace == 'mask':
            input_ids_mask_rationale = tokenizer(' '.join(concat))['input_ids'][1:-1].copy()
            input_ids_mask_non_rationale = input_ids_mask_rationale.copy()

            num_rationale = sum(annotation_ids[example_id])
            if num_rationale > 0:
                # Mask rationale
                for i in range(len(input_ids_mask_rationale)):
                    if annotation_ids[example_id][i] == 1:
                        input_ids_mask_rationale[i] = tokenizer.mask_token_id
                
                # Mask non-rationale
                mask_non_rationale_cnt = 0
                random_index = list(range(len(input_ids_mask_non_rationale)))
                shuffle(random_index)
                for i in random_index:
                    if annotation_ids[example_id][i] == 0:
                        input_ids_mask_non_rationale[i] = tokenizer.mask_token_id
                        mask_non_rationale_cnt += 1
                        if mask_non_rationale_cnt == num_rationale:
                            break
                
                input_ids_mask_rationale += [pad_token_id]*(max_seq_len-len(input_ids_mask_rationale))
                input_ids_mask_non_rationale += [pad_token_id]*(max_seq_len-len(input_ids_mask_non_rationale))

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_rationale,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_id,
                                    label=label,
                                    weight=1.,
                                    augmented_flag=1,
                            ))

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_mask_non_rationale,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_id,
                                    label=label,
                                    weight=1.,
                                    augmented_flag=1,
                            ))
                
                aug_flags[example_id] = 1

            else:
                input_ids_empty = [-1 for _ in range(max_seq_len)] # Just empty to make dataset aligned, will not be used
                attention_mask_empty = input_ids_empty
                token_type_id_empty = input_ids_empty
                intent_label_id_empty = -1

                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_empty,
                                    attention_mask=attention_mask_empty,
                                    token_type_ids=token_type_id_empty,
                                    label=intent_label_id_empty
                            ))
                aug_each_sample.append(
                    InputFeatures(input_ids=input_ids_empty,
                                    attention_mask=attention_mask_empty,
                                    token_type_ids=token_type_id_empty,
                                    label=intent_label_id_empty
                            ))

                

                
        else:
            aug_each_sample = replace_rationale_all(args, tokenizer, 
                                                    sentence_len, annotation_ids, example_id, origin_inputs,
                                                    attention_mask, token_type_id, label,
                                                    frequency, bert_decider, aug_flags
                                                    )

            aug_each_sample += replace_non_rationale_all(args, tokenizer, 
                                                         sentence_len, annotation_ids, example_id, origin_inputs,
                                                         attention_mask, token_type_id, label,
                                                         frequency, bert_decider, aug_flags
                                                        )

        aug_features.append(aug_each_sample)

    # for i in range(4):
    #     aug_1, aug_2, aug_3, aug_4, \
    #         aug_5, aug_6, aug_7, aug_8 = aug_features[i]
    #     temp_aug = tokenizer.convert_ids_to_tokens(aug_1.input_ids)
    #     if '[UNK]' not in temp_aug:
    #         print("Origin:")
    #         print(tokenizer.tokenize(' '.join(examples[i].sentence1 + ['[SEP]'] + examples[i].sentence2)))
    #         print()

    #         print("Rationales")
    #         print(examples[i].rationale)
    #         print()

    #         print("replace rationale")
    #         print(tokenizer.convert_ids_to_tokens(aug_1.input_ids))
    #         print()

    #         print("replace non-rationale")
    #         print(tokenizer.convert_ids_to_tokens(aug_6.input_ids))
    #         print()

            # print(aug_1.attention_mask)
            # print(aug_1.token_type_ids)
            # print(aug_1.intent_label_id)
            # exit()

    print("Max seq len cnt: ", max_seq_cnt)
    # exit()
    del bert_decider
    return aug_features, aug_flags, annotation_ids


def find_annotation_id_in_tokenizer_vocab_wnli(examples, tokenizer, args):
    def find_word_idx_in_premise(premise: list, hypothesis: list, rationales: list):
        rationale_idx = []
        for idx, word in enumerate(premise):
            if word in rationales:
                rationale_idx.append(idx+1) # +1 because [CLS] token in front of premise
        
        # premise_len = len(premise)
        # for idx, word in enumerate(hypothesis):
        #     if word in rationales:
        #         rationale_idx.append(idx+2+premise_len) # +2 because [CLS] and [SEP]
        
        return rationale_idx

    def find_word_idx_in_premise_and_hypothesis(premise: list, hypothesis: list, rationales: list):
        rationale_idx = []
        for idx, word in enumerate(premise):
            if word in rationales:
                rationale_idx.append(idx+1) # +1 because [CLS] token in front of premise
        
        premise_len = len(premise)
        for idx, word in enumerate(hypothesis):
            if word in rationales:
                rationale_idx.append(idx+2+premise_len) # +2 because [CLS] and [SEP]
        
        return rationale_idx


    annotation_id_ret = []
    for example in tqdm(examples, desc="find annotation id in vocab "):
        concat = ['[CLS]'] + example.sentence1 + ['[SEP]'] + example.sentence2 + ['[SEP]']
        if args.dataset == 'wnli':
            rationale_word_indices = find_word_idx_in_premise(example.sentence1, example.sentence2, example.rationale)
        elif args.dataset == 'esnli':
            rationale_word_indices = example.rationale_word_indices


        if rationale_word_indices != []:
            spans = [] # list of strings, e.g. ['i love', 'nlp', 'but...']
            spans_type = [] # list of int, 1=rationale span; 0=context span
            start_point = 0
            for rationale_index in rationale_word_indices:
                l_context = concat[start_point:rationale_index]
                rationale = concat[rationale_index]

                if len(l_context) != 0:
                    spans.append(' '.join(l_context))
                    spans_type.append(0)
                spans.append(rationale)
                spans_type.append(1)
                start_point = rationale_index+1

            last_r_context = concat[start_point:]
            if len(last_r_context) == 0:
                print("Error: last r context should be always not None!")
                exit(-1)
            else:
                spans.append(' '.join(last_r_context))
                spans_type.append(0)

            # print(concat)
            # print(rationale_word_indices)
            # print(spans)
            # print(spans_type)
            # print()
            # exit()

            rationale_tokenizer_id = []

            for span, span_type in zip(spans, spans_type):
                _len = len(tokenizer(span)['input_ids']) - 2
                rationale_tokenizer_id.extend(
                    [span_type for _ in range(_len)]
                )

        else:
            sentence_input_ids = tokenizer(' '.join(concat))['input_ids'][1:-1]
            rationale_tokenizer_id = [0 for _ in range(len(sentence_input_ids))]

        annotation_id_ret.append(rationale_tokenizer_id)
        
        # if sum(rationale_tokenizer_id) != 0:
        #     print(example.sentence1)
        #     print(example.sentence2)
        #     print(example.rationale)
        #     print(concat)
        #     print(tokenizer.tokenize(' '.join(concat)))
        #     print(rationale_tokenizer_id)
        #     exit()
    return annotation_id_ret


def cut_down_rationales(annotation_ids, percentage):
    
    def find_ones_indices(_list):
        ret = []
        for i, _item in enumerate(_list):
            if _item == 1:
                ret.append(i)
        return ret
    
    num_rationale_to_keep = int(len(annotation_ids) * percentage)
    if num_rationale_to_keep >= sum(annotation_ids):
        pass
    else:
        num_rationale_to_ignore = sum(annotation_ids) - num_rationale_to_keep
        ones_indices = find_ones_indices(annotation_ids)
        shuffle(ones_indices)
        rationales_to_ignore_indices = ones_indices[:num_rationale_to_ignore]
        for j_th_token in rationales_to_ignore_indices:
            annotation_ids[j_th_token] = -1 # -1 means that it IS rationale, but for the fluency of the sentence we decide not to replace it.
        


def replace_rationale_all(args, tokenizer, sentence_len, annotation_ids, example_id, origin_inputs, attention_mask, token_type_id, label, frequency, bert_decider, aug_flags):
    aug_each_sample = []

    cut_down_rationales(annotation_ids[example_id], args.max_rationale_percentage)

    num_rationale = sum(annotation_ids[example_id])
    if num_rationale == 0:
        input_ids_empty = [-1 for _ in range(args.max_seq_len)]
        attention_mask_empty = input_ids_empty
        token_type_id_empty = input_ids_empty
        label_empty = -1

        for r in range(args.replace_repeat):
            aug_each_sample.append(
                InputFeatures(input_ids=input_ids_empty,
                        attention_mask=attention_mask_empty,
                        token_type_ids=token_type_id_empty,
                        label=label_empty,
                ))
        return aug_each_sample
            

    replace = args.replace
    repeat = args.replace_repeat
    VOCAB_SIZE = tokenizer.vocab_size

    j_th_token_save = []
    replace_candidate_index = []
    replace_candidate_likelihood = []

    vocab_index_list = list(range(VOCAB_SIZE))

    for j_th_token in range(sentence_len):
        if (j_th_token == 0): # [CLS] token
            continue
        if origin_inputs['input_ids'][0][j_th_token] == tokenizer.sep_token_id:
            continue
        if origin_inputs['input_ids'][0][j_th_token] == tokenizer.pad_token_id:
            break
        if annotation_ids[example_id][j_th_token] == 0:
            continue

        j_th_token_save.append(j_th_token)

        if replace == 'random':
            replace_token_indices = random.sample(vocab_index_list, repeat)
            replace_token_likelihood = torch.full((len(replace_token_indices),), 1./repeat)
            
        elif replace == 'frequency':
            # frequency = find_token_frequency(examples, tokenizer)
            replace_token_indices, replace_token_likelihood = _sample(frequency, repeat)

        elif replace == 'bert':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            masked_inputs = copy.deepcopy(origin_inputs)
            masked_inputs = {key_: masked_inputs[key_].to(device) for key_ in masked_inputs}
            masked_inputs['input_ids'][0][j_th_token] = tokenizer.mask_token_id

            # bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncsaed', cache_dir='./bert-base-uncased/')
            output = bert_decider(**masked_inputs)
            logits = output[0]

            topk_return = torch.topk(logits[0][j_th_token], repeat)
            replace_token_indices = topk_return.indices
            replace_token_logits = topk_return.values  # Used to calculate likelihood of replacing origin token with this token
            replace_token_likelihood = torch.softmax(replace_token_logits, dim=0)

        replace_candidate_index.append(replace_token_indices)
        replace_candidate_likelihood.append(replace_token_likelihood)

    renormalized_liklihood = sum(replace_candidate_likelihood) / len(replace_candidate_likelihood)

    for r in range(repeat):
        inputs_copy = copy.deepcopy(origin_inputs)
        mean_likelihood = renormalized_liklihood[r]

        for ith_rationale, j_th_token in enumerate(j_th_token_save):
            inputs_copy['input_ids'][0][j_th_token] = replace_candidate_index[ith_rationale][r]
                
        if args.verbose:
            new_sentence = tokenizer.decode(inputs_copy['input_ids'][0][1:-1]) # debug
            print("Replace rationale")
            print(new_sentence)
            print()
        
        input_ids = inputs_copy['input_ids'][0].tolist()
        input_ids += [0 for _ in range(args.max_seq_len-len(input_ids))]

        aug_each_sample.append(
            InputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_id,
                        label=label,
                        weight=mean_likelihood,
                        augmented_flag=1
                ))

    aug_flags[example_id] = 1
    
    return aug_each_sample


def replace_non_rationale_all(args, tokenizer, sentence_len, annotation_ids, example_id, origin_inputs, attention_mask, token_type_id, label, frequency, bert_decider, aug_flags):
    aug_each_sample = []

    num_rationale = sum(annotation_ids[example_id])
    if num_rationale == 0:
        input_ids_empty = [-1 for _ in range(args.max_seq_len)]
        attention_mask_empty = input_ids_empty
        token_type_id_empty = input_ids_empty
        label_empty = -1
        slot_label_ids_empty = input_ids_empty

        for r in range(args.replace_repeat):
            aug_each_sample.append(
                InputFeatures(input_ids=input_ids_empty,
                        attention_mask=attention_mask_empty,
                        token_type_ids=token_type_id_empty,
                        label=label_empty,
                ))
        return aug_each_sample

    replace = args.replace
    repeat = args.replace_repeat
    VOCAB_SIZE = tokenizer.vocab_size

    j_th_token_save = []
    replace_candidate_index = []
    replace_candidate_likelihood = []
    num_non_rationale_to_replace = num_rationale

    random_token_ids = list(range(sentence_len))
    shuffle(random_token_ids)

    for j_th_token in random_token_ids:
        if (j_th_token == 0): # [CLS] token
            continue
        if origin_inputs['input_ids'][0][j_th_token] == tokenizer.sep_token_id:
            continue
        if origin_inputs['input_ids'][0][j_th_token] == tokenizer.pad_token_id:
            break
        if annotation_ids[example_id][j_th_token] == 1:
            continue

        j_th_token_save.append(j_th_token)
        num_non_rationale_to_replace -= 1

        if replace == 'random':
            vocab_index_list = list(range(VOCAB_SIZE))
            replace_token_indices = random.sample(vocab_index_list, repeat)
            replace_token_likelihood = torch.full((len(replace_token_indices),), 1./repeat)
            
        elif replace == 'frequency':
            # frequency = find_token_frequency(examples, tokenizer)
            replace_token_indices, replace_token_likelihood = _sample(frequency, repeat)

        elif replace == 'bert':
            masked_inputs = copy.deepcopy(origin_inputs)
            masked_inputs = {key_: masked_inputs[key_].cuda() for key_ in masked_inputs}
            masked_inputs['input_ids'][0][j_th_token] = tokenizer.mask_token_id
            # bert_decider = BertForMaskedLM.from_pretrained('bert-base-uncsaed', cache_dir='./bert-base-uncased/')
            output = bert_decider(**masked_inputs)
            logits = output[0]

            topk_return = torch.topk(logits[0][j_th_token], repeat)
            replace_token_indices = topk_return.indices
            replace_token_logits = topk_return.values  # Used to calculate likelihood of replacing origin token with this token
            replace_token_likelihood = torch.softmax(replace_token_logits, dim=0)

        replace_candidate_index.append(replace_token_indices)
        replace_candidate_likelihood.append(replace_token_likelihood)

        if num_non_rationale_to_replace == 0:
            break

    renormalized_liklihood = sum(replace_candidate_likelihood) / len(replace_candidate_likelihood)

    for r in range(repeat):
        inputs_copy = copy.deepcopy(origin_inputs)
        mean_likelihood = renormalized_liklihood[r]
        for ith_rationale, j_th_token in enumerate(j_th_token_save):
            inputs_copy['input_ids'][0][j_th_token] = replace_candidate_index[ith_rationale][r]
                
        if args.verbose:
            new_sentence = tokenizer.decode(inputs_copy['input_ids'][0][1:-1]) # debug
            print("Replace non-rationale:")
            print(new_sentence)
            print()
        
        input_ids = inputs_copy['input_ids'][0].tolist()
        input_ids += [0 for _ in range(args.max_seq_len-len(input_ids))]

        aug_each_sample.append(
            InputFeatures(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_id,
                        label=label,
                        weight=mean_likelihood,
                        augmented_flag=1
                ))
    aug_flags[example_id] = 1

    return aug_each_sample
