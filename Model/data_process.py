import os

import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from config import BertConfig, DataConfig


def build_NER(split, tag2id):
    assert split.lower() in ["train", "dev", "test"]
    word_lists = []
    tag_lists = []
    dir_name = "./medicine_data_NER/" + split + ".txt"
    with open(dir_name, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line.strip() != '':
                if line.find("。") == -1:
                    word, tag = line.strip('\n').split(' ')
                    word_list.append(word)
                    tag_list.append(tag2id[tag])
                else:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []

    tokenizer = BertTokenizer.from_pretrained(BertConfig.path)
    token_texts = []
    max_length = 0
    for text in word_lists:
        if len(text) > max_length:
            max_length = len(text)
        tokenized = tokenizer.encode_plus(text=text,
                                          max_length=DataConfig.max_length,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          padding='max_length',
                                          truncation=True)
        token_texts.append(tokenized)
    if split.lower() != "test" and split.lower() != "test_mini":
        for t in tag_lists:
            for i in range(DataConfig.max_length):
                if i >= len(t):
                    t.append(-1)
    return token_texts, tag_lists

