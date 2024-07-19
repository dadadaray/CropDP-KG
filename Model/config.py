# 设置训练参数
# 后续分成NER和RE两部分，单独参数！
import torch


class TrainingConfig(object):
    batch_size = 16
    epoches = 8
    print_step = 100
    model_name = "Bert_Prompt"
    device = "cuda" if torch.cuda.is_available() else "cpu"


class NERTrainingConfig(object):
    batch_size = 16
    epoches = 5
    model_name = "Bert_Prompt"
    device = "cuda"
    lr = 4e-5


class RETrainingConfig(object):
    batch_size = 4
    epoches = 5
    model_name = "Bert_Prompt"
    device = "cuda"
    lr = 2e-4


class DataConfig(object):
    max_length = 384


class BertConfig(object):
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    path = "/root/autodl-tmp/bert_CN"
    lr = 3e-5


class LSTMConfig(object):
    emb_size = 256  # 词向量的维数
    hidden_size = 256  # lstm隐向量的维数
    lr = 0.0001


class PromptConfig(object):
    pre_seq_len = 128
    prefix_projection = True
    num_hidden_layers = BertConfig.num_hidden_layers
    num_attention_heads = BertConfig.num_attention_heads
    hidden_size = 768
    prefix_hidden_size = 256
    shared_layers = 6
    independent_layers = 6

