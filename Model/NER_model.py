import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

from config import BertConfig, TrainingConfig, LSTMConfig, PromptConfig, DataConfig
from encode import PrefixEncoder

'''
# Freeze-Bert
class BERTSentenceEncoderFreeze(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        print("Freeze_Bert")
        self.PLM = BertModel.from_pretrained(BertConfig.path)
        for name, param in self.PLM.named_parameters():
            param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained(BertConfig.path)
        self.output_size = BertConfig.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids=None, past_key_values=None):
        outputs = self.PLM(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, past_key_values=past_key_values)
        return outputs
'''

# unfreeze-Bert
class BERTSentenceEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        print("Unfreeze_Bert")
        self.PLM = BertModel.from_pretrained(BertConfig.path).to(TrainingConfig.device)
        self.tokenizer = BertTokenizer.from_pretrained(BertConfig.path)
        self.output_size = BertConfig.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids, past_key_values=None):
        outputs = self.PLM(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, past_key_values=past_key_values)
        return outputs


class BiLSTM(nn.Module):
    def __init__(self, input_size, out_size, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, LSTMConfig.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * LSTMConfig.hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        emb, _ = self.bilstm(emb)
        scores = self.fc(emb)
        return scores

    def test(self, emb):
        logits = self.forward(emb)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids

'''
class Bert_Freeze_Prompt_NER(nn.Module):
    def __init__(self, out_size):
        print("Bert with P-tuning v2")
        super(Bert_Freeze_Prompt_NER, self).__init__()
        self.out_size = out_size
        self.bert = BERTSentenceEncoderFreeze()
        self.dense = torch.nn.Linear(BertConfig.hidden_size, out_size)
        self.dropout = nn.Dropout(0)
        self.device = TrainingConfig.device
        self.pre_seq_len = PromptConfig.pre_seq_len
        self.prefix_encoder = PrefixEncoder()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_hidden_layers = PromptConfig.num_hidden_layers
        self.num_attention_heads = PromptConfig.num_attention_heads
        self.num_embedding = PromptConfig.hidden_size // self.num_attention_heads

        Bert_param = 0
        for name, param in self.bert.PLM.named_parameters():
            Bert_param += param.numel()
        print('Bert param is {}'.format(Bert_param))

        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        print('all param is {}'.format(all_param))

        total_param = all_param - Bert_param
        print('train param is {}'.format(total_param))  # 4963106

    def get_prompt(self, batch_size=TrainingConfig.batch_size):
        prefix_tokens = torch.randint(0, self.pre_seq_len, (batch_size, self.pre_seq_len)).long().to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_hidden_layers * 2,
            self.num_attention_heads,
            self.num_embedding
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, token_texts, tags=None):
        print("device", self.device)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            input_ids.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            attention_mask.append(text["attention_mask"])
        input_ids = torch.stack(input_ids, dim=0).squeeze(1).to(self.device)        # tensor:[batch_size * seq_length]
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        attention_mask = torch.stack(attention_mask, dim=0).squeeze(1).to(self.device)

        batch_size = len(token_texts)
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.device)    # 全1矩阵->张量 batch_size x prompt_length
        total_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)              # 和mask拼接   [batch_size * (max_length+pre_seq_length)]

        bert_out = self.bert(input_ids=input_ids, attention_mask=total_mask, token_type_ids=token_type_ids, past_key_values=past_key_values)[0]
        bert_out = bert_out.permute(1, 0, 2)

        feats = self.dense(bert_out)    # feats:[384, 16, 34]

        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag)) #报错
            #print(f"Before stacking, tensor_tags length: {len(tensor_tags)}")
            # for i, t in enumerate(tensor_tags):
            #     print(f"tensor_tags[{i}] shape: {t.shape}")

            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)  #报错  # tensor_tags:[16, 384]
            # 检查 stack 之后的形状
            # print(f"After stacking, tensor_tags shape: {tensor_tags.shape}")

            outputs = feats.permute(1, 2, 0)                  # outputs:[16, 34, 384]   masks:[16, 384]
            logits = (outputs * attention_mask.unsqueeze(1)).permute(0, 2, 1).reshape(-1, self.out_size)
            tags = tensor_tags.view(-1)
            loss = self.loss_fn(logits, tags)
            predictions = torch.argmax(outputs, dim=1)
            return loss, predictions
        else:
            predictions = torch.argmax(feats.permute(1, 2, 0), dim=1)
            return predictions.tolist()
'''

'''
class Bert_Freeze_Prompt_CRF_NER(nn.Module):
    def __init__(self, out_size):
        print("Bert with P-tuning v2")
        super(Bert_Freeze_Prompt_CRF_NER, self).__init__()
        self.bert = BERTSentenceEncoderFreeze()
        self.dense = torch.nn.Linear(BertConfig.hidden_size, out_size)
        self.dropout = nn.Dropout(0)
        self.device = TrainingConfig.device
        self.pre_seq_len = PromptConfig.pre_seq_len
        self.prefix_tokens = torch.arange(PromptConfig.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder()
        self.num_hidden_layers = PromptConfig.num_hidden_layers
        self.num_attention_heads = PromptConfig.num_attention_heads
        self.num_embedding = PromptConfig.hidden_size // self.num_attention_heads
        self.crf = CRF(num_tags=out_size)

        Bert_param = 0
        for name, param in self.bert.PLM.named_parameters():
            Bert_param += param.numel()
        print('Bert param is {}'.format(Bert_param))

        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        print('all param is {}'.format(all_param))

        total_param = all_param - Bert_param
        print('train param is {}'.format(total_param))  # 4963106

    def neg_log_likelihood(self, emissions, tags=None, mask=None, reduction=None):
        return -1 * self.crf(emissions=emissions, tags=tags, mask=mask, reduction=reduction)

    def get_prompt(self, batch_size=TrainingConfig.batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_hidden_layers * 2,
            self.num_attention_heads,
            self.num_embedding
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, token_texts, tags=None):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            input_ids.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            attention_mask.append(text["attention_mask"])
        input_ids = torch.stack(input_ids, dim=0).squeeze(1).to(self.device)        # tensor:[batch_size * seq_length]
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        attention_mask = torch.stack(attention_mask, dim=0).squeeze(1).to(self.device)

        batch_size = len(token_texts)
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.device)    # 全1矩阵->张量 batch_size x prompt_length
        total_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)          # 和mask拼接

        bert_out = self.bert(input_ids=input_ids, attention_mask=total_mask, token_type_ids=token_type_ids, past_key_values=past_key_values)[0]
        bert_out = bert_out.permute(1, 0, 2)

        feats = self.dense(bert_out)    # feats:[384, 16, 34]

        # 格式转换
        masks = attention_mask.permute(1, 0)
        masks = masks.clone().detach().bool()
        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag))
            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)
            tensor_tags = tensor_tags.permute(1, 0)
            loss = self.neg_log_likelihood(feats, tensor_tags, masks, 'mean')  # feats:[384,16,34]  tensor_tags:[384,16]  masks:[384,16]
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return loss, predictions
        else:
            predictions = torch.argmax(feats.permute(1, 2, 0), dim=1)
            return predictions.tolist()
'''

class Bert_BiLSTM_CRF_NER(nn.Module):
    def __init__(self, out_size):
        super(Bert_BiLSTM_CRF_NER, self).__init__()
        self.device = TrainingConfig.device
        # bert层
        self.bert = BERTSentenceEncoder()
        # BiLSTM层
        self.lstm = BiLSTM(BertConfig.hidden_size, out_size=out_size)
        # CRF层
        self.crf = CRF(num_tags=out_size)

    def neg_log_likelihood(self, emissions, tags=None, mask=None, reduction=None):
        return -1 * self.crf(emissions=emissions, tags=tags, mask=mask, reduction=reduction)

    def forward(self, token_texts=None, tags=None):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """

        texts = []
        token_type_ids = []
        masks = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            texts.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            masks.append(text["attention_mask"])
        texts = torch.stack(texts, dim=0).squeeze(1).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        masks = torch.stack(masks, dim=0).squeeze(1).to(self.device)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.lstm(bert_out)

        # 格式转换
        masks = masks.permute(1, 0)
        masks = masks.clone().detach().bool()
        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag))
            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)
            tensor_tags = tensor_tags.permute(1, 0)
            loss = self.neg_log_likelihood(feats, tensor_tags, masks, 'mean')
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return predictions

    def predict(self, texts):
        tokenizer = BertTokenizer.from_pretrained(BertConfig.path)
        tokenized_text = tokenizer.encode_plus(text=texts,
                                               max_length=DataConfig.max_length,
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt',
                                               padding='do_not_pad',
                                               truncation=True)
        self.eval()
        tokenized_texts = [tokenized_text]
        with torch.no_grad():
            outputs = self.forward(tokenized_texts)
            # for i in range(len(tokenized_text["input_ids"][0].tolist())):
            #     # print(tokenizer.convert_ids_to_tokens(tokenized_text["input_ids"][0].tolist()[i]), outputs[0][i])
            predictions = outputs[0]
        return predictions


class Bert_Freeze_Prompt_BiLSTM_NER(nn.Module):
    def __init__(self, out_size):
        super(Bert_Freeze_Prompt_BiLSTM_NER, self).__init__()
        self.device = TrainingConfig.device
        # bert层
        self.bert = BERTSentenceEncoder()
        # BiLSTM层
        self.lstm = BiLSTM(BertConfig.hidden_size, out_size=out_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        # CRF层
        self.crf = CRF(num_tags=out_size)

    def forward(self, token_texts=None, tags=None):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """

        texts = []
        token_type_ids = []
        masks = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            texts.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            masks.append(text["attention_mask"])
        texts = torch.stack(texts, dim=0).squeeze(1).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        masks = torch.stack(masks, dim=0).squeeze(1).to(self.device)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.lstm(bert_out)

        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag))
            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)
            outputs = feats.permute(1, 2, 0)  # outputs:[16, 34, 384]   masks:[16, 384]
            loss = self.loss_fn(outputs * masks.unsqueeze(1), tensor_tags)
            loss = loss.mean(dim=1).mean(dim=0)
            predictions = torch.argmax(outputs, dim=1)
            return loss, predictions
        else:
            predictions = torch.argmax(feats.permute(1, 2, 0), dim=1)
            return predictions.tolist()


class Bert_CRF_NER(nn.Module):
    def __init__(self, out_size):
        super(Bert_CRF_NER, self).__init__()
        self.device = TrainingConfig.device
        # bert层
        self.bert = BERTSentenceEncoder()
        # dense层
        self.dense = nn.Linear(BertConfig.hidden_size, out_size)
        # CRF层
        self.crf = CRF(num_tags=out_size)

    def neg_log_likelihood(self, emissions, tags=None, mask=None, reduction=None):
        return -1 * self.crf(emissions=emissions, tags=tags, mask=mask, reduction=reduction)

    def forward(self, token_texts=None, tags=None):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """

        texts = []
        token_type_ids = []
        masks = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            texts.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            masks.append(text["attention_mask"])
        texts = torch.stack(texts, dim=0).squeeze(1).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        masks = torch.stack(masks, dim=0).squeeze(1).to(self.device)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.dense(bert_out)

        # 格式转换
        masks = masks.permute(1, 0)
        masks = masks.clone().detach().bool()
        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag))
            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)
            tensor_tags = tensor_tags.permute(1, 0)
            loss = self.neg_log_likelihood(feats, tensor_tags, masks, 'mean')   # feats:[384,16,34]  tensor_tags:[384,16]  masks:[384,16]
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return loss, predictions
        else:
            predictions = self.crf.decode(emissions=feats, mask=masks)
            return predictions
'''

class Bert_NER(nn.Module):
    def __init__(self, out_size):
        super(Bert_NER, self).__init__()
        self.device = TrainingConfig.device
        # bert层
        self.bert = BERTSentenceEncoder()
        # dense层
        self.dense = nn.Linear(BertConfig.hidden_size, out_size)
        # CRF层
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, token_texts=None, tags=None):
        """
        token_texts:{"input_size": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                    "token_type_ids": tensor,  [batch, 1, seq_len]->[batch, seq_len]
                     "attention_mask": tensor  [batch, 1, seq_len]->[batch, seq_len]->[seq_len, batch]
                     }
        tags:  [batch, seq_len]->[seq_len, batch]
        bert_out:  [batch, seq_len, hidden_size(768)]->[seq_len, batch, hidden_size]
        feats:  [seq_len, batch, tagset_size]
        loss:  tensor
        predictions:  [batch, num]
        """

        texts = []
        token_type_ids = []
        masks = []
        tensor_tags = []
        for i, text in enumerate(token_texts):
            texts.append(text["input_ids"])
            token_type_ids.append(text["token_type_ids"])
            masks.append(text["attention_mask"])
        texts = torch.stack(texts, dim=0).squeeze(1).to(self.device)
        token_type_ids = torch.stack(token_type_ids, dim=0).squeeze(1).to(self.device)
        masks = torch.stack(masks, dim=0).squeeze(1).to(self.device)
        bert_out = self.bert(input_ids=texts, attention_mask=masks, token_type_ids=token_type_ids)[0]
        bert_out = bert_out.permute(1, 0, 2)
        feats = self.dense(bert_out)

        # 格式转换
        # masks = masks.permute(1, 0)
        # masks = masks.clone().detach().bool()
        # 计算损失和、预测值
        if tags is not None:
            for i, tag in enumerate(tags):
                tensor_tags.append(torch.tensor(tag))
            tensor_tags = torch.stack(tensor_tags, dim=0).to(self.device)
            outputs = feats.permute(1, 2, 0)                  # outputs:[16, 34, 384]   masks:[16, 384]
            loss = self.loss_fn(outputs * masks.unsqueeze(1), tensor_tags)
            loss = loss.mean(dim=1).mean(dim=0)
            predictions = torch.argmax(outputs, dim=1)
            return loss, predictions
        else:
            outputs = feats.permute(1, 2, 0)
            predictions = torch.argmax(outputs, dim=1)
            return predictions
'''

