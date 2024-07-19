import pickle
import time
from copy import deepcopy

import torch
from tqdm import trange

from NER_model import Bert_BiLSTM_CRF_NER, Bert_CRF_NER

from config import TrainingConfig, BertConfig, PromptConfig, LSTMConfig
from metrics import Metrics


class model_cal_NER(object):
    def __init__(self, out_size):
        self.device = TrainingConfig.device
        self.bert_emb_size = BertConfig.hidden_size
        # ------------选择模型---------------
       # self.model = Bert_NER(out_size).to(self.device)
        # self.model = Bert_CRF_NER(out_size).to(self.device)
        self.model = Bert_BiLSTM_CRF_NER(out_size).to(self.device)
        # self.model = Bert_Freeze_Prompt_NER(out_size).to(self.device)
        # self.model = Bert_Freeze_Prompt_CRF_NER(out_size).to(self.device)
        # self.model = Bert_Freeze_Prompt_BiLSTM_NER(out_size).to(self.device)

        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = BertConfig.lr
        self.Bert_lr = BertConfig.lr
        self.LSTM_lr = LSTMConfig.lr
        self.batch_size = TrainingConfig.batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.optimizer = torch.optim.Adam([{"params": self.model.bert.PLM.parameters(), "lr": self.Bert_lr},
                                           {"params": self.model.lstm.parameters(), "lr": self.LSTM_lr},
                                           {"params": self.model.crf.parameters(), "lr": 0.001}], weight_decay=0.00001)
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, train_dataloader, dev_dataloader):
        print("训练数据总量:{}".format(len(train_dataloader.dataset)))
        epoch_iterator = trange(1, self.epoches + 1, desc="Epoch")
        for epoch in epoch_iterator:
            self.step = 0
            losses = 0.
            for idx in trange(0, len(train_dataloader.dataset), self.batch_size, desc="Iteration"):
                # batch_texts: list(batch_size) 每个元素包含tensor("input_ids", "token_type_ids", "attention_mask")
                batch_texts = train_dataloader.dataset.token_texts[idx:idx + self.batch_size]
                batch_tags = train_dataloader.dataset.tags[idx:idx + self.batch_size]
                losses += self.train_step(batch_texts, batch_tags)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(train_dataloader.dataset) // self.batch_size + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            val_loss = self.validate(dev_dataloader)
            print("Epoch {}, Val Loss:{:.4f}".format(epoch, val_loss))


    def train_step(self, batch_texts, batch_tags):
        self.model.train()
        self.step += 1

        loss, prediction = self.model(batch_texts, batch_tags)

        # 计算损失，反向传递
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_dataloader):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for idx in range(0, len(dev_dataloader.dataset), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_texts = dev_dataloader.dataset.token_texts[idx:idx + self.batch_size]
                batch_tags = dev_dataloader.dataset.tags[idx:idx + self.batch_size]

                # forward
                loss, prediction = self.model(batch_texts, batch_tags)

                # 计算损失
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存NER模型...")
                self.best_model = deepcopy(self.model)
                torch.save(self.best_model, './model/BERT_model.pt')
                self._best_val_loss = val_loss
            return val_loss

    def test(self, test_dataloader):
        self.best_model.eval()
        with torch.no_grad():
            pred_tag_lists = []
            tag_lists = []
            for idx in trange(0, len(test_dataloader.dataset), self.batch_size, desc="Iteration"):
                batch_texts = test_dataloader.dataset.token_texts[idx:idx + self.batch_size]
                batch_tags = test_dataloader.dataset.tags[idx:idx + self.batch_size]
                pred_tags = self.best_model(token_texts=batch_texts)
                pred_tag_lists.extend(pred_tags)
                tag_lists.extend(batch_tags)
        return pred_tag_lists, tag_lists
    
        


def train_model_NER(train_dataloader, dev_dataloader, test_dataloader, tag2id, remove_0=False):
    start = time.time()
    out_size = len(tag2id)

    model = model_cal_NER(out_size)
    model_name = TrainingConfig.model_name

    print("start to train the {} ...".format(model_name))
    model.train(train_dataloader, dev_dataloader)
    with open("./model/BERT_model.pt", 'wb') as f:
        pickle.dump(model, f)
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = model.test(test_dataloader)

    id2tag = {v: k for k, v in tag2id.items()}
    metrics = Metrics(test_tag_lists, pred_tag_lists, id2tag, remove_0=remove_0)
    metrics.report_scores(model_name=model_name)

    return pred_tag_lists

