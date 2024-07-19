import torch
from numpy.f2py import __version__
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config import TrainingConfig,BertConfig, DataConfig
from dataset import NERDataset, REDataset
from data_process import build_NER
from train import train_model_NER


def get_tag2id(tag_file):
    result = {}
    with open(tag_file, "r", encoding='utf-8') as f:
        tags = f.read().split()
        for i, tags in enumerate(tags):
            result[tags] = i
    return result


def get_NER_dataloader():
    tag2id = get_tag2id("medicine_data_NER/NER_labels.txt")
    id2tag = {v: k for k, v in tag2id.items()}
    entity_tag_size = len(tag2id)
    print("entity_tag_size", entity_tag_size)

    train_word_lists, train_tag_lists = build_NER("train",
                                                  tag2id)  # list (tensor: input_ids, token_type_ids, attention_mask)
    dev_word_lists, dev_tag_lists = build_NER("dev", tag2id)
    test_word_lists, test_tag_lists = build_NER("test", tag2id)

    train_dataset = NERDataset(train_word_lists, train_tag_lists)  # 保存list类型的token_texts和list
    dev_dataset = NERDataset(dev_word_lists, dev_tag_lists)
    test_dataset = NERDataset(test_word_lists, test_tag_lists)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=TrainingConfig.batch_size, shuffle=True,
                                  num_workers=4)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=TrainingConfig.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=TrainingConfig.batch_size, shuffle=True,
                                 num_workers=4)

    return train_dataloader, dev_dataloader, test_dataloader, tag2id

def NER_main():
    train_dataloader_NER, dev_dataloader_NER, test_dataloader_NER, tag2id = get_NER_dataloader()
    print("NER model training...")
    train_model_NER(train_dataloader_NER, dev_dataloader_NER, test_dataloader_NER, tag2id)


def test():
    load_model = torch.load("model/NER_model.pt")
    test_sentences = "菜豆锈病	Bean rust	锈病是菜豆的一种主要病害。各地均有分布，发生普遍。发病猛，传播快。发病率可达到100%，为害严重。	主要侵害叶片，严重时茎蔓、叶柄及豆荚均可发病。病初叶上生黄绿色或灰白色小斑点，随后凸起，变成黄褐色小疱。扩大病斑后，表皮破裂，散出红色粉末(夏孢子)。发病后期夏孢子堆转变为黑色的冬孢子堆。发病叶片会变形早落。叶片正面和豆荚上产生黄色小斑点，小点四周会产生橙红色斑点。	(1)发病时期。华南地区在任何种植季节都能发生，其中以4～7月发生最重；华北地区7～8月份为发病盛期。北方该病主要发生在夏秋两季，尤其是叶面结露及叶面上的水滴是锈菌孢子萌发和侵入的先决条件。(2)气候因素。菜豆进入开花结荚期，气温20℃左右，高湿，昼夜温差大及结露持续时间长，易流行。低洼地，排水不良，种植过密，通风不畅的地块，发病重。"
    with torch.no_grad():
        result = load_model.predict(test_sentences)
    print(result)



if __name__ == "__main__":
    NER_main()
    # RE_main()
    # NER_RE_main()
    # test()
   