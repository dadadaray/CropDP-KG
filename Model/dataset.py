from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, token_texts, tags):
        super(NERDataset, self).__init__()
        self.token_texts = token_texts
        self.tags = tags

    def __getitem__(self, index):
        token_text = self.token_texts[index]
        tag = self.tags[index]
        return token_text, tag

    def __len__(self):
        return len(self.token_texts)


class REDataset(Dataset):
    def __init__(self, token_texts, tags):
        super(REDataset, self).__init__()
        self.token_texts = token_texts
        self.tags = tags

    def __getitem__(self, index):
        token_text = self.token_texts[index]
        tag = self.tags[index]
        return token_text, tag

    def __len__(self):
        return len(self.token_texts)
