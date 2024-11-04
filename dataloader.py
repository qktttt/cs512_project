# Dataloader.py
import torch
from torch.utils.data import Dataset

# This data loader need to be elaborated 
# in order to load raw text data and convert it into numerical representation using vocabulary.
# TODO --------^-----------
class TextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = texts
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return torch.tensor([self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()], dtype=torch.long)
