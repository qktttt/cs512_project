import torch
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter

def build_vocab(data_dir, vocab_size=10000):
    word_counter = Counter()
    
    # Dynamically find files
    files = ["sentiment.train.0", "sentiment.train.1"]
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                word_counter.update(words)
    
    # Build vocabulary with most common words
    most_common_words = word_counter.most_common(vocab_size)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common_words, start=2):
        vocab[word] = idx
    
    return vocab

class TextDataset(Dataset):
    def __init__(self, data_dir, vocab):
        super(TextDataset, self).__init__()
        self.data = []
        self.vocab = vocab

        # Load data from the files
        files = ["sentiment.train.0", "sentiment.train.1"]
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.strip().split()
                    label = 1 if filename.endswith('.1') else 0  # Binary label
                    self.data.append((tokens, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Collate function to pad sequences in a batch
def collate_fn(batch):
    token_ids, labels = zip(*batch)
    lengths = [len(ids) for ids in token_ids]
    padded_token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_token_ids, labels, torch.tensor(lengths, dtype=torch.long)

# Example usage
if __name__ == '__main__':
    data_dir = "./data/sentiment_style_transfer/yelp"
    vocab = build_vocab(data_dir)
    dataset = TextDataset(data_dir, vocab)

    # DataLoader with collate function for padding
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for i, (token_ids, labels, lengths) in enumerate(data_loader):
        print(f"Batch {i + 1}")
        print(f"Token IDs:\n{token_ids}")
        print(f"Labels: {labels}")
        print(f"Lengths: {lengths}\n")
