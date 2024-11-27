# %%
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
from dataloader import *
from model import * 
from nltk.translate.bleu_score import sentence_bleu
import nltk 
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
num_epochs = 10
learning_rate = 0.001
target_confidence = 0.8 

# %%
def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_length = max(len(seq) for seq in inputs)
    
    # Convert each sequence to a list, pad with 0, and convert to tensor
    padded_inputs = [torch.cat([seq, torch.zeros(max_length - len(seq), dtype=torch.long)]) for seq in inputs]
    lengths = [len(seq) for seq in inputs]
    
    return torch.stack(padded_inputs), torch.tensor(labels, dtype=torch.float), lengths

def tokens_to_words(token_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    return [inv_vocab.get(token_id, '<UNK>') for token_id in token_ids if token_id != 0]  # Exclude padding


class TextDatasetTest(Dataset):
    def __init__(self, data_dir, vocab):
        super(TextDatasetTest, self).__init__()
        self.data = []
        self.vocab = vocab

        # Load data from the files
        files = ["sentiment.test.0", "sentiment.test.1"]
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
    
data_dir = "./data/sentiment_style_transfer/yelp"
vocab = build_vocab(data_dir)
dataset = TextDatasetTest(data_dir, vocab)
data_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)

# %%
class DisentangledVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, style_dim, content_dim):
        super(DisentangledVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder
        self.encoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.style_fc = nn.Linear(hidden_dim * 2, style_dim)  # Style embedding
        self.content_fc = nn.Linear(hidden_dim * 2, content_dim)  # Content embedding

        # Latent space (reparameterization layers)
        self.style_mu = nn.Linear(style_dim, style_dim)
        self.style_logvar = nn.Linear(style_dim, style_dim)
        self.content_mu = nn.Linear(content_dim, content_dim)
        self.content_logvar = nn.Linear(content_dim, content_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(style_dim + content_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, vocab_size)

        # Style and content classifiers for multi-task and adversarial loss
        self.style_classifier = nn.Linear(style_dim, 2)  # Assume binary classification for styles
        self.content_classifier = nn.Linear(content_dim, vocab_size)  # Content classification (BoW or POS tags)

    def encode(self, x):
        """Encode input text into style and content latent embeddings."""        # Move data to the appropriate device
        token_ids = token_ids.to(device)
        style_labels = style_labels.to(device, dtype=torch.long)  # Ensure correct type for labels
        
        # Generate dummy content labels (use zeros since content labels are not provided)
        content_labels = torch.zeros(token_ids.size(0), dtype=torch.long).to(device)
        
        # Forward pass through the model
        recon_x, style_mu, style_logvar, content_mu, content_logvar, style, content = vae(token_ids)
        
        # Calculate VAE loss
        loss_vae = vae_loss(recon_x, token_ids, style_mu, style_logvar, content_mu, content_logvar)
        
        # Multi-task loss for style classification (content classification loss is skipped here)
        style_preds = vae.classify_style(style)
        loss_multi_task = F.cross_entropy(style_preds, style_labels)
        
        # Total loss
        loss = loss_vae + loss_multi_task
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        embedded = self.embedding(x)
        _, hidden = self.encoder_rnn(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Combine bidirectional GRU outputs

        style_h = self.style_fc(hidden)  # Style embedding
        content_h = self.content_fc(hidden)  # Content embedding

        style_mu, style_logvar = self.style_mu(style_h), self.style_logvar(style_h)
        content_mu, content_logvar = self.content_mu(content_h), self.content_logvar(content_h)
        return style_mu, style_logvar, content_mu, content_logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, style, content, x):
        """Decode from latent space to generate output text."""
        latent = torch.cat([style, content], dim=1)
        hidden = self.latent_to_hidden(latent).unsqueeze(0)  # Initialize decoder hidden state
        embedded = self.embedding(x)
        outputs, _ = self.decoder_rnn(embedded, hidden)
        logits = self.output_fc(outputs)
        return logits

    def forward(self, x):
        """Forward pass: encode, reparameterize, and decode."""
        style_mu, style_logvar, content_mu, content_logvar = self.encode(x)
        style = self.reparameterize(style_mu, style_logvar)
        content = self.reparameterize(content_mu, content_logvar)
        recon_x = self.decode(style, content, x)
        return recon_x, style_mu, style_logvar, content_mu, content_logvar, style, content

    def classify_style(self, style):
        """Style classification for multi-task/adversarial loss."""
        return self.style_classifier(style)

    def classify_content(self, content):
        """Content classification for multi-task/adversarial loss."""
        return self.content_classifier(content)
    

def vae_loss(recon_x, x, style_mu, style_logvar, content_mu, content_logvar):
    recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=0)  # Reconstruction loss
    kl_style = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())  # KL divergence for style
    kl_content = -0.5 * torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp())  # KL divergence for content
    return recon_loss + kl_style + kl_content

def multi_task_loss(style_preds, style_labels, content_preds, content_labels):
    style_loss = F.cross_entropy(style_preds, style_labels)  # Style classification loss
    content_loss = F.cross_entropy(content_preds, content_labels)  # Content classification loss
    return style_loss + content_loss

def adversarial_loss(style_preds, content_preds):
    adversarial_style_loss = -F.cross_entropy(style_preds, torch.zeros_like(style_preds))  # Fool style classifier
    adversarial_content_loss = -F.cross_entropy(content_preds, torch.zeros_like(content_preds))  # Fool content classifier
    return adversarial_style_loss + adversarial_content_loss


# %%
# Modified training loop
def train_vae_with_dataset(vae, optimizer, data_loader, device):
    """
    Train the VAE with a custom dataset.
    
    Args:
        vae: Disentangled VAE model.
        optimizer: Optimizer for the VAE model.
        data_loader: DataLoader providing (token_ids, labels) for the dataset.
        device: Device (CPU or GPU) to run the training on.
    
    Returns:
        Average loss for the epoch.
    """
    vae.train()
    total_loss = 0
    print(len(data_loader))

    for token_ids, style_labels, _ in data_loader:  # Ignore `lengths`
        # Move data to the appropriate device
        token_ids = token_ids.to(device)
        style_labels = style_labels.to(device, dtype=torch.long)  # Ensure correct type for labels
        
        # Generate dummy content labels (use zeros since content labels are not provided)
        content_labels = torch.zeros(token_ids.size(0), dtype=torch.long).to(device)
        
        # Forward pass through the model
        recon_x, style_mu, style_logvar, content_mu, content_logvar, style, content = vae(token_ids)
        
        # Calculate VAE loss
        loss_vae = vae_loss(recon_x, token_ids, style_mu, style_logvar, content_mu, content_logvar)
        
        # Multi-task loss for style classification (content classification loss is skipped here)
        style_preds = vae.classify_style(style)
        loss_multi_task = F.cross_entropy(style_preds, style_labels)
        
        # Total loss
        loss = loss_vae + loss_multi_task
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 256
style_dim = 16
content_dim = 128
learning_rate = 1e-3
epochs = 10

# Initialize model, optimizer, and device
vae = DisentangledVAE(vocab_size, embedding_dim, hidden_dim, style_dim, content_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    loss = train_vae_with_dataset(vae, optimizer, data_loader, device)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


