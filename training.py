# Training.py
import torch
from torch.utils.data import DataLoader
from Model import VAE, vae_loss
from Dataloader import TextDataset

# Parameters
vocab_size = 10000
embed_dim = 300
hidden_dim = 256
style_dim = 16
content_dim = 128
epochs = 10
batch_size = 16
learning_rate = 1e-3

# Sample vocabulary and data (replace this with actual data loading)
vocab = {'<PAD>': 0, '<UNK>': 1, 'example': 2, 'text': 3}  # Simplified vocab example
texts = ["example text", "another example text"]  # Replace with actual data

# Data loading
dataset = TextDataset(texts, vocab)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = VAE(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x in data_loader:
        optimizer.zero_grad()
        x_reconstructed, style_mean, style_logvar, content_mean, content_logvar = model(x)
        loss = vae_loss(x_reconstructed, x, style_mean, style_logvar, content_mean, content_logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")
