import torch
import torch.optim as optim
# import dataloader
from torch.utils.data import DataLoader
from model import StyleTransferModel, vae_loss  # Import model and loss function
from dataloader import TextDataset, collate_fn, build_vocab  # Import dataset and collate function

# Hyperparameters
num_epochs = 10
learning_rate = 0.001
target_confidence = 0.8  # Desired style confidence for counterfactual adjustment

# Initialize model, optimizer, and data loader
vocab_size = 10000  # Example vocabulary size
embed_dim = 300
hidden_dim = 256
style_dim = 16
content_dim = 128

# Assuming data_dir points to the location of your dataset
data_dir = "./data/sentiment_style_transfer/yelp"
vocab = build_vocab(data_dir)  # Build vocabulary
dataset = TextDataset(data_dir, vocab)  # Initialize dataset
data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)  # Initialize DataLoader

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StyleTransferModel(vocab_size, embed_dim, hidden_dim, style_dim, content_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (input_tokens, labels, lengths) in enumerate(data_loader):
        # Move inputs to device (if using GPU)
        input_tokens = input_tokens.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass through VAE with counterfactual adjustment
        x_reconstructed, style_mean, content_mean, s_prime = model(input_tokens, target_confidence=target_confidence)
        
        # Calculate VAE loss (reconstruction + KL divergence)
        style_logvar = torch.zeros_like(style_mean)  # Assuming style_logvar is zero-initialized for simplicity
        content_logvar = torch.zeros_like(content_mean)
        loss = vae_loss(x_reconstructed, input_tokens, style_mean, style_logvar, content_mean, content_logvar)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for monitoring
        epoch_loss += loss.item()
    
    # Logging epoch loss
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(data_loader)}")
