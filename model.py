import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(Encoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Bi-GRU layer
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers to compute mean and log variance for style and content
        self.fc_style_mean = nn.Linear(hidden_dim * 2, style_dim)
        self.fc_style_logvar = nn.Linear(hidden_dim * 2, style_dim)
        self.fc_content_mean = nn.Linear(hidden_dim * 2, content_dim)
        self.fc_content_logvar = nn.Linear(hidden_dim * 2, content_dim)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)
        
        # Bi-GRU layer
        _, h = self.rnn(x)
        h = torch.cat((h[-2], h[-1]), dim=1)  # Concatenate the last hidden states from both directions
        
        # Compute style and content mean and log variance
        style_mean = self.fc_style_mean(h)
        style_logvar = self.fc_style_logvar(h)
        content_mean = self.fc_content_mean(h)
        content_logvar = self.fc_content_logvar(h)

        return style_mean, style_logvar, content_mean, content_logvar


class Reparameterization(nn.Module):
    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(Decoder, self).__init__()
        # Embedding layer for decoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GRU layer for decoding
        self.rnn = nn.GRU(embed_dim + style_dim + content_dim, hidden_dim, batch_first=True)
        
        # Output fully connected layer to map to vocabulary size
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, style, content):
        # Embed input tokens
        x = self.embedding(x)
        
        # Concatenate style and content embeddings with each input embedding
        style_content = torch.cat((style, content), dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, style_content), dim=2)
        
        # GRU layer for decoding
        output, _ = self.rnn(x)
        
        # Map each time step to a vocabulary distribution
        return self.fc_out(output)


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(VAE, self).__init__()
        # Encoder and decoder
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)

    def forward(self, x):
        # Encode to obtain means and log variances
        style_mean, style_logvar, content_mean, content_logvar = self.encoder(x)
        
        # Reparameterize to get sampled embeddings
        style = Reparameterization.reparameterize(style_mean, style_logvar)
        content = Reparameterization.reparameterize(content_mean, content_logvar)

        # Decode using sampled embeddings
        x_reconstructed = self.decoder(x, style, content)
        
        return x_reconstructed, style_mean, style_logvar, content_mean, content_logvar


# Loss functions for VAE
def vae_loss(reconstructed, x, style_mean, style_logvar, content_mean, content_logvar):
    # Reconstruction loss
    recon_loss = F.cross_entropy(reconstructed.view(-1, reconstructed.size(-1)), x.view(-1))
    
    # KL Divergence for style and content
    kl_style = -0.5 * torch.sum(1 + style_logvar - style_mean.pow(2) - style_logvar.exp())
    kl_content = -0.5 * torch.sum(1 + content_logvar - content_mean.pow(2) - content_logvar.exp())
    
    return recon_loss + kl_style + kl_content


# Counterfactual Reasoning Module
class CounterfactualReasoning(nn.Module):
    def __init__(self, style_dim):
        super(CounterfactualReasoning, self).__init__()
        # MLP classifier for style adjustment
        self.fc = nn.Linear(style_dim, 1)

    def forward(self, s, target_confidence):
        # Initialize s_prime as a clone of s with gradient tracking
        s_prime = s.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([s_prime], lr=0.01)

        for _ in range(50):  # Perform 50 optimization steps to adjust s_prime
            optimizer.zero_grad()
            prediction = torch.sigmoid(self.fc(s_prime))
            mse_loss = (prediction - target_confidence).pow(2).mean()
            l1_loss = F.l1_loss(s_prime, s)
            total_loss = mse_loss + l1_loss
            total_loss.backward()
            optimizer.step()

        return s_prime


# Putting everything together: model with VAE and counterfactual reasoning
class StyleTransferModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(StyleTransferModel, self).__init__()
        self.vae = VAE(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)
        self.counterfactual_reasoning = CounterfactualReasoning(style_dim)

    def forward(self, x, target_confidence=None):
        # Forward pass through VAE
        x_reconstructed, style_mean, style_logvar, content_mean, content_logvar = self.vae(x)
        
        # Counterfactual adjustment for style
        if target_confidence is not None:
            s_prime = self.counterfactual_reasoning(style_mean, target_confidence)
        else:
            s_prime = style_mean  # Use original style embedding if no target confidence is provided

        return x_reconstructed, style_mean, content_mean, s_prime

