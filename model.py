# Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.fc_mean_style = nn.Linear(hidden_dim * 2, style_dim)
        self.fc_logvar_style = nn.Linear(hidden_dim * 2, style_dim)
        self.fc_mean_content = nn.Linear(hidden_dim * 2, content_dim)
        self.fc_logvar_content = nn.Linear(hidden_dim * 2, content_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        
        style_mean = self.fc_mean_style(h)
        style_logvar = self.fc_logvar_style(h)
        content_mean = self.fc_mean_content(h)
        content_logvar = self.fc_logvar_content(h)

        return style_mean, style_logvar, content_mean, content_logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + style_dim + content_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, style, content):
        x = self.embedding(x)
        style_content = torch.cat((style, content), dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, style_content), dim=2)
        output, _ = self.rnn(x)
        return self.fc_out(output)


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, style_dim, content_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, style_dim, content_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        style_mean, style_logvar, content_mean, content_logvar = self.encoder(x)
        
        style = self.reparameterize(style_mean, style_logvar)
        content = self.reparameterize(content_mean, content_logvar)

        x_reconstructed = self.decoder(x, style, content)
        return x_reconstructed, style_mean, style_logvar, content_mean, content_logvar


def vae_loss(reconstructed, x, style_mean, style_logvar, content_mean, content_logvar):
    recon_loss = F.cross_entropy(reconstructed.view(-1, reconstructed.size(-1)), x.view(-1))
    kl_style = -0.5 * torch.sum(1 + style_logvar - style_mean.pow(2) - style_logvar.exp())
    kl_content = -0.5 * torch.sum(1 + content_logvar - content_mean.pow(2) - content_logvar.exp())
    return recon_loss + kl_style + kl_content


class CounterfactualReasoning(nn.Module):
    def __init__(self, style_dim):
        super(CounterfactualReasoning, self).__init__()
        self.fc = nn.Linear(style_dim, 1)
    
    def forward(self, s, target_confidence):
        adjusted_s = s.clone().detach()
        adjusted_s.requires_grad = True
        optimizer = torch.optim.Adam([adjusted_s], lr=0.01)
        
        for _ in range(50):
            optimizer.zero_grad()
            prediction = torch.sigmoid(self.fc(adjusted_s))
            loss = F.mse_loss(prediction, target_confidence) + F.l1_loss(adjusted_s, s)
            loss.backward()
            optimizer.step()
        
        return adjusted_s
