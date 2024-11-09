import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  
from dataloader import *
from model import * 
from nltk.translate.bleu_score import sentence_bleu
import nltk

def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_length = max(len(seq) for seq in inputs)
    
    # Convert each sequence to a list, pad with 0, and convert to tensor
    padded_inputs = [torch.cat([seq, torch.zeros(max_length - len(seq), dtype=torch.long)]) for seq in inputs]
    lengths = [len(seq) for seq in inputs]
    
    return torch.stack(padded_inputs), torch.tensor(labels, dtype=torch.float), lengths


# Hyperparameters
num_epochs = 10
learning_rate = 0.001
target_confidence = 0.8


data_dir = "./data/sentiment_style_transfer/yelp"
vocab = build_vocab(data_dir)
dataset = TextDataset(data_dir, vocab)
data_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StyleTransferModel(len(vocab), 300, 256, 16, 128).to(device)  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    
    for input_tokens, labels, lengths in progress_bar:
        input_tokens = input_tokens.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        x_reconstructed, style_mean, content_mean, s_prime = model(input_tokens, target_confidence)
        style_logvar = torch.zeros_like(style_mean)
        content_logvar = torch.zeros_like(content_mean)
        loss = vae_loss(x_reconstructed, input_tokens, style_mean, style_logvar, content_mean, content_logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=epoch_loss / (progress_bar.n + 1))
    
    print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {epoch_loss / len(data_loader)}")

# Download necessary NLTK resources
nltk.download('punkt')

# Function to convert token IDs back to words using the vocabulary
def tokens_to_words(token_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    return [inv_vocab.get(token_id, '<UNK>') for token_id in token_ids if token_id != 0]  # Exclude padding

# Function to calculate BLEU score for a batch
def calculate_bleu_score(data_loader, model, vocab, device):
    model.eval()  # Set the model to evaluation mode
    total_bleu_score = 0
    num_sentences = 0

    with torch.no_grad():
        for input_tokens, _, lengths in data_loader:
            input_tokens = input_tokens.to(device)
            x_reconstructed, _, _, _ = model(input_tokens)
            x_reconstructed = x_reconstructed.argmax(dim=-1)  # Get the predicted token IDs

            # Calculate BLEU score for each sentence
            for i in range(len(input_tokens)):
                original_sentence = tokens_to_words(input_tokens[i].tolist(), vocab)
                reconstructed_sentence = tokens_to_words(x_reconstructed[i].tolist(), vocab)

                # Calculate BLEU score
                bleu_score = sentence_bleu([original_sentence], reconstructed_sentence)
                total_bleu_score += bleu_score
                num_sentences += 1

    # Return the average BLEU score
    return total_bleu_score / num_sentences if num_sentences > 0 else 0

# Calculate the BLEU score
bleu_score = calculate_bleu_score(data_loader, model, vocab, device)
print(f"Average BLEU Score: {bleu_score:.4f}")