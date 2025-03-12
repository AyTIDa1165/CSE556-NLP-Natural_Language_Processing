import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
# from radam import RAdam
# from adabound import AdaBound
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Import the WordPieceTokenizer from task1.py (assumes they are in the same directory)
from task1 import WordPieceTokenizer

# Custom Dataset Class for Word2Vec
class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, tokenizer, window_size=2, max_context_len=4):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_context_len = max_context_len
        self.corpus = self._load_corpus(corpus_file)
        self.tokenized_corpus = [self.tokenizer.tokenize(sentence) for sentence in self.corpus]
        self.vocab = self.tokenizer.vocab
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.data = self._generate_cbow_data()

    def _load_corpus(self, corpus_file):
        with open(corpus_file, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]

    def _generate_cbow_data(self):
        cbow_data = []
        for sentence in self.tokenized_corpus:
            for i, target_word in enumerate(sentence):
                context_words = []
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if i != j:
                        context_words.append(sentence[j])
                cbow_data.append((context_words, target_word))
        return cbow_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_words, target_word = self.data[idx]
        # Pad context words to max_context_len
        padded_context_indices = self._pad_context([self.word_to_idx[word] for word in context_words])
        target_index = self.word_to_idx[target_word]
        return torch.tensor(padded_context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)

    def _pad_context(self, context_indices):
        # Pad with the index of the [PAD] token (assumed to be at index 1 in the vocabulary)
        pad_idx = self.word_to_idx["[PAD]"]
        padded = context_indices[:self.max_context_len]  # Truncate if longer than max_context_len
        padded += [pad_idx] * (self.max_context_len - len(padded))  # Pad if shorter than max_context_len
        return padded

# Word2Vec Model Class
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedded = self.embeddings(inputs).mean(dim=1)  # Average embeddings of context words
        output = self.linear(embedded)
        return output

# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(contexts)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for contexts, targets in val_loader:
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = model(contexts)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    # Paths and Hyperparameters
    corpus_file = "corpus.txt"
    vocabulary_file = "vocabulary.txt"
    checkpoint_file = "word2vec_checkpoint.pth"
    window_size = 2
    max_context_len = 4  # Maximum number of context words
    embedding_dim = 100
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    val_split = 0.2  # 20% of the data for validation

    # Load Tokenizer and Vocabulary
    tokenizer = WordPieceTokenizer()
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        tokenizer.vocab = [line.strip() for line in f]

    # Create Dataset and DataLoader
    dataset = Word2VecDataset(corpus_file, tokenizer, window_size, max_context_len)
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    vocab_size = len(tokenizer.vocab)
    device = torch.device("cpu")
    model = Word2VecModel(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    # You can choose an optimizer. Here we use Adam.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Save Model Checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, checkpoint_file)
    print(f"Model checkpoint saved to {checkpoint_file}\n")

    # Plot Training and Validation Loss
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epochs")
    plt.legend()
    plt.savefig("training_and_validation_loss.png")
    plt.show()

    # Compute Cosine Similarity Triplets
    embeddings = model.embeddings.weight.detach().cpu().numpy()

    def get_cosine_similarity(word1, word2):
        idx1, idx2 = dataset.word_to_idx[word1], dataset.word_to_idx[word2]
        vec1, vec2 = embeddings[idx1], embeddings[idx2]
        return cosine_similarity([vec1], [vec2])[0][0]

    # triplets = [
    #     ("happy", "joyful", "sad"),
    #     ("lonely", "isolated", "loved")
    # ]
    # for triplet in triplets:
    #     sim1 = get_cosine_similarity(triplet[0], triplet[1])
    #     sim2 = get_cosine_similarity(triplet[0], triplet[2])
    #     print(f"Cosine Similarity between '{triplet[0]}' and '{triplet[1]}': {sim1:.4f}")
    #     print(f"Cosine Similarity between '{triplet[0]}' and '{triplet[2]}': {sim2:.4f}")