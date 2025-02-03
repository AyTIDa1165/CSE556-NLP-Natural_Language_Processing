import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from radam import RAdam
from adabound import AdaBound
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Add parent directory to the system path to allow importing Task1's WordPieceTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Task1.word_piece_tokenizer import WordPieceTokenizer  # Importing the tokenizer from Task 1


# Custom Dataset Class for Word2Vec
class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, tokenizer, window_size=2, max_context_len=4):
        """
        Initializes the dataset for Word2Vec training.
        :param corpus_file: Path to the text corpus file.
        :param tokenizer: Tokenizer instance to tokenize sentences.
        :param window_size: Size of the context window around the target word.
        :param max_context_len: Maximum number of context words to consider.
        """
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_context_len = max_context_len
        self.corpus = self._load_corpus(corpus_file)  # Load the corpus from the file
        self.tokenized_corpus = [self.tokenizer.tokenize(sentence) for sentence in self.corpus]  # Tokenize each sentence
        self.vocab = self.tokenizer.vocab  # Vocabulary from the tokenizer
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}  # Map words to indices
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}  # Map indices to words
        self.data = self._generate_cbow_data()  # Generate CBOW data pairs (context, target)

    def _load_corpus(self, corpus_file):
        """
        Loads the corpus from a text file.
        :param corpus_file: Path to the text corpus file.
        :return: List of sentences from the corpus.
        """
        with open(corpus_file, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]

    def _generate_cbow_data(self):
        """
        Generates CBOW training data by pairing context words with target words.
        :return: List of tuples (context_words, target_word).
        """
        cbow_data = []
        for sentence in self.tokenized_corpus:
            for i, target_word in enumerate(sentence):
                # Collect context words within the window size
                context_words = []
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if i != j:
                        context_words.append(sentence[j])
                cbow_data.append((context_words, target_word))
        return cbow_data

    def __len__(self):
        """
        Returns the number of CBOW data pairs in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a CBOW data pair at the given index.
        :param idx: Index of the data pair.
        :return: Tuple of (context_indices, target_index).
        """
        context_words, target_word = self.data[idx]
        # Pad context words to max_context_len
        padded_context_indices = self._pad_context([self.word_to_idx[word] for word in context_words])
        target_index = self.word_to_idx[target_word]
        return torch.tensor(padded_context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)

    def _pad_context(self, context_indices):
        """
        Pads or truncates context indices to ensure fixed-length input.
        :param context_indices: List of context word indices.
        :return: Padded list of context indices.
        """
        # Pad with the index of the [PAD] token (assumed to be at index 1 in the vocabulary)
        pad_idx = self.word_to_idx["[PAD]"] # Index of the [PAD] token
        padded = context_indices[:self.max_context_len]  # Truncate if longer than max_context_len
        padded += [pad_idx] * (self.max_context_len - len(padded))  # Pad if shorter than max_context_len
        return padded


# Word2Vec Model Class
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the Word2Vec model.
        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimensionality of the word embeddings.
        """
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # Embedding Layer
        self.linear = nn.Linear(embedding_dim, vocab_size) # Linear layer for prediction

    def forward(self, inputs):
        """
        Forward pass of the Word2Vec model.
        :param inputs: Context word indices.
        :return: Predicted logits for the target word.
        """
        embedded = self.embeddings(inputs).mean(dim=1)  # Average embeddings of context words
        output = self.linear(embedded) # Pass through linear layer
        return output


# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the Word2Vec model and evaluates on the validation set.
    :param model: Word2Vec model instance.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param criterion: Loss function.
    :param optimizer: Optimizer for training.
    :param num_epochs: Number of training epochs.
    :param device: Device (CPU/GPU) to run the model on.
    :return: Lists of training and validation losses.
    """
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad() # Zero gradients
            outputs = model(contexts) # Forward pass
            loss = criterion(outputs, targets) # Loss Compute
            loss.backward() # Backward pass
            optimizer.step() # Update weights
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


# Main Script
if __name__ == "__main__":
    # Paths and Hyperparameters
    corpus_file = "NLP A1\\Task1\\corpus.txt"  # Path to the corpus file
    vocabulary_file = "NLP A1\\Task2\\complete_vocabulary.txt"  # Path to the vocabulary file
    checkpoint_file = "NLP A1\\Task2\\word2vec_checkpoint.pth"  # Path to save the model checkpoint
    window_size = 2  # Context window size
    max_context_len = 4  # Maximum number of context words
    embedding_dim = 100  # Dimensionality of word embeddings
    batch_size = 32  # Batch size for training
    # Trying results with various no. of training epochs.
    # num_epochs = 10
    num_epochs = 30
    # num_epochs = 100
    # num_epochs = 1000
    learning_rate = 0.001 # Learning rate for the optimizer
    val_split = 0.2  # 20% of the data for validation

    # Load Tokenizer
    tokenizer = WordPieceTokenizer()
    tokenizer.vocab = []
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        # Load the complete vocabulary into the tokenizer
        # The above line ensures that the tokens from the complete vocabulary file are being stored in tokenizer.
        tokenizer.vocab = [line.strip() for line in f]

    # Create Dataset and DataLoader
    dataset = Word2VecDataset(corpus_file, tokenizer, window_size, max_context_len)
    val_size = int(val_split * len(dataset))  # Calculate validation set size
    train_size = len(dataset) - val_size  # Calculate training set size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Split dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for training
    # Shuffle = True for the training DataLoader ensures that the model sees a diverse set of examples in each batch, 
    # promoting better generalization.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for validation
    # Shuffle = False for the validation DataLoader is sufficient because the order of examples does not affect the 
    # evaluation process.

    # Initialize Model, Loss, and Optimizer
    vocab_size = len(tokenizer.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    model = Word2VecModel(vocab_size, embedding_dim).to(device) # Initialize the Word2Vec model
    criterion = nn.CrossEntropyLoss() # Loss function
    # Basic optimisers
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Advanced optimisers
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)     # Initialize AdamW optimizer
    # optimizer = RAdam(model.parameters(), lr=learning_rate)                        # Initialize RAdam optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True) # SGD with Momentum and Nesterov Momentum
    optimizer = AdaBound(model.parameters(), lr=learning_rate, final_lr=0.1)       ######## BEST Results as of now.

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
        """
        Computes cosine similarity between two words.
        :param word1: First word.
        :param word2: Second word.
        :return: Cosine similarity score.
        """
        idx1, idx2 = dataset.word_to_idx[word1], dataset.word_to_idx[word2]
        vec1, vec2 = embeddings[idx1], embeddings[idx2]
        return cosine_similarity([vec1], [vec2])[0][0]

    # Triplets Used:
    triplets = [
        ("happy", "joyful", "sad"),
        ("lonely", "isolated", "loved")
    ]
    for triplet in triplets:
        sim1 = get_cosine_similarity(triplet[0], triplet[1]) # Similarity between similar words
        sim2 = get_cosine_similarity(triplet[0], triplet[2]) # Similarity between opposite words
        print(f"Cosine Similarity between '{triplet[0]}' and '{triplet[1]}': {sim1:.4f}")
        print(f"Cosine Similarity between '{triplet[0]}' and '{triplet[2]}': {sim2:.4f}")

# Key Observations:
# 1. Adabound emerged as the top-performing optimizer, achieving the highest accuracy on both training and validation datasets.
# 2. Furthermore, Adabound demonstrated superior performance in terms of cosine similarity accuracy. Specifically, it yielded 
# higher similarity scores for semantically similar words and lower scores for antonymous words.