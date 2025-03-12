import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from task1 import WordPieceTokenizer #TAsk-1!

class NeuralLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_size=3):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.sentences = self._read_file(file_path)
        self.data = []  # (context, target)
        self._build_dataset()

    def _read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _build_dataset(self):
        for sentence in self.sentences:
            processed = self.tokenizer.preprocess_data(sentence)
            tokens = self.tokenizer.tokenize(processed)
            if len(tokens) < self.context_size + 1:
                continue
            for i in range(self.context_size, len(tokens)):
                context = tokens[i - self.context_size:i]
                target = tokens[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context_tokens, target_token = self.data[index]
        context_indices = [self._token_to_idx(tok) for tok in context_tokens]
        target_idx = self._token_to_idx(target_token)
        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

    def _token_to_idx(self, token):
        if token in self.tokenizer.vocab:
            return self.tokenizer.vocab.index(token)
        else:
            return self.tokenizer.vocab.index("[UNK]")
#__________________________________________________________________NLM_CLASSES!_______________________________

# LM1: Basic one-hidden-layer MLP with ReLU
class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim=128):
        super(NeuralLM1, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embeddings(x)
        flat = emb.view(emb.size(0), -1)
        hidden = self.relu(self.fc(flat))
        return self.out_layer(hidden)


# LM2: Three hidden layers with Tanh and dropout
class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim=256, dropout_rate=0.3):
        super(NeuralLM2, self).__init__()
        self.embeds = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.out_fc = nn.Linear(hidden_dim // 4, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embeds(x).view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh(self.fc3(x))
        return self.out_fc(x)

# LM3: Three hidden layers with LeakyReLU and dropout everywhere
class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim=256, dropout_rate=0.5):
        super(NeuralLM3, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.l1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out_layer = nn.Linear(hidden_dim // 2, vocab_size)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.embed(x).view(x.size(0), -1)
        x = self.leaky(self.l1(x))
        x = self.dropout(x)
        x = self.leaky(self.l2(x))
        x = self.dropout(x)
        x = self.leaky(self.l3(x))
        return self.out_layer(x)

def run_epoch(model, loader, criterion, device, optimizer=None):
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        train_losses.append(train_loss)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Epoch-{epoch+1:02d}/{epochs:02d}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


def compute_accuracy(model, loader, device):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def compute_perplexity(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
    avg_loss = total_loss / count if count else float('inf')
    return np.exp(avg_loss)


def plot_losses(loss_map, epochs, save_path="training_validation_loss.png"):
    plt.figure(figsize=(10, 6))
    for name, (train_loss, val_loss) in loss_map.items():
        plt.plot(range(1, epochs + 1), train_loss, label=f"{name} Train")
        plt.plot(range(1, epochs + 1), val_loss, label=f"{name} Val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch for Neural LM Variations")
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def prepare_context(sentence, tokenizer, context_size):
    processed = tokenizer.preprocess_data(sentence)
    tokens = tokenizer.tokenize(processed)
    pad_token = "[PAD]"
    if len(tokens) < context_size:
        tokens = [pad_token] * (context_size - len(tokens)) + tokens
    else:
        tokens = tokens[-context_size:]
    return tokens

def tokens_to_indices(tokens, tokenizer):
    return [
        tokenizer.vocab.index(tok) if tok in tokenizer.vocab else tokenizer.vocab.index("[UNK]")
        for tok in tokens
    ]

# The nomenclature for "logits and tensor was also inspired from the resources named in the Documentation!!!"
def get_model_prediction(model, context_indices, device, mode="greedy"):
    inp_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(inp_tensor)
        chosen_idx = torch.argmax(logits, dim=1).item()
    return chosen_idx

def update_context(tokens, new_token): #This felt the only way, however I have another idea too, Remember to discuss with SIR!
    return tokens[1:] + [new_token]

def predict_next_tokens(model, sentence, tokenizer, context_size, num_tokens=3, device="cpu"):
    model.eval()
    context_tokens = prepare_context(sentence, tokenizer, context_size)
    predictions = []
    for i in range(num_tokens):
        context_indices = tokens_to_indices(context_tokens, tokenizer)
        chosen_idx = get_model_prediction(model, context_indices, device)
        next_token = tokenizer.vocab[chosen_idx]
        predictions.append(next_token)
        context_tokens = update_context(context_tokens, next_token)
        print(f"Step-{i+1}: context:{context_tokens} predi:'{next_token}'")
    
    return predictions

def initialize_model(model_class, vocab_size, embed_dim, context_size, pretrained_embeds, device, **kwargs):
    model = model_class(vocab_size, embed_dim, context_size, **kwargs).to(device)
    if hasattr(model, 'embeddings'):
        model.embeddings.weight.data.copy_(pretrained_embeds)
    elif hasattr(model, 'embeds'):
        model.embeds.weight.data.copy_(pretrained_embeds)
    elif hasattr(model, 'embed'):
        model.embed.weight.data.copy_(pretrained_embeds)
    else:
        raise ValueError("No embed attr!")
    return model

def run_training_pipeline(models_info, train_loader, val_loader, criterion, epochs, device):
    results = {}
    for name, model, optimizer in models_info:
        print(f"Training-{name}")
        train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)
        results[name] = (train_loss, val_loss)
    return results


if __name__ == "__main__":
    TRAIN_FILE = "corpus.txt"     
    TEST_FILE = "test.txt"
    CONTEXT_SIZE = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 0.001
    device = torch.device("cpu")

# # Load the tokenizer and its vocabulary.
    tokenizer = WordPieceTokenizer()
    with open("vocabulary.txt", "r", encoding="utf-8") as vf:
        tokenizer.vocab = [line.strip() for line in vf]
    vocab_size = len(tokenizer.vocab)
    print("Vocab size:", vocab_size)

#  #(80-20 Split!!!!)
    dataset = NeuralLMDataset(TRAIN_FILE, tokenizer, context_size=CONTEXT_SIZE)
    total_samples = len(dataset)
    val_samples = int(0.2 * total_samples)
    train_samples = total_samples - val_samples
    train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# #Checkpoint of Task-2!
    ckpt = torch.load("word2vec_checkpoint.pth", map_location=device)
    pretrained_embeds = ckpt['model_state_dict']['embeddings.weight']
    embed_dim = pretrained_embeds.size(1)
    print("Embedding dimension:", embed_dim)

    model1 = initialize_model(NeuralLM1, vocab_size, embed_dim, CONTEXT_SIZE, pretrained_embeds, device, hidden_dim=128)
    model2 = initialize_model(NeuralLM2, vocab_size, embed_dim, CONTEXT_SIZE, pretrained_embeds, device, hidden_dim=256, dropout_rate=0.3)
    model3 = initialize_model(NeuralLM3, vocab_size, embed_dim, CONTEXT_SIZE, pretrained_embeds, device, hidden_dim=256, dropout_rate=0.5)

    optimizer1 = optim.Adam(model1.parameters(), lr=LR)
    optimizer2 = optim.Adam(model2.parameters(), lr=LR)
    optimizer3 = optim.Adam(model3.parameters(), lr=LR)

# Inspiration for "Criterion" from Pytorch Documentation and codes on Github with other NLP tasks!!!!
# The links for which have been attached in the report!!!!!!
    criterion = nn.CrossEntropyLoss()
    models_to_train = [
        ("NeuralLM1", model1, optimizer1),
        ("NeuralLM2", model2, optimizer2),
        ("NeuralLM3", model3, optimizer3)
    ]

# A custom pipeline for RUNNING THE CODES FOR THE 3 Neural LM MODELS!!
    training_results = run_training_pipeline(models_to_train, train_loader, val_loader, criterion, NUM_EPOCHS, device)
    plot_losses(training_results, NUM_EPOCHS)

    for name, model, _ in models_to_train:
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)
        train_ppl = compute_perplexity(model, train_loader, criterion, device)
        val_ppl = compute_perplexity(model, val_loader, criterion, device)
        print(f"\n{name} - Accuracy: Train:{train_acc*100:.2f}%, Val:{val_acc*100:.2f}%")
        print(f"{name} - Perplexity: Train:{train_ppl:.2f}, Val:{val_ppl:.2f}")

    with open(TEST_FILE, "r") as testf:
        for line in testf:
            sentence = line.strip()
            if not sentence:
                continue
            predicted_tokens = predict_next_tokens(model3, sentence, tokenizer, CONTEXT_SIZE, num_tokens=3, device=device)
            print(f"\nInput Sentence: {sentence}")
            print(f"Predited Tokens: {predicted_tokens}")

    torch.save(model1.state_dict(), "neural_lm1.pth")
    torch.save(model2.state_dict(), "neural_lm2.pth")
    torch.save(model3.state_dict(), "neural_lm3.pth")