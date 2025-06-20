###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Categories for the dataset
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'}


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    :param model: PyTorch model
    :return: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Get data function
def get_data(categories=None, portion=1.0):
    """
    Get data for given categories and portion of the training data.
    :param categories: List of categories to include.
    :param portion: Portion of the data to use for training.
    :return: x_train, y_train, x_test, y_test
    """
    # Fetch the dataset
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # Training data
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = np.array(data_train.target[:train_len])

    # Remove empty entries in training data
    non_empty_train = x_train != ""
    x_train, y_train = x_train[non_empty_train].tolist(), y_train[non_empty_train].tolist()

    # Test data
    x_test = np.array(data_test.data)
    y_test = np.array(data_test.target)

    # Remove empty entries in test data
    non_empty_test = x_test != ""
    x_test, y_test = x_test[non_empty_test].tolist(), y_test[non_empty_test].tolist()

    return x_train, y_train, x_test, y_test

# Dataset preprocessing and data loading
def preprocess_data(portion=1.0):
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Convert text to TFIDF vectors
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
    x_test_tfidf = vectorizer.transform(x_test).toarray()
    return x_train_tfidf, y_train, x_test_tfidf, y_test

# Custom Dataset for PyTorch
class TextDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Single-layer MLP (Log-linear classifier)
class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SingleLayerMLP, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Multi-layer MLP
class MultiLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MultiLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Training function
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Evaluate on validation data
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
    return train_losses, val_accuracies

# Evaluation function
def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return accuracy_score(all_labels, all_predictions)


# Plot results
def plot_results(train_losses, val_accuracies, portion, is_mlp):
    model_type = "Multi-layer MLP" if is_mlp else "Single-layer MLP"
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f"Train Loss (Portion: {portion})")
    plt.title(f"{model_type} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label=f"Validation Accuracy (Portion: {portion})")
    plt.title(f"{model_type} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# MLP Classification wrapper function
def MLP_classification(portion=1.0, model="single", print_params=False):
    """
    Perform classification with either a single-layer or multi-layer MLP.
    :param portion: Portion of the data to use for training.
    :param model: Type of MLP to use - "single" for Single-layer MLP, "multi" for Multi-layer MLP.
    :param print_params: If True, print the number of parameters for the model (only once).
    :return: None
    """
    x_train, y_train, x_test, y_test = preprocess_data(portion=portion)

    # Prepare data loaders
    train_dataset = TextDataset(x_train, y_train)
    test_dataset = TextDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Model and device setup
    input_dim = x_train.shape[1]
    num_classes = len(category_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select model based on the type
    if model == "single":
        model_type = SingleLayerMLP(input_dim, num_classes).to(device)
        if print_params:
            num_params = count_trainable_parameters(model_type)
            print(f"Number of trainable parameters in Single-layer MLP: {num_params}")
    elif model == "multi":
        hidden_dim = 500
        model_type = MultiLayerMLP(input_dim, hidden_dim, num_classes).to(device)
        if print_params:
            num_params = count_trainable_parameters(model_type)
            print(f"Number of trainable parameters in Multi-layer MLP: {num_params}")
    else:
        raise ValueError("Invalid model_type. Use 'single' or 'multi'.")

    # Train the model
    train_losses, val_accuracies = train_model(model_type, train_loader, test_loader, device=device)

    # Plot results
    is_mlp = model == "multi"
    plot_results(train_losses, val_accuracies, portion, is_mlp)


# Q3
def transformer_classification(portion=1.0, print_params=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import evaluate
    from tqdm import tqdm

    # Dataset class for tokenized data
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        """
        model.train()
        total_loss = 0.
        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, metric, dev='cpu'):
        """
        Evaluate the model and calculate accuracy
        """
        model.eval()
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())
        return metric.compute()['accuracy']

    # Load data
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Initialize model, tokenizer, optimizer, and evaluation metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    if print_params:
        num_params = count_trainable_parameters(model)
        print(f"Number of trainable parameters in Transformer model: {num_params}")

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    metric = evaluate.load("accuracy")

    # Tokenize data
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=512)

    # Create datasets and loaders
    train_dataset = Dataset(train_encodings, y_train)
    val_dataset = Dataset(test_encodings, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training and evaluation
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev=dev)
        train_losses.append(train_loss)

        val_accuracy = evaluate_model(model, val_loader, metric, dev=dev)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Plot results
    plot_transformer_results(train_losses, val_accuracies, portion)


# Plotting function for Transformer results
def plot_transformer_results(train_losses, val_accuracies, portion):
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title(f"Transformer - Training Loss (Portion: {portion})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o')
    plt.title(f"Transformer - Validation Accuracy (Portion: {portion})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]

    # Q1 - Single-layer MLP
    print("\nRunning Single-layer MLP experiments:")
    for i, portion in enumerate(portions):
        print(f"\nRunning Single-layer MLP for Portion: {portion}")  # Print for all portions
        if i == 0:  # Print trainable parameters for the first portion
            MLP_classification(portion=portion, model="single", print_params=True)
        else:
            MLP_classification(portion=portion, model="single", print_params=False)

    # Q2 - Multi-layer MLP
    print("\nRunning Multi-layer MLP experiments:")
    for i, portion in enumerate(portions):
        print(f"\nRunning Multi-layer MLP for Portion: {portion}")  # Print for all portions
        if i == 0:  # Print trainable parameters for the first portion
            MLP_classification(portion=portion, model="multi", print_params=True)
        else:
            MLP_classification(portion=portion, model="multi", print_params=False)

    # Q3 - Transformer
    print("\nTransformer results:")
    for i, portion in enumerate(portions[:2]):  # Only run for portions 0.1 and 0.2
        print(f"\nRunning Transformer experiment for Portion: {portion}")  # Print for all portions
        if i == 0:  # Print trainable parameters for the first portion
            transformer_classification(portion=portion, print_params=True)
        else:
            transformer_classification(portion=portion, print_params=False)
