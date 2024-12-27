import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloader import *
from transform import *
from sklearn.metrics import roc_auc_score, average_precision_score

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data loading and preprocessing
folder = '../preops_cv/'
outcomes = ['cardiac', 'AF', 'arrest', 'DVT_PE', 'post_aki_status', 'total_blood']

# Load inputs
with open('../inputs_new.pickle', 'rb') as handle:
    inputs = pkl.load(handle)

# Rename columns
inputs.outcomes.rename(columns={'orlogid_encoded': 'orlogid'}, inplace=True)
inputs.texts.rename(columns={'orlogid_encoded': 'orlogid'}, inplace=True)


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Neural Network model
class MultiLabelNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out  # No activation function here; we'll use BCEWithLogitsLoss


# Best Hyper Parameters
num_folds = 5
num_epochs = 5
batch_size = 64
learning_rate = 0.001
hidden_size = 128
output_size = len(outcomes)  # 6 outcomes

# Initialize dictionaries to store AUROC and AUPRC per fold per label
fold_aurocs = {label: [] for label in outcomes}
fold_auprcs = {label: [] for label in outcomes}

for i in range(num_folds):
    foldername = folder + 'arrest' + '/'
    print(f"\nFold {i + 1}/{num_folds}")

    # Load data
    X_train = pkl.load(open(foldername + f'X_train_{i}.pickle', 'rb'))
    X_test = pkl.load(open(foldername + f'X_test_{i}.pickle', 'rb'))
    train_data = pkl.loads(open('../preops/X_sampled_non_cardiac.pickle', 'rb').read())
    X_train2 = train_data
    train_outcome = pkl.loads(open('../preops/outcome_data_non_cardiac.pickle', 'rb').read())
    y_train2 = train_outcome[outcomes].to_numpy()
    y_train2 = np.where(y_train2 > 0, 1, 0)
    ids_train = pkl.load(open(foldername + f'train_ids_{i}.pickle', 'rb'))
    ids_test = pkl.load(open(foldername + f'test_ids_{i}.pickle', 'rb'))

    X_train = np.concatenate((X_train, X_train2), axis=0)
    # Remove column 163
    X_train = np.delete(X_train, 163, axis=1)
    X_test = np.delete(X_test, 163, axis=1)

    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Get y_train and y_test as multi-label targets
    train_outcome = inputs.outcomes[inputs.outcomes.orlogid.isin(ids_train)].copy()
    y_train = train_outcome[outcomes].to_numpy()
    y_train = np.where(y_train > 0, 1, 0)
    y_train = np.concatenate((y_train, y_train2), axis=0)
    test_outcome = inputs.outcomes[inputs.outcomes.orlogid.isin(ids_test)].copy()
    y_test = test_outcome[outcomes].to_numpy()
    y_test = np.where(y_test > 0, 1, 0)

    # Build datasets and dataloaders
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer
    input_size = X_train.shape[1]
    model = MultiLabelNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Print loss every few epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluation
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            outputs = torch.sigmoid(outputs).cpu()  # Apply sigmoid to get probabilities
            all_outputs.append(outputs)
            all_targets.append(y_batch)

    all_outputs = torch.cat(all_outputs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate AUROC and AUPRC for each label
    for idx in range(output_size):
        try:
            auroc = roc_auc_score(all_targets[:, idx], all_outputs[:, idx])
            auprc = average_precision_score(all_targets[:, idx], all_outputs[:, idx])
        except ValueError:
            # Handle the case when only one class is present in y_true
            auroc = float('nan')
            auprc = float('nan')
        fold_aurocs[outcomes[idx]].append(auroc)
        fold_auprcs[outcomes[idx]].append(auprc)
        print(f'Outcome: {outcomes[idx]}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

# Calculate and print overall performance
print("\nOverall Performance Across Folds:")
for label in outcomes:
    aurocs = fold_aurocs[label]
    auprcs = fold_auprcs[label]
    aurocs_clean = [auc for auc in aurocs if not np.isnan(auc)]
    auprcs_clean = [auc for auc in auprcs if not np.isnan(auc)]
    print(f'\nOutcome: {label}')
    if aurocs_clean:
        print(f'Average AUROC: {np.mean(aurocs_clean):.4f}, Std AUROC: {np.std(aurocs_clean):.4f}')
    else:
        print('AUROC could not be calculated due to single-class issues.')
    if auprcs_clean:
        print(f'Average AUPRC: {np.mean(auprcs_clean):.4f}, Std AUPRC: {np.std(auprcs_clean):.4f}')
    else:
        print('AUPRC could not be calculated due to single-class issues.')