import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import numpy as np
import argparse

# Define your LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Define your dataset class
class AminoAcidDataset(Dataset):
    def __init__(self, df, amino_acids):
        self.sequences = df['Sequence'].tolist()
        self.classes = df['Label'].tolist()
        self.amino_acid_mapping = {acid: i for i, acid in enumerate(amino_acids)}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.classes[idx]
        one_hot_encoding = self.sequence_to_one_hot(sequence)
        label_tensor = torch.tensor([1, 0] if label == 1 else [0, 1], dtype=torch.float32)
        return one_hot_encoding, label_tensor
    
    def sequence_to_one_hot(self, sequence):
        sequence = sequence.upper()
        one_hot_encoding = torch.zeros(len(sequence), len(self.amino_acid_mapping))
        for i, amino_acid in enumerate(sequence):
            if amino_acid in self.amino_acid_mapping:
                one_hot_encoding[i, self.amino_acid_mapping[amino_acid]] = 1
        return one_hot_encoding

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)

# Save the model's parameters
def save_model(model):
    torch.save(model.state_dict(), 'lstm_model.pth')

# Load the model's parameters
def load_model(input_size, hidden_size, num_classes):
    loaded_model = LSTMModel(input_size, hidden_size, num_classes)
    loaded_model.load_state_dict(torch.load('lstm_model.pth'))
    return loaded_model

# Preprocess input sequence and make prediction
def predict_sequence(sequence, model, amino_acids, device):
    # Preprocess the sequence
    amino_acid_mapping = {acid: i for i, acid in enumerate(amino_acids)}
    one_hot_encoding = torch.zeros(len(sequence), len(amino_acid_mapping))
    for i, amino_acid in enumerate(sequence.upper()):
        if amino_acid in amino_acid_mapping:
            one_hot_encoding[i, amino_acid_mapping[amino_acid]] = 1
    one_hot_encoding = one_hot_encoding.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(one_hot_encoding)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    return prediction

if __name__ == "__main__":
    # Define constants
    input_size = 20
    hidden_size = 100
    num_classes = 2
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(input_size, hidden_size, num_classes)
    model.to(device)

    # Ask the user for the sequence
    sequence = input("Enter the amino acid sequence: ")

    # Predict the provided sequence
    prediction = predict_sequence(sequence, model, amino_acids, device)

    print(f"Prediction for sequence '{sequence}': {prediction}")
