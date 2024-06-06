import sys
sys.path.append('../../')

from main import load_and_prepare_sessions
from processing.session_sampling import MiceAnalysis
from analysis.timepoint_analysis import sample_signals_and_metrics, sample_low_and_high_signals
from config import all_brain_regions, all_event_types, all_metrics
from itertools import product
import numpy as np
from utils import mouse_br_events_count

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

window_size = 5
window = np.ones(window_size) / window_size

sessions = load_and_prepare_sessions("../../../Baseline", load_from_pickle=True, remove_bad_signal_sessions=True)
mouse_analyser = MiceAnalysis(sessions)

# generate all aggregated signals
all_event_signals = []
labels = []

for mouse in mouse_analyser.mice_dict.values():
    mouse_sessions = mouse.sessions
    for brain_region, event in product(all_brain_regions, ['hit', 'mistake', 'miss', 'cor_reject']):
        mouse_signals = [] 
        for session in mouse_sessions:
            if session.signal_info.get((brain_region, event)) is None:
                continue
            signals = sample_signals_and_metrics([session], event, brain_region)[0]
            mouse_signals.append(signals[:, 150:250])
        if len(mouse_signals) == 0:
            continue
        mouse_signals = np.vstack(mouse_signals)
        sample_idxs = np.random.choice(len(mouse_signals), 100, replace=True)
        all_event_signals.append(mouse_signals)
        labels.extend([(mouse.mouse_id, brain_region, event)] * len(mouse_signals))

all_event_signals = np.vstack(all_event_signals)

mouse_labels, br_labels, event_labels = zip(*labels)
mouse_labels = np.array(mouse_labels)
br_labels = np.array(br_labels)
event_labels = np.array(event_labels)

# Encode the br_labels and event_labels to numerical values
br_label_encoder = LabelEncoder()
br_labels_encoded = br_label_encoder.fit_transform(br_labels)

event_label_encoder = LabelEncoder()
event_labels_encoded = event_label_encoder.fit_transform(event_labels)

unique_mouse_labels = np.unique(mouse_labels)
train_mice, test_mice = train_test_split(unique_mouse_labels, test_size=0.4, random_state=42)

train_mask = np.isin(mouse_labels, train_mice)
test_mask = np.isin(mouse_labels, test_mice)
all_event_signals_train = all_event_signals[train_mask]
all_event_signals_test = all_event_signals[test_mask]
br_labels_train = br_labels_encoded[train_mask]
br_labels_test = br_labels_encoded[test_mask]

event_labels_train = event_labels_encoded[train_mask]
event_labels_test = event_labels_encoded[test_mask]

import torch
from torch.utils.data import DataLoader, TensorDataset

# Assuming all_event_signals_train, br_labels_train, event_labels_train, 
# all_event_signals_test, br_labels_test, event_labels_test are already defined

# Convert the data to PyTorch tensors
all_event_signals_train = torch.tensor(all_event_signals_train, dtype=torch.float32)
br_labels_train = torch.tensor(br_labels_train, dtype=torch.long)
event_labels_train = torch.tensor(event_labels_train, dtype=torch.long)

all_event_signals_test = torch.tensor(all_event_signals_test, dtype=torch.float32)
br_labels_test = torch.tensor(br_labels_test, dtype=torch.long)
event_labels_test = torch.tensor(event_labels_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(all_event_signals_train, br_labels_train, event_labels_train)
test_dataset = TensorDataset(all_event_signals_test, br_labels_test, event_labels_test)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, br_output_size, event_output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_br = nn.Linear(hidden_size, br_output_size)
        self.fc_event = nn.Linear(hidden_size, event_output_size)
    
    def forward(self, x):
        # Set initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step for both br and event labels
        br_out = self.fc_br(out[:, -1, :])
        event_out = self.fc_event(out[:, -1, :])
        return br_out, event_out

# Hyperparameters
input_size = 1  # One feature per time step
hidden_size = 128
br_output_size = len(torch.unique(br_labels_train))  # Number of unique br_labels
event_output_size = len(torch.unique(event_labels_train))  # Number of unique event_labels
num_layers = 1

# Initialize the model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, br_output_size, event_output_size, num_layers)
criterion_br = nn.CrossEntropyLoss()
criterion_event = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move model to the configured device

model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    for signals, br_labels, event_labels in train_loader:
        # Move tensors to the configured device
        signals = signals.to(device).unsqueeze(-1)
        br_labels = br_labels.to(device)
        event_labels = event_labels.to(device)
        
        # Forward pass
        br_outputs, event_outputs = model(signals)
        br_loss = criterion_br(br_outputs, br_labels)
        event_loss = criterion_event(event_outputs, event_labels)
        loss = br_loss + event_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    br_correct = 0
    event_correct = 0
    total = 0
    for signals, br_labels, event_labels in test_loader:
        signals = signals.to(device).unsqueeze(-1)
        br_labels = br_labels.to(device)
        event_labels = event_labels.to(device)
        
        br_outputs, event_outputs = model(signals)
        _, br_predicted = torch.max(br_outputs.data, 1)
        _, event_predicted = torch.max(event_outputs.data, 1)
        
        total += br_labels.size(0)
        br_correct += (br_predicted == br_labels).sum().item()
        event_correct += (event_predicted == event_labels).sum().item()
    
    print(f'Test Accuracy for br_labels: {100 * br_correct / total:.2f}%')
    print(f'Test Accuracy for event_labels: {100 * event_correct / total:.2f}%')