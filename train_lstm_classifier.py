import numpy as np
import mlflow
from mlflow.models import infer_signature
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix

from globals import SPLIT_SPECTROGRAM_FOLDER_PATH

# %%
mlflow.set_experiment("Ballroom Dance Experiment")
# IMPORTANT: Enable system metrics monitoring
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

# Training parameters
params = {
    "epochs": 10,
    "learning_rate":1e-3,
    "batch_size": 32,
    "optimizer": "Adam",
    "model_type": "LSTM",
    "hidden_units": [128, 128],
}
NUM_WORKERS = 12
train_size = 0.8

#%% Define transformations
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

full_dataset = ImageFolder(
    root=SPLIT_SPECTROGRAM_FOLDER_PATH,
    transform=data_transforms
)

train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, 1 - train_size]
)
# Create the DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=params['batch_size'], 
    shuffle=True, 
    num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=params['batch_size'], 
    shuffle=False, 
    num_workers=NUM_WORKERS
)

#%% Define Neural Network
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128x128 -> 64x64
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # 2. LSTM Layer
        # After pooling, height (frequency) is 32 and we have 32 channels.
        # We treat the width (32) as our "Time Steps".
        self.lstm_input_size = 32 * 32  # channels * frequency_height
        self.hidden_size = 128
        self.num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 # Helpful for small datasets like Ballroom
        )
        
        # 3. Fully Connected Head
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 128, 128]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Current shape: [batch, 32, 32, 32] -> (batch, channels, freq, time)
        # To use LSTM, we need: [batch, time, features]
        x = x.permute(0, 3, 1, 2).contiguous() # [batch, 32, 32, 32] -> (batch, time, channels, freq)
        batch_size, time_steps, channels, freq = x.size()
        x = x.view(batch_size, time_steps, -1) # [batch, 32, 1024]
        
        # LSTM forward pass
        # out: [batch, time, hidden_size]
        out, _ = self.lstm(x)
        
        # Take the hidden state of the LAST time step
        out = out[:, -1, :]
        
        # Classification
        out = self.fc(out)
        return out
    
# Instantiate the new model
model = CRNN(num_classes=len(full_dataset.classes))

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# CrossEntropyLoss is ideal for classification
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam or SGD are common choices
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

with mlflow.start_run() as run:

    mlflow.log_params(params)
    print(f"Starting training on {device}...")

    for epoch in range(params['epochs']):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader, 0):
            # Get the inputs and labels, and move them to the correct device
            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Log batch metrics (every 100 batches)
            if batch_idx % 100 == 0:
                batch_loss = train_loss / (batch_idx + 1)
                batch_acc = 100.0 * correct / total
                mlflow.log_metrics(
                    {"batch_loss": batch_loss, "batch_accuracy": batch_acc},
                    step=epoch * len(train_loader) + batch_idx,
                )
        # Log epoch metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print('Finished Training.')

        print('Beginning Model Validation')
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        # Calculate and log epoch validation metrics
        val_loss = val_loss / len(test_loader)
        val_acc = 100.0 * val_correct / val_total

        # Log epoch metrics
        mlflow.log_metrics(
            {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )
        # Log checkpoint at the end of each epoch
        mlflow.pytorch.log_model(model, name=f"checkpoint_{epoch}")

        print(
            f"Epoch {epoch+1}/{params['epochs']}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Log the final trained model
    model_info = mlflow.pytorch.log_model(model, name="final_model")