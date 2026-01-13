import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define transformations
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

NUM_WORKERS = 16

# 1. Create the Dataset
full_dataset = ImageFolder(
    root='data/BallroomData/split_spectrograms',
    transform=data_transforms
)

# 2. Split into training and testing/validation sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, 
    [train_size, test_size]
)

# 3. Create the DataLoaders
batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=NUM_WORKERS
)

import torch.nn as nn
import torch.nn.functional as F
import torch

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

import torch.optim as optim

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss Function: CrossEntropyLoss is ideal for classification
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam or SGD are common choices
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15 # Adjust as needed

print(f"Starting training on {device}...")

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels, and move them to the correct device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass (calculate gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.3f}')

print('Finished Training.')


import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, data_loader, device, full_dataset):
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    
    # Lists to store all true labels and predictions for detailed analysis
    all_preds = []
    all_labels = []

    # Disable gradient calculations
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get the predicted class (index of the highest logit)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {total} test images: {accuracy:.2f}%')
    
    return np.array(all_labels), np.array(all_preds), full_dataset.classes

# --- Run the Evaluation ---
# Assuming 'model', 'test_loader', 'device', and 'full_dataset' are defined from Section 1 & 3
true_labels, predictions, class_names = evaluate_model(model, test_loader, device, full_dataset)

# The classification report provides Precision, Recall, and F1-score per class
print("\n### Detailed Classification Report ###")
print(classification_report(true_labels, predictions, target_names=class_names))

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Generate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

save = input("Save the model? Y/N")
if save in ['y', 'Y']:
    torch.save(model.state_dict(), "./models/lstm_model_weights.pth")
    print("Model save successfully!")