import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Set the path for saving datasets and model
dataset_path = "D:/datasets"
model_path = "./models"
os.makedirs(model_path, exist_ok=True)

# Data transformations and loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Fixed output size
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)  # Output fixed to 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleCNN().to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate the model on the test set
def evaluate_model(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

# Training function
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # print("images.size(): ", images.size(), "labels.size(): ", labels.size())
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"Loss": loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'\nEpoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader)
        
        # Save the best model based on overall test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(model_path, 'Verifier_simple_cnn_mnist.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Overall Test Accuracy: {best_accuracy*100:.2f}%\n")

# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=5)
