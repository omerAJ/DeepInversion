import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Set the path for saving datasets and model
dataset_path = "D:/datasets"
model_path = "./models"
os.makedirs(model_path, exist_ok=True)  # Ensure the models directory exists

# Step 1: Set up data transformations and load the MNIST dataset (no resizing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5 for grayscale
])

train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Modify ResNet50 for MNIST (1 channel input and 10 output classes)
class ResNet50Mnist(nn.Module):
    def __init__(self):
        super(ResNet50Mnist, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # Modify the input layer to accept 1 channel (grayscale)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        # Modify the final layer to output 10 classes (for MNIST)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        
    def forward(self, x):
        return self.model(x)

# Instantiate the model
model = ResNet50Mnist().to('cuda' if torch.cuda.is_available() else 'cpu')

# Step 3: Set up loss function and optimizer
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
    return accuracy

# Step 4: Train the model with evaluation and best model saving
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    best_accuracy = 0.0  # Track the best test accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"Loss": loss.item()})
        
        # Calculate average training loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.4f}')
        
        # Save the best model based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(model_path, 'best_resnet50_mnist.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Test Accuracy: {best_accuracy*100:.4f}%")

# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=5)

# # At the end, best model will be saved in the 'models' directory
# print(f"Training completed. Best model saved as '{best_model_path}'")
