import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import numpy as np

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

# Step 2: Create a biased dataset
def create_biased_dataset(dataset, few_samples_per_class=5, normal_samples_per_class=1000):
    indices = []
    class_counts = {i: 0 for i in range(10)}
    
    for i, (_, label) in enumerate(dataset):
        # For classes 0-4, we limit the samples to a smaller number
        if label in [0] and class_counts[label] < few_samples_per_class:
            indices.append(i)
            class_counts[label] += 1
        # For classes 5-9, we keep a normal amount of samples
        elif label in [1, 2, 3, 4, 5, 6, 7, 8, 9] and class_counts[label] < normal_samples_per_class:
            indices.append(i)
            class_counts[label] += 1
    
    # Display the class distribution for verification
    print("Training class distribution:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")
    
    return Subset(dataset, indices)

# Create biased train dataset
biased_train_dataset = create_biased_dataset(train_dataset)
train_loader = DataLoader(biased_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 3: Modify ResNet50 for MNIST (1 channel input and 10 output classes)
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

# Step 4: Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to evaluate the model on the test set with class-wise accuracy
def evaluate_model(model, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = correct / total
    class_accuracies = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)}
    
    print("\nClass-wise accuracy:")
    for class_label, acc in class_accuracies.items():
        print(f"Class {class_label}: {acc*100:.2f}%")
        
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    return accuracy, class_accuracies

# Step 5: Train the model with evaluation and best model saving
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
        print(f'\nEpoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        
        # Evaluate on test set with class-wise accuracy
        test_accuracy, class_accuracies = evaluate_model(model, test_loader)
        
        # Save the best model based on overall test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(model_path, 'only5SamplesZeroClass_resnet50_mnist.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Overall Test Accuracy: {best_accuracy*100:.2f}%\n")

# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=5)
