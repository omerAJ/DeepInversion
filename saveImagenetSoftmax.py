import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import imagenetLabels  # Import your imagenetLabels.py
import numpy as np
import os

# Function to load and preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),        # Resize the image to 256x256
        transforms.CenterCrop(224),    # Center crop to 224x224
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(          # Normalize with ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load the pre-trained ResNet-50 model
model = resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode

# Function to perform forward pass and get softmax output for one image
def get_softmax_output(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Run the forward pass
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)

    # Apply softmax to get probabilities
    softmax = torch.nn.functional.softmax(outputs[0], dim=0)
    return softmax

# Function to process all images in a folder and save softmax outputs
def process_images_in_folder(folder_path, output_file):
    # List all image files in the folder
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Initialize an array to store softmax outputs
    all_softmax_outputs = []

    # Process each image and store its softmax output
    for image_file in image_files:
        softmax_output = get_softmax_output(image_file)
        all_softmax_outputs.append(softmax_output.numpy())  # Convert tensor to numpy array

    # Stack outputs into a single numpy array of shape [num_images, num_classes]
    all_softmax_outputs = np.stack(all_softmax_outputs, axis=0)

    # Save the array to a .npy file
    np.save(output_file, all_softmax_outputs)
    print(f"Saved softmax outputs to {output_file}")

# Example usage
# Specify the folder path and output file path
folder_path = r"D:\datasets\imagenetImages\basketball"
output_file = r"D:\datasets\imagenetImages\basketball_softmax_outputs.npy"
process_images_in_folder(folder_path, output_file)
