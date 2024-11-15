import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import imagenetLabels  # Import your imagenetLabels.py

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

# Function to perform forward pass and get softmax output
def get_softmax_output(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Run the forward pass
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)

    # Apply softmax to get probabilities
    softmax = torch.nn.functional.softmax(outputs[0], dim=0)
    return softmax

# Example usage
# Provide the path to your image
image_path = r"D:\datasets\imagenetImages\basketball\018.jpg"
softmax_output = get_softmax_output(image_path)

# Display top 5 predictions with labels
_, indices = torch.topk(softmax_output, 5)
for idx in indices:
    class_name = imagenetLabels.imagenet_labels[idx.item()]  # Get label from imagenetLabels
    probability = softmax_output[idx].item()
    print(f"Class: {class_name}, Probability: {probability}")
