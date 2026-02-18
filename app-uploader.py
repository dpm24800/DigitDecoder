import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load Trained Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=device))  # Your saved model
model.to(device)
model.eval()

# Streamlit UI
st.title("DigitDecoder")
st.write("Upload an image of a handwritten digit with a black background and white foreground, as the model was trained on this format.")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=280)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()

    st.success(f"Predicted Digit: {pred}")

