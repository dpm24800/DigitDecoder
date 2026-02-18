import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# ---------------- CNN Model Definition ---------------- #
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

# ---------------- Load Trained Model ---------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.to(device)
model.eval()

# ---------------- Streamlit UI ---------------- #
st.title("MNIST Digit Recognition - Draw Your Digit")
st.write("Draw a digit (0-9) in the canvas below and get prediction.")

# Canvas settings
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=15,
    stroke_color="#FFFFFF",  # White brush
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert drawn image to grayscale PIL image
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = img.resize((28,28))  # MNIST size
        img = np.array(img)

        # Invert colors (MNIST is white digit on black)
        img = 255 - img
        img = Image.fromarray(img)

        # Show preprocessed image
        st.subheader("Preprocessed Input (28x28)")
        st.image(img, width=140)

        # Preprocess for model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()

        st.success(f"Predicted Digit: {pred}")
