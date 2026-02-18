import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


# Load Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


# Streamlit UI
st.title("DigitDecoder")
st.write("Draw a digit from 0 to 9.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# Prediction
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # canvas image is RGBA -> convert safely to grayscale
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img = img.convert("L")          # grayscale
        img = img.resize((28, 28))      # MNIST size

        st.subheader("Preprocessed image (28 x 28)")
        st.image(img, width=140)

        # Same normalization as MNIST training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        image_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()

        st.success(f"Predicted digit: {prediction}")