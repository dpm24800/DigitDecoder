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
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

# Streamlit App
st.title("DigitDecoder")
st.write("Draw a digit (0-9) or upload an image of a handwritten digit with a black background and white foreground, as the model was trained on this format.")

tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

# Draw Digit Tab
with tab1:
    st.write("Draw a digit (0–9) below:")
    canvas_result = st_canvas(
        fill_color="#000000",      # Black background
        stroke_width=15,
        stroke_color="#FFFFFF",    # White digit
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict Drawing"):
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("L")
            img = img.resize((28,28))
            st.subheader("Preprocessed image (28x28)")
            st.image(img, width=140)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            image_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                prediction = output.argmax(dim=1).item()

            st.success(f"Predicted digit: {prediction}")

#  Upload Image Tab 
with tab2:
    uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=280)

        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()

        st.success(f"Predicted digit: {prediction}")
