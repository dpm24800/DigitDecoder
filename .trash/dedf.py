import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import io

# -------------------------------------------------------
# Page setup
# -------------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    layout="centered"
)

# -------------------------------------------------------
# Force dark UI + light icons / buttons
# -------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    .stButton>button,
    .stDownloadButton>button {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    svg {
        fill: white !important;
    }

    header, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- CNN Model ----------------
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# ---------------- Load trained model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- UI ----------------
st.title("MNIST Digit Recognition")
st.write("Background is black and drawing color is white. Draw a digit (0–9).")

# ---------------- Canvas ----------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",      # white drawing
    background_color="#000000",  # black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------- Prediction ----------------
if canvas_result.image_data is not None:

    # Convert RGBA canvas to grayscale
    img = Image.fromarray(
        canvas_result.image_data.astype("uint8")
    ).convert("L")

    # Resize to MNIST size
    img_28 = img.resize((28, 28))

    img_np = np.array(img_28)

    # Invert to match MNIST style (white digit on black)
    img_np = 255 - img_np

    processed_img = Image.fromarray(img_np)

    st.subheader("Preprocessed image (28×28)")
    st.image(processed_img, width=140)

    # -------- Download button (FIXED & WORKING) --------
    buffer = io.BytesIO()
    processed_img.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download preprocessed image",
        data=buffer,
        file_name="mnist_input.png",
        mime="image/png"
    )

    # -------- Model preprocessing --------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_tensor = transform(processed_img).unsqueeze(0).to(device)

    # -------- Prediction --------
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    st.subheader(f"Predicted digit : {pred}")

    st.write("Prediction confidence:")
    for i, p in enumerate(probs):
        st.write(f"{i} : {p * 100:.2f} %")
