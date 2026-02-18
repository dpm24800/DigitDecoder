import os
from PIL import Image, ImageOps

# --------- CONFIG ---------
input_folder = "input_images"
output_folder = "output_images"
# --------------------------

os.makedirs(output_folder, exist_ok=True)

supported_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

for filename in os.listdir(input_folder):

    if filename.lower().endswith(supported_ext):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open image
        img = Image.open(input_path)

        # Convert to RGB (important for invert to work correctly)
        img = img.convert("RGB")

        # Invert colors
        inverted_img = ImageOps.invert(img)

        # Save result
        inverted_img.save(output_path)

        print(f"Inverted: {filename}")

print("Done. All images processed.")
