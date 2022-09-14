import base64
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from st_clickable_images import clickable_images
from torchvision import models, transforms

CURRENT_DIR = Path(__file__).cwd()
num_classes = 4

"""Analyze breast density based on mammogram image"""

model_ft = models.swin_t()
num_ftrs = model_ft.head.in_features
model_ft.head = nn.Linear(num_ftrs, num_classes)
input_size = 224

model_ft.load_state_dict(torch.load("NN_model.pt", map_location=torch.device('cpu')))
model_ft.eval()

device = "cpu"
transforms = transforms.Compose([
    #       transforms.RandomResizedCrop(input_size),
    transforms.Resize(input_size),  # tło może być ważne
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = st.file_uploader("Pick a mammogram to analyze", type=['png', 'jpg', 'jpeg'])

images = []
for file in [os.path.join(CURRENT_DIR, "examples/density0.png"),
             os.path.join(CURRENT_DIR, "examples/density1.png"),
             os.path.join(CURRENT_DIR, "examples/density2.png"),
             os.path.join(CURRENT_DIR, "examples/density3.png")]:
    with open(file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
        images.append(f"data:image/jpeg;base64,{encoded}")

st.write(f"Or choose one of the examples:")
clicked = -1
if img is None:
    clicked = clickable_images(
        images,
        titles=["Density 1", "Density 2", "Density 3", "Density 4"],
        div_style={"display": "grid",
                   "grid-template-columns": "200px 200px",
                   "grid-row": "auto auto",
                   "grid-column-gap": "20px",
                   "grid-row-gap": "20px"},
        img_style={"margin": "5px", "height": "200px"},
    )

pillow_image = None
if img is not None:
    pillow_image = Image.open(img)
    img.seek(0)
elif clicked > -1:
    pillow_image = Image.open(os.path.join(CURRENT_DIR, f"examples/density{clicked}.png"))

if pillow_image is not None:
    st.image(pillow_image, caption="Chosen image")
    pillow_image = Image.merge('RGB', (pillow_image, pillow_image, pillow_image))
    preprocessed_img = transforms(pillow_image)
    preprocessed_img = preprocessed_img.unsqueeze(0)
    preprocessed_img = preprocessed_img.to(device)
    output = model_ft(preprocessed_img)
    _, preds = torch.max(output, 1)
    st.write(f"Density class prediction: {preds.item()+1}")
    if clicked > -1:
        st.write(f"Density marked by radiologist for this image: {clicked+1}")

    st.write("\n")
    st.write("Scores reached by model for particular classes:")
    st.write(f"Class 1 score: {output[0][0].item()}")
    st.write(f"Class 2 score: {output[0][1].item()}")
    st.write(f"Class 3 score: {output[0][2].item()}")
    st.write(f"Class 4 score: {output[0][3].item()}")
