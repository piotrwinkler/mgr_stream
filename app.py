import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

num_classes = 4

"""Analyze breast density based on mammogram image with Swin Transformer"""

model_ft = models.swin_t()
num_ftrs = model_ft.head.in_features
model_ft.head = nn.Linear(num_ftrs, num_classes)
input_size = 224

model_ft.load_state_dict(torch.load("NN_model.pt"))
model_ft.eval()

device = "cpu"
transforms = transforms.Compose([
    #       transforms.RandomResizedCrop(input_size),
    transforms.Resize(input_size),  # tło może być ważne
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = st.file_uploader("Pick mammogram to analyze", type=['png', 'jpg', 'jpeg'])

if img is not None:
    img = Image.open(img)
    st.image(img, caption="Uploaded image")
    img = Image.merge('RGB', (img, img, img))
    preprocessed_img = transforms(img)
    preprocessed_img = preprocessed_img.unsqueeze(0)
    preprocessed_img = preprocessed_img.to(device)
    output = model_ft(preprocessed_img)
    _, preds = torch.max(output, 1)
    st.write(f"Density class prediction: {preds.item()+1}")
    st.write(f"Class 1 score: {output[0][0].item()}")
    st.write(f"Class 2 score: {output[0][1].item()}")
    st.write(f"Class 3 score: {output[0][2].item()}")
    st.write(f"Class 4 score: {output[0][3].item()}")
