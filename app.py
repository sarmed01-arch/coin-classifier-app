import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import load_model

st.title("Clasificador de Monedas")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_coinnet_local.pth"

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, DEVICE)

model, idx_to_class = get_model()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return idx_to_class[pred], probs[0][pred].item()

option = st.radio("Elige opción", ["Subir imagen", "Hacer foto"])

image = None

if option == "Subir imagen":
    file = st.file_uploader("Sube una imagen")
    if file:
        image = Image.open(file)

else:
    file = st.camera_input("Haz una foto")
    if file:
        image = Image.open(file)

if image:
    st.image(image)

    pred, conf = predict(image)

    st.success(f"Predicción: {pred}")
    st.write(f"Confianza: {conf:.2%}")
