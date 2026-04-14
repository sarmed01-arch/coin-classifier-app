import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import gdown

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Clasificador de Monedas 💰",
    page_icon="💰",
    layout="centered"
)

# =========================================================
# DESCARGAR MODELO
# =========================================================
MODEL_PATH = "best_coinnet_local.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=1Chh-2AsTSVUU7eA0mrkcuif80TLrWarR"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================================================
# CLASES
# =========================================================
class_names = [
    "1 cent", "2 cents", "5 cents", "10 cents",
    "20 cents", "50 cents", "1 euro", "2 euros",
    "moneda rara", "otra"
]

# =========================================================
# CARGAR MODELO
# =========================================================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =========================================================
# TRANSFORMACIONES
# =========================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Opciones")
st.sidebar.write("Sube una imagen o usa la cámara")

modo = st.sidebar.radio(
    "Selecciona modo:",
    ["Subir imagen", "Usar cámara"]
)

# =========================================================
# TÍTULO
# =========================================================
st.title("💰 Clasificador de Monedas")
st.markdown("Sube una imagen de una moneda y te diré cuál es 👇")

# =========================================================
# INPUT
# =========================================================
image = None

if modo == "Subir imagen":
    uploaded_file = st.file_uploader("📂 Sube una imagen", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif modo == "Usar cámara":
    camera_image = st.camera_input("📸 Haz una foto")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# =========================================================
# PREDICCIÓN
# =========================================================
if image is not None:
    st.image(image, caption="Imagen subida", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)

    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx] * 100

    st.markdown("---")
    st.subheader("🔎 Resultado")

    st.success(f"Predicción: **{pred_class}**")
    st.info(f"Confianza: **{confidence:.2f}%**")

    # =====================================================
    # TOP 3
    # =====================================================
    st.subheader("📊 Top 3 predicciones")

    top3_idx = probs.argsort()[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} → {probs[i]*100:.2f}%")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("App creada con ❤️ usando Streamlit + ResNet18")
