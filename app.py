import os
import numpy as np
import streamlit as st
import torch
import gdown
from PIL import Image
from torchvision import transforms

from model import load_model

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Clasificador de Monedas",
    page_icon="💰",
    layout="centered"
)

# =========================================================
# ESTILOS
# =========================================================
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 14px;
        background-color: #f3f7ff;
        border: 1px solid #dbe7ff;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# PARÁMETROS
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_coinnet_local.pth"

# Cambia esta URL si cambias el archivo de Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Chh-2AsTSVUU7eA0mrkcuif80TLrWarR"

# =========================================================
# DESCARGA DEL MODELO
# =========================================================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Descargando modelo..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =========================================================
# CARGA DEL MODELO REAL
# =========================================================
@st.cache_resource
def get_model():
    model, idx_to_class = load_model(MODEL_PATH, DEVICE)
    model.eval()
    return model, idx_to_class

model, idx_to_class = get_model()

# =========================================================
# TRANSFORMACIONES
# =========================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Opciones")
modo = st.sidebar.radio(
    "Selecciona una opción",
    ["Subir imagen", "Usar cámara"]
)

st.sidebar.markdown("---")
st.sidebar.info("App de clasificación de monedas con modelo entrenado en PyTorch.")

# =========================================================
# CABECERA
# =========================================================
st.markdown('<div class="title">💰 Clasificador de Monedas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube una imagen o haz una foto para predecir la clase de la moneda</div>', unsafe_allow_html=True)

# =========================================================
# ENTRADA DE IMAGEN
# =========================================================
image = None

if modo == "Subir imagen":
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_file = st.camera_input("Haz una foto")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

# =========================================================
# PREDICCIÓN
# =========================================================
if image is not None:
    st.image(image, caption="Imagen cargada", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = idx_to_class[pred_idx]
        confidence = float(probs[pred_idx]) * 100

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("Resultado")
    st.success(f"Predicción: {pred_class}")
    st.info(f"Confianza: {confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Top 3 predicciones")
    top3_idx = np.argsort(probs)[-3:][::-1]

    for i in top3_idx:
        clase = idx_to_class[i]
        prob = probs[i] * 100
        st.write(f"**{clase}** — {prob:.2f}%")
        st.progress(float(probs[i]))

else:
    st.markdown("### 👇 Sube una imagen para empezar")

st.markdown("---")
st.caption("Hecho con Streamlit + PyTorch")
