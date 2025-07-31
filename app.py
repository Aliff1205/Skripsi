# isi kode streamlit kamu di sini
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# --- Unduh model dari Google Drive jika belum ada ---
MODEL_PATH = "best_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1nx03ynTjClMAO6_TwL4zLQXR5BGy11IO"  # ← Ganti kalau file ID berubah
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# --- Load model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Daftar label kelas (urutan harus sama seperti saat training) ---
class_names = ['immature', 'mature', 'normal']

# --- UI Streamlit ---
st.set_page_config(page_title="Klasifikasi Katarak", layout="centered")
st.title("👁️ Aplikasi Klasifikasi Katarak dengan CNN")
st.markdown("Upload gambar mata untuk diprediksi sebagai **immature**, **mature**, atau **normal**.")

# --- Upload gambar ---
uploaded_file = st.file_uploader("📁 Pilih gambar mata", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # --- Preprocessing ---
    img = image.resize((224, 224))  # Pastikan sesuai ukuran input model
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

    # --- Prediksi ---
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # --- Tampilkan hasil ---
    st.markdown(f"### ✅ Prediksi: **{pred_class.upper()}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")

