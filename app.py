import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Nama file model di repo
MODEL_PATH = "best_model_tf15.h5"

# Load model sekali saja
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Daftar nama kelas sesuai urutan training
class_names = ['immature', 'mature', 'normal']

# Tampilan Streamlit
st.set_page_config(page_title="Klasifikasi Katarak", layout="centered")
st.title("👁️ Aplikasi Klasifikasi Katarak dengan CNN")
st.markdown("Upload gambar mata untuk diprediksi sebagai **immature**, **mature**, atau **normal**.")

# Upload gambar
uploaded_file = st.file_uploader("📁 Pilih gambar mata", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output hasil
    st.markdown(f"### ✅ Prediksi: **{pred_class.upper()}**")
    st.markdown(f"Confidence: `{confidence:.2f}%`")
