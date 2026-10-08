from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Models" / "leafguard_mobilenetv2.keras"
CLASS_PATH = PROJECT_DIR / "Models" / "class_names.json"


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, encoding="utf-8") as file:
        class_names = json.load(file)
    return model, class_names


st.set_page_config(page_title="LeafGuard AI", page_icon="🌿")
st.title("🌿 LeafGuard AI")
st.write("Upload a clear photo of one plant leaf to get a disease-class prediction.")

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error("Model files are missing. Check that both files are in the Models folder.")
    st.stop()

model, class_names = load_model_and_classes()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    resized_image = image.resize((224, 224))
    image_array = np.asarray(resized_image, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    best_index = int(top_indices[0])
    predicted_class = class_names[best_index].replace("___", " — ").replace("_", " ")

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {probabilities[best_index]:.1%}")

    st.write("Top predictions:")
    for index in top_indices:
        label = class_names[int(index)].replace("___", " — ").replace("_", " ")
        st.write(f"- {label}: {probabilities[index]:.1%}")

    st.info(
        "This is an educational prediction based on the classes in the training dataset. "
        "For crop treatment decisions, verify the result with a qualified agricultural expert."
    )
