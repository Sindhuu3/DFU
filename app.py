# ============================================================
# DFU RISK ASSESSMENT - STREAMLIT DEPLOYMENT APP
# Low | Moderate | High Risk + Grad-CAM
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DFU Risk Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🦶 Diabetic Foot Ulcer Risk Assessment")
st.caption("AI-powered risk prediction with visual explanation (Grad-CAM)")

# ---------------- LOAD MODEL & CONFIG ----------------
@st.cache_resource
def load_model_and_config():
    model = tf.keras.models.load_model("dfu_risk_model.keras")
    with open("deployment_config.json", "r") as f:
        config = json.load(f)
    return model, config

model, config = load_model_and_config()

IMG_SIZE = tuple(config["image_size"])
LAST_CONV_LAYER = config["last_conv_layer"]

# ---------------- RISK MAPPING ----------------
def prob_to_risk(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Moderate Risk"
    else:
        return "High Risk"

# ---------------- GRAD-CAM ----------------
def make_gradcam_heatmap(img_array, model):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# ---------------- STREAMLIT UI ----------------
uploaded_file = st.file_uploader(
    "Upload a foot image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Risk"):
        with st.spinner("Analyzing image..."):
            # Preprocess
            img_resized = cv2.resize(img_np, IMG_SIZE)
            img_norm = img_resized / 255.0
            img_batch = np.expand_dims(img_norm, axis=0)

            # Prediction
            prob = float(model.predict(img_batch, verbose=0)[0][0])
            risk = prob_to_risk(prob)
            confidence = abs(prob - 0.5) * 200

            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_batch, model)
            gradcam_img = overlay_gradcam(img_resized, heatmap)

        # ---------------- RESULTS ----------------
        st.subheader("📊 Risk Assessment Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Type", risk)
        col2.metric("Risk Score", f"{prob:.3f}")
        col3.metric("Confidence", f"{confidence:.1f}%")

        st.subheader("🧠 Model Explanation (Grad-CAM)")
        st.image(gradcam_img, caption="Highlighted regions influencing prediction")

        # ---------------- MEDICAL NOTE ----------------
        st.subheader("📝 Interpretation")
        if risk == "Low Risk":
            st.success("Low risk detected. Continue routine foot care and monitoring.")
        elif risk == "Moderate Risk":
            st.warning("Moderate risk detected. Medical consultation recommended.")
        else:
            st.error("High risk detected. Immediate medical attention advised.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚠ This tool is for clinical decision support only. Not a replacement for professional diagnosis.")
