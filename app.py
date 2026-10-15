"""
GroundedCXR - Explainable Chest X-Ray Classification
Streamlit application entry point.

Loads your locally trained ResNet18 checkpoint and runs real inference
plus a Grad-CAM explanation on an uploaded chest X-ray image.
"""

import os
import sys

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

# Make "src" importable when running via `streamlit run app.py`
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import load_trained_model
from src.inference import preprocess_image, predict, build_prediction_context
from src.gradcam import generate_gradcam
from src.chatbot import get_chatbot_response, is_api_key_configured

MODEL_PATH = os.path.join("models", "best_pneumonia_resnet18.pth")

st.set_page_config(
    page_title="GroundedCXR",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main > div {padding-top: 2rem;}
        .app-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0;}
        .app-subtitle {font-size: 1.05rem; color: #6b7280; margin-top: 0.2rem;}
        .disclaimer-box {
            background-color: #fff8e1;
            border: 1px solid #f0c419;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
            color: #7a5c00;
            margin-top: 1.2rem;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.4rem;
        }
        .pred-label {font-size: 1.8rem; font-weight: 700;}
        .pred-normal {color: #1a7f37;}
        .pred-pneumonia {color: #cf222e;}
        .confidence-caption {color: #6b7280; font-size: 0.9rem;}
        .caption-text {color: #6b7280; font-size: 0.9rem; text-align: center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached model loading (loads once, not on every upload)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ResNet18 model...")
def get_model_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(MODEL_PATH, device)
    return model, device


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">GroundedCXR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Explainable Chest X-Ray Classification</div>',
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Load model (friendly error handling)
# ---------------------------------------------------------------------------
model = None
device = None
model_load_error = None

try:
    model, device = get_model_and_device()
except FileNotFoundError as e:
    model_load_error = str(e)
except RuntimeError as e:
    model_load_error = (
        "The model file was found but could not be loaded. This usually means "
        "the saved weights don't match the ResNet18 architecture defined in "
        "src/model.py.\n\nTechnical detail: " + str(e)
    )
except Exception as e:
    model_load_error = f"An unexpected error occurred while loading the model: {e}"

if model_load_error:
    st.error(f"⚠️ Could not load the model.\n\n{model_load_error}")
    st.stop()

if device is not None and device.type == "cpu":
    st.caption("Running on CPU (CUDA not available). Inference may be slower.")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.write("Upload a chest X-ray to analyze")
uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>Research Prototype</strong> — This AI-generated prediction is for
        educational/research purposes only and is not a medical diagnosis.
        Please consult a qualified healthcare professional for medical interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Read + validate image
# ---------------------------------------------------------------------------
try:
    original_image = Image.open(uploaded_file)
    original_image.load()
    original_image = original_image.convert("RGB")
except UnidentifiedImageError:
    st.error(
        "⚠️ This file could not be read as an image. "
        "Please upload a valid JPG or PNG chest X-ray."
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ The uploaded file appears to be corrupted or unsupported: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Original X-ray + Prediction
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">Original X-ray</div>', unsafe_allow_html=True)
    st.image(original_image, use_container_width=True)

try:
    input_tensor = preprocess_image(original_image)
    predicted_label, confidence, probabilities = predict(model, input_tensor, device)
    # Structured result, independent of the UI — this is what a future
    # chatbot stage will read instead of re-deriving values from display code.
    prediction_context = build_prediction_context(predicted_label, probabilities)
    st.session_state["prediction_context"] = prediction_context
except Exception as e:
    st.error(f"⚠️ Something went wrong while running the model on this image: {e}")
    st.stop()

with col2:
    st.markdown('<div class="section-header">AI Prediction</div>', unsafe_allow_html=True)
    label_class = "pred-normal" if predicted_label == "NORMAL" else "pred-pneumonia"
    st.markdown(
        f'<div class="pred-label {label_class}">{predicted_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="confidence-caption">Confidence: {confidence:.2f}% '
        "(model softmax probability, not a medical certainty)</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption(f"NORMAL probability: {prediction_context['normal_probability'] * 100:.2f}%")
    st.caption(f"PNEUMONIA probability: {prediction_context['pneumonia_probability'] * 100:.2f}%")

# ---------------------------------------------------------------------------
# Grad-CAM explanation
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">Model Explanation</div>', unsafe_allow_html=True)

predicted_idx = 0 if predicted_label == "NORMAL" else 1

try:
    overlay = generate_gradcam(model, original_image, predicted_idx)

    gc_col1, gc_col2 = st.columns(2)
    with gc_col1:
        st.image(
            original_image.resize((224, 224)),
            use_container_width=True,
            caption="Original X-ray",
        )
    with gc_col2:
        st.image(overlay, use_container_width=True, caption="Grad-CAM")

    st.markdown(
        '<div class="caption-text">Highlighted regions represent areas that '
        "influenced the model's prediction. This is not proof of disease "
        "location.</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.warning(f"⚠️ Grad-CAM explanation could not be generated: {e}")

# ---------------------------------------------------------------------------
# GroundedCXR AI Assistant (chatbot)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">GroundedCXR AI Assistant</div>', unsafe_allow_html=True)

if "prediction_context" not in st.session_state:
    st.info("Upload and analyze an X-ray to activate the GroundedCXR AI Assistant.")
else:
    # Each uploaded file gets its own conversation. If a new image has been
    # uploaded (different name/size than the one the current chat history
    # belongs to), start a fresh conversation so old answers never leak
    # into the new X-ray's context.
    current_image_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("active_image_signature") != current_image_signature:
        st.session_state["active_image_signature"] = current_image_signature
        st.session_state["chat_history"] = []

    if not is_api_key_configured():
        st.warning(
            "⚠️ Gemini API key is not configured. Please add GEMINI_API_KEY "
            "to your .env file."
        )

    # Replay the conversation so far (for this X-ray only)
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    user_question = st.chat_input(
        "Ask about this prediction (e.g. 'Why did the model predict pneumonia?')"
    )

    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_chatbot_response(
                    user_question=user_question,
                    prediction_context=st.session_state["prediction_context"],
                    chat_history=st.session_state["chat_history"][:-1],
                )
            st.write(answer)

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# ---------------------------------------------------------------------------
# Footer disclaimer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
    <strong>Research Prototype</strong> — This AI-generated prediction is for
    educational/research purposes only and is not a medical diagnosis.
    Please consult a qualified healthcare professional for medical interpretation.
    </div>
    """,
    unsafe_allow_html=True,
)