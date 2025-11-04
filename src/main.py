import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import sys
import os

sys.path.append('core')
from core.network import NeuralNetwork

st.set_page_config(
    page_title="MNIST Digit recognizer",
    page_icon="🔢",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        font-size: 4rem;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 2rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource 
def load_model(model_path="./models/mnist.pkl"): #most unessary function of the day
    """Loads the model"""
    return NeuralNetwork.load_mode(model_path)

def preprocess_canvas_image(canvas_data):
    """Convert canvas drawing into MNIST Format: 28x28"""
    if canvas_data is None:
        return None, None
    
    img = canvas_data.image_data

    img = Image.fromarray(img.astype('uint8'), 'RGBA')
    img = img.convert('L') #to grayscale

    img = ImageOps.invert(img) #mnist has blanc digits on a noir background

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img).astype('float32') / 255.0

    img_flat = img_array.reshape(1, 784)

    return img_flat, img_array

st.markdown('<h1 class="main-header">MNIST Digit Recognizer</h1>', unsafe_allow_html=True)
st.markdown("### Draw a digit (0-9). The AI will predict it.")

#load
model = load_model()

if model is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Draw here")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=20,
            stroke_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    
    with col2:
        st.markdown("#### Prediction")

        if canvas_result.image_data is not None:
            if np.sum(canvas_result.image_data[:, :, 3]) > 0:
                processed_img, img_28x28 = preprocess_canvas_image(canvas_result)

                if processed_img is not None:
                    probabilities = model.forward(processed_img)
                    prediction = np.argmax(probabilities[0])
                    confidence = probabilities[0][prediction] * 100

                    st.markdown(f'<div class="prediction-box">Prediction: {prediction}</div>', unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence-text">Confidence: {confidence:.1f}%</p>', unsafe_allow_html=True)

                    with st.expander("All probs"):
                        for digit in range(10):
                            prob = probabilities[0][digit] * 100
                            st.write(f"Digit {digit}: {prob:.2f}%")
                            st.progress(prob / 100)
        else:
            st.info("Draw a digit on the canvas")
else:
    st.info("Draw a digit on the canvas")

