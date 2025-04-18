import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import os
import google.generativeai as genai

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        st.error("No file uploaded.")
        st.stop()

def get_gemini_response(input_text, image_parts, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_text, image_parts[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Streamlit App
st.set_page_config(page_title="Object Detection system")
st.header("Object Detection system")

input_text = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

input_prompt = """
You are an expert in analyzing images for a HARC model. Provide a concise, 50-word description of each image, focusing on the setting, visible objects, and human activity. Highlight key postures or movements relevant to the classification, ensuring clarity and relevance to the recognition task.
"""

if st.button("Search"):
    try:
        image_data = input_image_setup(uploaded_file)
        st.write("Processing image and input...")
        response = get_gemini_response(input_text, image_data, input_prompt)
        if response:
            st.subheader("The Response is:")
            st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")