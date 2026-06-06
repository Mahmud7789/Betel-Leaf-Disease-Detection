import os
from keras.models import load_model  # type: ignore
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np

disease_names = [
    'Bacterial _Leaf_Disease',
    'Dried_Leaf',
    'Fungal_Brown_Disease',
    'Healthy_Leaf',
    'Leaf_Spot_Disease'
]

# Disease → Solutions / Recommendations
disease_solutions = {
    'Bacterial _Leaf_Disease': """
### 🌱 Solution for **Bacterial Leaf Disease**
- Remove and destroy infected leaves.
- Avoid overhead watering to reduce moisture.
- Apply copper-based bactericides.
- Ensure proper plant spacing to improve airflow.
""",

    'Dried_Leaf': """
### 🌱 Solution for **Dried Leaf Problem**
- Increase watering frequency (avoid over-watering).
- Apply balanced fertilizers.
- Improve soil moisture retention using mulch.
- Protect from excessive sunlight.
""",

    'Fungal_Brown_Disease': """
### 🌱 Solution for **Fungal Brown Disease**
- Apply anti-fungal sprays (e.g., Mancozeb or Copper Fungicides).
- Remove infected leaves immediately.
- Avoid waterlogging and improve drainage.
- Maintain proper spacing between plants.
""",

    'Healthy_Leaf': """
### 🌱 Your Leaf is Healthy 🌿
- No treatment needed!
- Continue proper care: adequate sunlight, watering, and nutrient balance.
""",

    'Leaf_Spot_Disease': """
### 🌱 Solution for **Leaf Spot Disease**
- Use fungicides like Chlorothalonil or Copper-based solutions.
- Remove infected leaves to stop disease spread.
- Avoid splashing water on leaves.
- Improve air circulation around plants.
"""
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "About"])

# Home Page
if app_mode == "Home":
    st.header("BETEL LEAF DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    ### Team: Access Denied
    **Members:**  
    - Golam Mahmud  
    - Shaon Akan  
    - Dhurba Bakchi  

    **Dataset:**  
    Contains 4.5K images across 5 disease categories.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    model = load_model('Betel_leaf_disease_recognition_model.keras')

    uploaded_file = st.file_uploader("Upload an Image")

    if uploaded_file is not None:
        file_path = os.path.join("test", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, width=250, caption="Uploaded Image")

        if st.button("Predict"):
            input_image = tf.keras.utils.load_img(file_path, target_size=(200, 200))
            input_array = tf.keras.utils.img_to_array(input_image)
            input_expand = tf.expand_dims(input_array, 0)

            predictions = model.predict(input_expand)
            result = tf.nn.softmax(predictions[0])

            predicted_class = disease_names[np.argmax(result)]

            # Display prediction result
            st.success(f"### 🌿 Prediction: **{predicted_class}**")

            # Display treatment solution
            st.markdown(disease_solutions[predicted_class])