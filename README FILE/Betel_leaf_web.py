import os
from keras.models import load_model  # type: ignore
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np
from fpdf import FPDF
import datetime


disease_names = [
    'Bacterial _Leaf_Disease',
    'Dried_Leaf',
    'Fungal_Brown_Disease',
    'Healthy_Leaf',
    'Leaf_Spot_Disease'
]

# Disease → Causes + Solutions / Recommendations
disease_solutions = {
    'Bacterial _Leaf_Disease': """
### Cause of **Bacterial Leaf Disease**
- Caused by *Xanthomonas* or *Pseudomonas* bacteria.
- Spreads through water splashes, insects, and contaminated tools.
- High humidity and warm temperatures accelerate infection.
- Overcrowded plants with poor airflow increase bacterial growth.

### Solution for **Bacterial Leaf Disease**
- Remove and destroy infected leaves.
- Avoid overhead watering to reduce moisture.
- Apply copper-based bactericides.
- Ensure proper plant spacing to improve airflow.
""",

    'Dried_Leaf': """
### Cause of **Dried Leaf Problem**
- Insufficient watering or irregular watering schedule.
- Excessive sunlight causing leaf burn.
- Lack of essential nutrients (Nitrogen, Potassium).
- Poor soil moisture retention.
- Sometimes due to fungal infection or root damage.

### Solution for **Dried Leaf Problem**
- Increase watering frequency (avoid over-watering).
- Apply balanced fertilizers.
- Improve soil moisture retention using mulch.
- Protect from excessive sunlight.
""",

    'Fungal_Brown_Disease': """
### Cause of **Fungal Brown Disease**
- Caused by fungi like *Colletotrichum* or *Phytophthora*.
- Occurs due to constant moisture on leaves.
- Spread by rain splash, wind, or infected soil.
- Poor drainage and waterlogged soil promote fungal growth.

### Solution for **Fungal Brown Disease**
- Apply anti-fungal sprays (e.g., Mancozeb or Copper Fungicides).
- Remove infected leaves immediately.
- Avoid waterlogging and improve drainage.
- Maintain proper spacing between plants.
""",

    'Healthy_Leaf': """
### Your Leaf is Healthy!
- No disease detected.
- Good soil, correct watering, and balanced sunlight are keeping the plant healthy.

### Recommendation
- Continue proper care: adequate sunlight, watering, and nutrient balance.
""",

    'Leaf_Spot_Disease': """
### Cause of **Leaf Spot Disease**
- Caused by fungal pathogens such as *Cercospora*, *Phoma*, or *Alternaria*.
- Triggered by high moisture and poor air circulation.
- Spread via water splashes, infected tools, or old plant debris.
- Often worsens during rainy or humid seasons.

### Solution for **Leaf Spot Disease**
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
    Welcome to the Betel Leaf Disease Recognition System! 🌿🔍
    
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

    # Function to generate PDF report
    def generate_pdf(image_path, predicted_class, solution_text):
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Betel Leaf Disease Detection Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        pdf.cell(0, 10, f"Prediction: {predicted_class}", ln=True)
       # pdf.cell(0, 10, f"Confidence: {confidence}%", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

        pdf.ln(5)
        pdf.multi_cell(0, 8, f"Recommended Solution:\n{solution_text}")

        # Add leaf image
        pdf.ln(15)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Uploaded Image:", ln=True)
        pdf.image(image_path, x=60, y=175, w=80)

        pdf_path = "Disease_Report.pdf"
        pdf.output(pdf_path)
        return pdf_path


    # WebCam or File Upload Selector
    input_mode = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

    captured_image = None
    file_path = None

    # -------------------------------
    # 1️⃣ Webcam Capture
    # -------------------------------
    if input_mode == "Use Webcam":
        captured_image = st.camera_input("Capture a leaf image")

        if captured_image:
            file_path = os.path.join("test", "webcam_leaf.jpg")
            with open(file_path, 'wb') as f:
                f.write(captured_image.getbuffer())
            st.image(file_path, width=250, caption="Captured Image")

    # -------------------------------
    # 2️⃣ File Upload
    # -------------------------------
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a leaf image")

        if uploaded_file:
            file_path = os.path.join("test", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(uploaded_file, width=250, caption="Uploaded Image")


    # -------------------------------
    # Prediction Button
    # -------------------------------
    if file_path and st.button("Predict"):
        input_image = tf.keras.utils.load_img(file_path, target_size=(200, 200))
        input_array = tf.keras.utils.img_to_array(input_image)
        input_expand = tf.expand_dims(input_array, 0)

        predictions = model.predict(input_expand)
        result = tf.nn.softmax(predictions[0])

        predicted_class = disease_names[np.argmax(result)]
        confidence = round(100 * np.max(result), 2)
        solution_text = disease_solutions[predicted_class]

        # Display results
        st.success(f"### 🌿 Prediction: **{predicted_class}**")
        st.markdown(f"{solution_text}")

        # -------------------------------
        # Generate and Download PDF
        # -------------------------------
        pdf_path = generate_pdf(file_path, predicted_class,solution_text)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_file,
                file_name="Betel_Leaf_Disease_Report.pdf",
                mime="application/pdf"
            )
