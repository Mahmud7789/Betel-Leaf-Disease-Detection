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
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "View Saved Reports", "About"])


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
------------------------------------------------------------------------------------------------------------------------------------------
    🌿 Betel Leaf Disease Detection System
    The Betel Leaf Disease Detection System is an AI-powered tool that helps farmers
    and plant enthusiasts quickly detect common diseases in betel leaves.
    Upload images or use your webcam, get predictions, view solutions,
    and generate PDF reports — all in one place!
------------------------------------------------------------------------------------------------------------------------------------------
    👨‍💻 Team: Access Denied

    Team Members:

        🙎‍♂️Golam Mahmud

        🙎‍♂️Shaon Akan

        🙎‍♂️Dhurba Bakchi

    We created this system to combine AI and agriculture for real-world impact.
------------------------------------------------------------------------------------------------------------------------------------------
    📊 Dataset (2,100 images)

        🦠 Bacterial Leaf Disease – 230 images

        🌱 Healthy Leaf – 510 images

        🍂 Leaf Spot Disease – 500 images

        ⚫ Fungal Brown Disease – 350 images

        🌵 Dried Leaf Disease – 500 images
------------------------------------------------------------------------------------------------------------------------------------------

    ✨ Key Features

        🔍 AI-based disease detection

        📷 Single & multiple image upload

        📄 PDF report generation & inline preview

        💾 Save, view, download, and delete reports

        🖥️ User-friendly Streamlit interface
-----------------------------------------------------------------------------------------------------------------------------------------
    ### 🎯 Goal

    Help farmers detect leaf diseases early, reduce crop loss, and provide a fast, accessible digital solution for healthier betel crops.
------------------------------------------------------------------------------------------------------------------------------------------
    """)
# ============================================================
# VIEW SAVED REPORTS PAGE (WITH DELETE FEATURE)
# ============================================================
elif app_mode == "View Saved Reports":
    st.header("📁 Saved Reports")

    single_folder = "reports/single_reports"
    batch_folder = "reports/batch_reports"

    # --------------------------------------------------------
    # Function to delete files
    # --------------------------------------------------------
    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)
            st.success(f"Deleted: {os.path.basename(path)}")
            st.experimental_rerun()  # Refresh page

    # --------------------------------------------------------
    # SINGLE REPORTS
    # --------------------------------------------------------
    st.subheader("📄 Single Image Reports")
    if not os.path.exists(single_folder) or not os.listdir(single_folder):
        st.info("No single image reports found.")
    else:
        single_pdfs = sorted(os.listdir(single_folder))

        for pdf_file in single_pdfs:
            pdf_path = os.path.join(single_folder, pdf_file)
            cols = st.columns([4, 1, 1])

            cols[0].write(pdf_file)

            with open(pdf_path, "rb") as f:
                cols[1].download_button(
                    label="⬇️ Download",
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf",
                    key=f"download_single_{pdf_file}"
                )

            # DELETE BUTTON
            if cols[2].button("🗑️ Delete", key=f"delete_single_{pdf_file}"):
                delete_file(pdf_path)

    # --------------------------------------------------------
    # BATCH REPORTS
    # --------------------------------------------------------
    st.subheader("📦 Batch Reports")
    if not os.path.exists(batch_folder) or not os.listdir(batch_folder):
        st.info("No batch reports found.")
    else:
        batch_pdfs = sorted(os.listdir(batch_folder))

        for pdf_file in batch_pdfs:
            pdf_path = os.path.join(batch_folder, pdf_file)
            cols = st.columns([4, 1, 1])

            cols[0].write(pdf_file)

            with open(pdf_path, "rb") as f:
                cols[1].download_button(
                    label="⬇️ Download",
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf",
                    key=f"download_batch_{pdf_file}"
                )

            # DELETE BUTTON
            if cols[2].button("🗑️ Delete", key=f"delete_batch_{pdf_file}"):
                delete_file(pdf_path)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    model = load_model('Betel_leaf_disease_recognition_model.keras')
    # Create folders for saving reports
    os.makedirs("reports/single_reports", exist_ok=True)
    os.makedirs("reports/batch_reports", exist_ok=True)
    # ------------------------------
    # PDF FUNCTION (Single Image)
    # ------------------------------
    def generate_pdf(image_path, predicted_class, solution_text):
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Betel Leaf Disease Detection Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        pdf.cell(0, 10, f"Prediction: {predicted_class}", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

        pdf.ln(5)
        pdf.multi_cell(0, 8, f"Recommended Solution:\n{solution_text}")

        # Add Image
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Uploaded Image:", ln=True)
        pdf.image(image_path, x=60, y=180, w=80)

    # File name
        pdf_name = f"Report_{os.path.basename(image_path)}.pdf"
        pdf_path = os.path.join("reports/single_reports", pdf_name)
        pdf.output(pdf_path)
        return pdf_path

    # ------------------------------
    # PDF FUNCTION (BATCH REPORT)
    # ------------------------------
    def generate_batch_pdf(results):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Batch Disease Detection Report", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Arial", size=12)

        for item in results:
            pdf.cell(0, 10, f"Image: {item['image_name']}", ln=True)
            pdf.cell(0, 10, f"Prediction: {item['predicted_class']}", ln=True)
            pdf.multi_cell(0, 8, f"Solution:\n{item['solution_text']}")
            pdf.ln(5)
            pdf.image(item["image_path"], x=60, w=80)
            pdf.add_page()

       # Save in batch_reports folder
        batch_pdf_path = os.path.join("reports/batch_reports", "Batch_Report.pdf")
        pdf.output(batch_pdf_path)
        return batch_pdf_path

    # ------------------------------
    # INPUT MODE SELECTION
    # ------------------------------
    input_mode = st.radio("Choose Input Method", [
        "Upload Single Image",
        "Upload Multiple Images",
        "Use Webcam"
    ])

    uploaded_files = []
    webcam_image_path = None

    # ------------------------------
    # SINGLE IMAGE UPLOAD
    # ------------------------------
    if input_mode == "Upload Single Image":
        single_file = st.file_uploader("Upload one leaf image", type=["jpg", "jpeg", "png"])
        if single_file:
            st.image(single_file, width=270, caption=f"Uploaded: {single_file.name}")
            uploaded_files = [single_file]

    # ------------------------------
    # MULTIPLE IMAGE UPLOAD
    # ------------------------------
    if input_mode == "Upload Multiple Images":
        uploaded_files = st.file_uploader(
            "Upload multiple leaf images",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_files:
            st.write("### Uploaded Images:")
            cols = st.columns(3)
            idx = 0

            for img in uploaded_files:
                cols[idx].image(img, width=200, caption=img.name)
                idx = (idx + 1) % 3

    # ------------------------------
    # WEBCAM MODE
    # ------------------------------
    if input_mode == "Use Webcam":
        captured_image = st.camera_input("Capture a leaf image")
        if captured_image:
            webcam_image_path = os.path.join("test", "webcam_leaf.jpg")
            with open(webcam_image_path, 'wb') as f:
                f.write(captured_image.getbuffer())
            st.image(webcam_image_path, caption="Captured Image")
            uploaded_files = [webcam_image_path]

    # ------------------------------
    # PREDICT BUTTON
    # ------------------------------
    if st.button("Predict") and uploaded_files:

        results = []
        file_paths = []

        st.write("### Processing Images...")
        progress = st.progress(0)

        # Convert uploaded files → actual file paths
        total_files = len(uploaded_files)
        counter = 0

        for file in uploaded_files:

            # Handle webcam (already a filepath)
            if isinstance(file, str):
                file_path = file
            else:
                file_path = os.path.join("test", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            file_paths.append(file_path)

            # ---------------- PREDICT ----------------
            img = tf.keras.utils.load_img(file_path, target_size=(200, 200))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            pred = model.predict(img_array)
            score = tf.nn.softmax(pred[0])

            predicted_class = disease_names[np.argmax(score)]
            solution_text = disease_solutions[predicted_class]

            # Store results
            results.append({
                "image_name": os.path.basename(file_path),
                "image_path": file_path,
                "predicted_class": predicted_class,
                "solution_text": solution_text
            })

            # Update progress bar
            counter += 1
            progress.progress(counter / total_files)

        # ------------------------------
        # DISPLAY RESULTS
        # ------------------------------
        st.markdown("## Results Summary:")

        for res in results:
            st.markdown("---")
            st.subheader(f"🖼️ Processing: **{os.path.basename(res['image_name'])}**")
            cols = st.columns([1, 2])

            cols[0].image(res["image_path"], width=220)

            cols[1].success(f"### **🌿 Prediction:** {res['predicted_class']}")
            cols[1].write(res['solution_text'])

            # Single PDF Button
            pdf_path = generate_pdf(res["image_path"], res["predicted_class"], res["solution_text"])
            with open(pdf_path, "rb") as pdf_file:
                cols[1].download_button(
                    label=f"📄 Download PDF ({res['image_name']})",
                    data=pdf_file,
                    file_name=f"Report_{res['image_name']}.pdf",
                    mime="application/pdf"
                )

        # ------------------------------
        # BATCH PDF DOWNLOAD
        # ------------------------------
        if len(results) > 1:
            st.markdown("## Batch Report")
            batch_pdf_path = generate_batch_pdf(results)

            with open(batch_pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📦 Download Batch PDF Report",
                    data=pdf_file,
                    file_name="Batch_Report.pdf",
                    mime="application/pdf"
                )


