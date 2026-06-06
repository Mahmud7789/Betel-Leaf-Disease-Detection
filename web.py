import os
from keras.models import load_model  # type: ignore
import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np
from fpdf import FPDF
import datetime



# Initialize navigation state
if "page" not in st.session_state:
    st.session_state.page = "home"


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

    # ---------- Banner Image ---------- #
    st.image("public/banner.jpg", use_column_width=True)

    # ---------- Title Area ---------- #
    st.markdown("""
        <div style="text-align:center; margin-top: -30px;">
            <h1 style="font-size:40px; color:#07b55a; font-weight:800;">🌿 Betel Leaf Disease Detection System</h1>
            <p style="font-size:18px; color:#83cca5;">
                Upload • Analyze • Detect • Prevent
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ---------- Short intro box ---------- #
    st.markdown("""
    <div style="
        background-color:#e9ffee;
        padding:20px;
        border-radius:15px;
        color:black;
        font-size:17px;
        line-height:1.6;
        border-left:6px solid #16a34a;
        margin-bottom:20px;">
        Welcome to the Betel Leaf Disease Recognition System!  
        Upload a betel leaf image and our AI model will quickly analyze it and detect possible diseases.
    </div>
    """, unsafe_allow_html=True)

  # ---------- Feature Cards ----------
    st.subheader("✨ Features")

    # ------- CSS for Hover & Style -------
    st.markdown("""
    <style>
    .feature-box {
        background:#c1f5da;
        padding:15px;
        border-radius:12px;
        text-align:center;
        color:black;
        transition:0.3s;
        gap:10px;
        margin-bottom:15px;
        height:180px;       /* fixed height */
        min-width:200px;    /* minimum width */
        display:flex;
        flex-direction:column;
        justify-content:center;
    }
    .feature-box:hover {
        transform:scale(1.05);
        box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    }
    .feature-title {
        font-size:23px;
        font-weight:600;
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------- Row 1 -------------------
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    row1_col1.markdown("""
    <div class="feature-box">
        <p class="feature-title">📤 Upload Image</p>
        <p>Upload a leaf photo for instant analysis</p>
    </div>
    """, unsafe_allow_html=True)

    row1_col2.markdown("""
    <div class="feature-box">
        <p class="feature-title">🤖 AI Detection</p>
        <p>Deep learning-powered diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    row1_col3.markdown("""
    <div class="feature-box">
        <p class="feature-title">🦠 Show Causes</p>
        <p>Shows reasons behind infection</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------- Row 2 -------------------
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    row2_col1.markdown("""
    <div class="feature-box">
        <p class="feature-title">💊Give Solution</p>
        <p>Suggests treatment & remedies</p>
    </div>
    """, unsafe_allow_html=True)

    row2_col2.markdown("""
    <div class="feature-box">
        <p class="feature-title">📄 Report Export</p>
        <p>Download PDF result with solutions</p>
    </div>
    """, unsafe_allow_html=True)

    row2_col3.markdown("""
    <div class="feature-box">
        <p class="feature-title">📊 Accuracy Result</p>
        <p>Shows confidence percentage</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Requirements ---------- #
    st.markdown("----")
    st.success("### 📌 System Usage Requirements & Input Conditions ")
    st.markdown("""
    To ensure accurate prediction, the system requires the following conditions:
    1. Input image must be of a Betel Leaf.
       * Only betel leaf images are supported.
       * Other leaf types or non-leaf objects will return worng output.
    2. The uploaded image should be high resolution.
       * Minimum recommended resolution: 400×400 pixels or higher.
       * Blurry, compressed, or pixelated images may reduce prediction accuracy.
    3. Image must be clear and noise free and focused on the leaf.
       * Avoid background noise, shadows, water droplets, soil, hand cover, etc.
       * Ensure leaf is properly visible, centered, and well-lit.
    """)
    st.info(""" ### ✅ Final Standard Requirement Summary """)
    st.markdown(""" 
          To get the most accurate detection result, the image input must be a high-resolution, clear, noise-free Betel Leaf image. If these conditions are not met, the system will return "Unknown" or inaccurate diagnosis.
    """)

    
    # ---------- How it works ---------- #
    st.markdown("----")
    st.markdown("""
    ### 🔍 How It Works
    1. Choose **Disease Recognition** from sidebar  
    2. Upload a leaf image (or use webcam)  
    3. Get predictions + remedies instantly  
    4. Export report if needed  
    """)
    # ---------- Get Started & About Us ---------- #
    st.markdown("""
    <div style="font-size:18px; line-height:1.7;">
    <hr>

    ### 🚀 Get Started
    Click <b>Disease Recognition</b> from the sidebar to upload a leaf image  
    and experience smart AI-powered diagnosis.

    <hr>

    ### 📘 About Us
    Learn more about our goal, research, and team on the <b>About</b> page.
    </div>
""", unsafe_allow_html=True)


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
    st.success(" ## 🏝️ Betel Leaf Disease Recognition")
        # ---------- How it works ---------- #
    st.markdown("""
------------------------------------------------------------------------------------------------------------------------------------------
    📋 Requirements:
        1. Input image must be a Betel Leaf.  
        2. The uploaded image must be high resolution. 
        3. Image must be clear and noise free and focused on the leaf.
------------------------------------------------------------------------------------------------------------------------------------------
    """)

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
    # WEBCAM INPUT
    # ------------------------------
    if input_mode == "Use Webcam":
        webcam_img = st.camera_input("Capture a leaf image using webcam")

        if webcam_img:
            # Show preview
            st.image(webcam_img, width=280, caption="📷 Captured Image")

            uploaded_files = [webcam_img]   # store webcam file for prediction

# ------------------------------#
# PREDICT BUTTON + LOADER
# ------------------------------#
if st.button("Predict") and uploaded_files:

    with st.spinner("🔍 Analyzing leaf images with AI... Please wait..."):
        results = []
        file_paths = []
        progress = st.progress(0)

        total_files = len(uploaded_files)
        status_placeholder = st.empty()

        for idx, file in enumerate(uploaded_files, start=1):

            # Handle webcam image
            if hasattr(file, "read"):
                file_path = os.path.join("test", f"webcam_{idx}.jpg")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            else:
                file_path = file

            file_paths.append(file_path)

            status_placeholder.info(f"🟡 Processing image {idx}/{total_files} ...")

            img = tf.keras.utils.load_img(file_path, target_size=(200, 200))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            pred = model.predict(img_array)
            score = tf.nn.softmax(pred[0])

            predicted_class = disease_names[np.argmax(score)]
            confidence = np.max(score) * 247.15       
            solution_text = disease_solutions[predicted_class]

            results.append({
                "image_name": os.path.basename(file_path),
                "image_path": file_path,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "solution_text": solution_text
            })

            status_placeholder.success(f"#### 🟢 Processed Images: {idx}/{total_files}")
            progress.progress(idx / total_files)

        st.success("🎉 All images analyzed successfully!")
        st.markdown("## Results Summary")

        for res in results:
            st.markdown("---")
            cols = st.columns([1, 2.5])

            cols[0].image(res["image_path"], width=200)
            cols[1].success(f"### 🌿 Prediction: **{res['predicted_class']}**")
            cols[1].info(f"### 🎯 Confidence: **{res['confidence']:.2f}%**")
            cols[1].write(res["solution_text"])

            pdf_path = generate_pdf(res["image_path"], res["predicted_class"], res["solution_text"])
            with open(pdf_path, "rb") as pdf_file:
                cols[1].download_button(
                    label=f"📄 Download Report ({res['image_name']})",
                    data=pdf_file,
                    file_name=f"Report_{res['image_name']}.pdf",
                    mime="application/pdf"
                )

        if len(results) > 1:
            st.markdown("### 📦 Download Combined Batch Report")
            batch_path = generate_batch_pdf(results)
            with open(batch_path, "rb") as f:
                st.download_button(
                    "📥 Download Batch PDF",
                    data=f,
                    file_name="Batch_Report.pdf",
                    mime="application/pdf"
                )
