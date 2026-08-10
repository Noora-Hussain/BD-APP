import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model


# Page Configuration
st.set_page_config(page_title="Bahraini Currency Recognition",
                   page_icon="💵",
                   layout="wide")


# Money Theme
st.markdown("""
<style>

    /* =========================
       MAIN PAGE
       ========================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #f4f8f1 0%,
            #ffffff 50%,
            #eef6ed 100%
        );
    }


    /* =========================
       ALL TEXT
       ========================= */

    .stMarkdown p,
    .stMarkdown li,
    .stText,
    label {
        color: #26352b !important;
    }


    /* =========================
       MAIN TITLE
       ========================= */

    .main-title {
        text-align: center;
        color: #176b3a !important;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #536653 !important;
        font-size: 18px;
        margin-bottom: 30px;
    }


    /* =========================
       HEADINGS
       ========================= */

    h1,
    h2,
    h3,
    h4 {
        color: #176b3a !important;
    }


    /* =========================
       TABS
       ========================= */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 25px;
        color: #176b3a !important;
        font-weight: 700;
        border-radius: 10px 10px 0px 0px;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #176b3a !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #176b3a !important;
        color: white !important;
    }

    .stTabs [aria-selected="true"] p {
        color: white !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stFormSubmitButton > button {
        background: linear-gradient(
            90deg,
            #176b3a,
            #2f8f57
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 25px;
    }

    .stFormSubmitButton > button p {
        color: white !important;
    }

    .stFormSubmitButton > button:hover {
        background: linear-gradient(
            90deg,
            #12542d,
            #247343
        ) !important;
        color: white !important;
    }


    /* =========================
       INFO CARDS
       ========================= */

    .info-card {
        background-color: #ffffff !important;
        color: #26352b !important;

        padding: 20px;

        border-radius: 15px;

        border-left: 5px solid #c5a34a;

        box-shadow:
            0px 4px 12px rgba(23, 107, 58, 0.08);

        margin-bottom: 20px;
    }

    .info-card h3 {
        color: #176b3a !important;
        margin-bottom: 10px;
    }

    .info-card p {
        color: #26352b !important;
        font-size: 17px;
    }

    .info-card b {
        color: #176b3a !important;
    }


    /* =========================
       METRIC
       ========================= */

    [data-testid="stMetric"] {
        background-color: #ffffff !important;

        border: 2px solid #d4e5d2;

        border-radius: 12px;

        padding: 15px;

        box-shadow:
            0px 3px 10px rgba(23, 107, 58, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #536653 !important;
    }

    [data-testid="stMetricLabel"] p {
        color: #536653 !important;
    }

    [data-testid="stMetricValue"] {
        color: #176b3a !important;
    }

    [data-testid="stMetricValue"] div {
        color: #176b3a !important;
    }


    /* =========================
       FILE UPLOADER
       ========================= */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #9ab88e !important;
        border-radius: 15px !important;
        padding: 10px !important;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        background-color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #26352b !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #536653 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background-color: #f4f8f1 !important;
        color: #176b3a !important;
        border: 1px solid #9ab88e !important;
    }
    
    [data-testid="stFileUploader"] button span {
        color: #176b3a !important;
    }
    
    [data-testid="stFileUploader"] p {
        color: #26352b !important;
    }

    /* =========================
       CHECKBOX
       ========================= */

    [data-testid="stCheckbox"] label {
        color: #26352b !important;
    }

    [data-testid="stCheckbox"] p {
        color: #26352b !important;
    }


    /* =========================
       TABLE
       ========================= */

    [data-testid="stTable"] {
        background-color: #ffffff !important;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stTable"] table {
        background-color: #ffffff !important;
    }

    [data-testid="stTable"] th {
        background-color: #176b3a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stTable"] th * {
        color: #ffffff !important;
    }

    [data-testid="stTable"] td {
        background-color: #ffffff !important;
        color: #26352b !important;
    }

    [data-testid="stTable"] td * {
        color: #26352b !important;
    }


    /* =========================
       DATAFRAME
       ========================= */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
    }


    /* =========================
       TIPS
       ========================= */

    .tips-text {
        color: #26352b !important;
        font-size: 17px;
        line-height: 1.9;
    }

    .tips-text p {
        color: #26352b !important;
    }


    /* =========================
       WARNING / SUCCESS
       ========================= */

    [data-testid="stAlert"] {
        border-radius: 12px;
    }


</style>
""", unsafe_allow_html=True)



# Class Names
class_names = ["0.05",
                "0.100",
                "0.25",
                "0.5 BD",
                "0.50",
                "10 BD",
                "1BD",
                "20 BD",
                "5 BD"]


# Load Model Once
@st.cache_resource
def load_currency_model():
    return load_model("currency_model.keras")


model = load_currency_model()


# Main Header
st.markdown(
    '<div class="main-title">💵 Bahraini Currency Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Currency Classification System</div>',
    unsafe_allow_html=True
)


# Tabs
tab1, tab2, tab3 = st.tabs(["💵 Recognize Currency", "📊 Model Information", "📖 How to Use"])




# ---
# TAB 1 - RECOGNIZE CURRENCY
# ---

with tab1:
    st.header("💵 Recognize Bahraini Currency")
    st.write("Upload an image of Bahraini currency and let the Deep Learning "
             "model identify the currency.")

    with st.form("currency_form"):
        uploaded_file = st.file_uploader("Choose a currency image", type=["jpg", "jpeg", "png"])
        predict_button = st.form_submit_button("🔍 Predict Currency")

    if predict_button:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image) / 255.0
            image_array = np.expand_dims(image_array,axis=0)

            # Prediction
            try:
                prediction = model.predict(image_array, verbose=0)
            
                predicted_index = int(np.argmax(prediction[0]))
                predicted_currency = class_names[predicted_index]
                confidence = float(prediction[0][predicted_index] * 100)
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()

            # Display Image and Result
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Prediction Result")
                st.success(f"Predicted Currency: {predicted_currency}")
                st.metric(label="Confidence", value=f"{confidence}%")
                
                if confidence < 50:
                    st.warning("The confidence is low. Try a clearer image "
                                "with better lighting.")

            # Show probabilities
            show_probabilities = st.checkbox("Show probabilities for all classes")

            if show_probabilities:
                probabilities_df = pd.DataFrame({"Currency": class_names,
                                                 "Probability (%)": np.round(prediction[0] * 100, 2)})

                st.subheader("All Class Probabilities")
                st.dataframe(probabilities_df, use_container_width=True)


# ---
# TAB 2 - MODEL INFORMATION
# ---

with tab2:
    st.header("📊 Model Information")
    st.markdown(
            """
            <div class="info-card">
    
            <h3>About the Model</h3>
    
            <p>
            This project uses a Deep Learning classification model
            to recognize Bahraini currency from an uploaded image.
            </p>
    
            </div>
            """,
            unsafe_allow_html=True)

    st.metric("Number of Currency Classes", 9)
    st.subheader("💰 Currency Classes")
    classes_table = pd.DataFrame({"Class Number": range(1, 10),
                                  "Currency": class_names})

    st.table(classes_table)
    st.subheader("🔄 Model Steps")
    st.markdown(
            """
            <div class="info-card">
    
            <p>1. Upload a currency image</p>
    
            <p>2. Resize the image to 224 × 224</p>
    
            <p>3. Normalize pixel values</p>
    
            <p>4. Send the image to the trained model</p>
    
            <p>5. Predict the currency class</p>
    
            <p>6. Display the predicted currency and confidence</p>
    
            </div>
            """,
            unsafe_allow_html=True)





# ---
# TAB 3 - HOW TO USE
# ---

with tab3:

    st.header("📖 How to Use")
    st.markdown(
            """
            <div class="info-card">
    
            <h3>Step 1</h3>
    
            <p>
            Upload a clear image of a Bahraini currency note or coin.
            </p>
    
            </div>
    
    
            <div class="info-card">
    
            <h3>Step 2</h3>
    
            <p>
            Click the <b>Predict Currency</b> button.
            </p>
    
            </div>
    
    
            <div class="info-card">
    
            <h3>Step 3</h3>
    
            <p>
            The model will display the predicted currency and its confidence.
            </p>
    
            </div>
    
    
            <div class="info-card">
    
            <h3>Step 4</h3>
    
            <p>
            You can enable <b>Show probabilities for all classes</b>
            to see the prediction probability for every currency class.
            </p>
    
            </div>
            """,
            unsafe_allow_html=True)
    

    st.subheader("💡 Tips for Better Results")
    st.markdown(
            """
            <div class="tips-text">
    
            • Use a clear image<br>
    
            • Make sure the currency is visible<br>
    
            • Use good lighting<br>
    
            • Avoid excessive blur<br>
    
            • Try different backgrounds and angles
    
            </div>
            """,
            unsafe_allow_html=True
        )
