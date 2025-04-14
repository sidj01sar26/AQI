import os

import streamlit as st
import tensorflow as tf

# Set page config
st.set_page_config(page_title="AQI Prediction App", layout="centered")

# ----------------------------- #
#           UI HEADER           #
# ----------------------------- #
st.title("🌍 AQI Prediction App")
st.markdown(
    """
    This web application uses a deep learning model to *predict the Air Quality Index (AQI)* 
    based on environmental parameters including *PM2.5, PM10, NO2, CO, O3, and SO2*.
    """
)

# ----------------------------- #
#       SIDEBAR INPUTS         #
# ----------------------------- #
st.sidebar.title("🛠 Input Parameters")
st.sidebar.date_input("📅 Select Date")

PM2_5 = st.sidebar.number_input(
    "PM2.5", min_value=0.0, max_value=500.0, value=25.0, step=10.0
)
PM10 = st.sidebar.number_input(
    "PM10", min_value=0.0, max_value=500.0, value=25.0, step=10.0
)
NO2 = st.sidebar.number_input(
    "NO2", min_value=0.0, max_value=500.0, value=25.0, step=10.0
)
CO = st.sidebar.number_input("CO", min_value=0.0, max_value=40.0, value=0.5, step=0.5)
O3 = st.sidebar.number_input(
    "O3", min_value=0.0, max_value=1000.0, value=25.0, step=10.0
)
SO2 = st.sidebar.number_input(
    "SO2", min_value=0.0, max_value=2000.0, value=25.0, step=10.0
)


# ----------------------------- #
#       INPUT PROCESSING       #
# ----------------------------- #
def preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2):
    return tf.constant([[PM2_5, PM10, NO2, CO, O3, SO2]], dtype=tf.float32)


input_data = preprocess_input(PM2_5, PM10, NO2, CO, O3, SO2)

# ----------------------------- #
#     MODEL SELECTION + LOAD   #
# ----------------------------- #
model_names = ["model_91", "model_169", "model_187"]
model_selection = st.selectbox("📦 Select Model", model_names)

model_path = f"./DL_Models/models/{model_selection}.h5"
model = tf.keras.models.load_model(model_path)

# ----------------------------- #
#         PREDICTION UI        #
# ----------------------------- #
col1, col2 = st.columns([1, 1])
col1.subheader("🎯 Click to Predict AQI")

if col2.button("Predict"):
    prediction = model.predict(input_data)[0][0]
    st.subheader("📊 AQI Prediction:")

    # Define color ranges for AQI
    color_ranges = {
        (0, 50): "#1FE140",  # Good
        (51, 100): "#F5B700",  # Satisfactory
        (101, 150): "#F26430",  # Moderate
        (151, 200): "#DF2935",  # Poor
        (201, 300): "#D77A61",  # Very Poor
        (301, float("inf")): "#4D5061",  # Severe
    }

    aqi_quality_table = {
        (0, 50): "Good",
        (51, 100): "Satisfactory",
        (101, 150): "Moderate",
        (151, 200): "Poor",
        (201, 300): "Very Poor",
        (301, float("inf")): "Severe",
    }

    # Match AQI range
    prediction_color = "black"
    aqi_quality = ""
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            aqi_quality = aqi_quality_table[range_]
            break

    # Display prediction box
    st.markdown(
        f"""
        <div style="background-color:{prediction_color}; padding:16px; border-radius:10px;">
            <p style="color:white; font-size:22px; text-align:center;"><strong>{aqi_quality}</strong></p>
            <p style="color:white; font-size:26px; text-align:center;"><strong>Predicted AQI: {prediction:.2f}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ----------------------------- #
#     PROJECT INFORMATION      #
# ----------------------------- #
st.markdown(
    """
This app compares different deep learning models for AQI prediction. Each model name reflects the 
**number of trainable parameters**. More parameters generally mean more complex and powerful models.
"""
)

# ----------------------------- #
#         IMAGE SLIDER         #
# ----------------------------- #
image_folder = "./DL_Models/images"
image_files = sorted(os.listdir(image_folder))

st.subheader("🖼 Project Images")
image_index = st.slider("Browse Images", 0, len(image_files) - 1, 0)

image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split(".")[0].replace("_", " ").title()

st.title(image_title)
st.image(image_path, use_container_width=True)

st.markdown("---")
