import streamlit as st
import pickle
import os
import pandas as pd

# ----------------------------- #
#         PAGE SETTINGS         #
# ----------------------------- #
st.set_page_config(page_title="AQI Prediction App", layout="centered")

# ----------------------------- #
#     LOAD MODEL COMPONENTS     #
# ----------------------------- #
scaler = pickle.load(open("./exp/objects/scaler.pkl", "rb"))
model = pickle.load(open("./exp/objects/models/model_no_2.pkl", "rb"))
feature_cols = ["PM2.5", "PM10"]

# ----------------------------- #
#           UI HEADER           #
# ----------------------------- #
st.title("🌍 AQI Prediction App")
st.markdown(
    """
    This web application uses a trained machine learning model to **predict the Air Quality Index (AQI)** 
    based on **PM2.5 and PM10** values.
    """
)

# ----------------------------- #
#       SIDEBAR INPUTS         #
# ----------------------------- #
st.sidebar.title("🛠️ Input Parameters")

PM2_5 = st.sidebar.number_input(
    "PM2.5", min_value=0.0, max_value=500.0, value=20.0, step=10.0
)
PM10 = st.sidebar.number_input(
    "PM10", min_value=0.0, max_value=500.0, value=20.0, step=10.0
)

input_data = {
    "PM2.5": PM2_5,
    "PM10": PM10,
}

# ----------------------------- #
#       INPUT PROCESSING       #
# ----------------------------- #
data_df = pd.DataFrame([input_data], columns=feature_cols)
data_scaled = scaler.transform(data_df)

# ----------------------------- #
#         PREDICTION UI        #
# ----------------------------- #
col1, col2 = st.columns([1, 1])
col1.subheader("🎯 Click to Predict AQI")

if col2.button("Predict"):
    prediction = model.predict(data_scaled)[0]
    st.subheader("📊 AQI Prediction:")

    # Define AQI color ranges
    color_ranges = {
        (0, 50): "#1FE140",  # Good
        (51, 100): "#F5B700",  # Moderate
        (101, 150): "#F26430",  # Unhealthy for Sensitive Groups
        (151, 200): "#DF2935",  # Unhealthy
        (201, 300): "#D77A61",  # Very Unhealthy
        (301, float("inf")): "#4D5061",  # Hazardous
    }

    prediction_color = "black"
    for range_, color in color_ranges.items():
        if range_[0] <= prediction <= range_[1]:
            prediction_color = color
            break

    st.markdown(
        f'<div style="background-color:{prediction_color}; padding:16px; border-radius:10px;">'
        f'<p style="color:white; font-size:26px; text-align:center;"><strong>Predicted AQI: {int(prediction)}</strong></p>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ----------------------------- #
#         IMAGE SLIDER         #
# ----------------------------- #
image_folder = "./exp/images"
image_files = sorted(os.listdir(image_folder))

st.subheader("🖼️ Project Images")
image_index = st.slider("Browse Images", 0, len(image_files) - 1, 0)

image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split(".")[0].replace("_", " ").title()

st.title(image_title)
st.image(image_path, use_container_width=True)

st.markdown("---")