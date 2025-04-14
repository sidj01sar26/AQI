import streamlit as st
import pickle
import os
import pandas as pd

# Set page config
st.set_page_config(page_title="AQI Prediction App", layout="centered")

# Load pre-trained components
encoder = pickle.load(open("./objects/encoder.pkl", "rb"))
scaler = pickle.load(open("./objects/scaler.pkl", "rb"))
feature_cols = pickle.load(open("./objects/feature_cols.pkl", "rb"))
city_list = pickle.load(open("./objects/city_list.pkl", "rb"))
model = pickle.load(open("./objects/model_no_3.pkl", "rb"))

# ----------------------------- #
#           UI HEADER           #
# ----------------------------- #
st.title("🌍 AQI Prediction App")
st.markdown(
    """
    This web application uses a trained machine learning model to **predict the Air Quality Index (AQI)** 
    based on environmental parameters including **PM2.5, PM10, NO, NO2, CO, O3, and SO2**.
    """
)

# ----------------------------- #
#       SIDEBAR INPUTS         #
# ----------------------------- #
st.sidebar.title("🛠️ Input Parameters")
st.sidebar.date_input("📅 Select Date")
City = st.sidebar.selectbox("🏙️ Select a City", city_list)

PM2_5 = st.sidebar.number_input(
    "PM2.5", min_value=0.0, max_value=500.0, value=25.0, step=10.0
)
PM10 = st.sidebar.number_input(
    "PM10", min_value=0.0, max_value=500.0, value=25.0, step=10.0
)
NO = st.sidebar.number_input(
    "NO", min_value=0.0, max_value=500.0, value=25.0, step=10.0
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
input_data = {
    "City": City,
    "PM2.5": PM2_5,
    "PM10": PM10,
    "NO": NO,
    "NO2": NO2,
    "CO": CO,
    "O3": O3,
    "SO2": SO2,
}

data = {}
for feature in feature_cols:
    base_feature = feature.split("_")[0]
    if base_feature in input_data:
        data[feature] = input_data[base_feature]

# Encode city
city_encoded_dict = dict(zip(city_list, encoder.transform(city_list)))
data["City_Encoded"] = city_encoded_dict[City]

# Create DataFrame and scale
data_df = pd.DataFrame([data], columns=feature_cols)
data_scaled = scaler.transform(data_df)

# ----------------------------- #
#         PREDICTION UI        #
# ----------------------------- #
col1, col2 = st.columns([1, 1])
col1.subheader("🎯 Click to Predict AQI")

if col2.button("Predict"):
    prediction = model.predict(data_scaled)[0]
    st.subheader("📊 AQI Prediction:")

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
image_folder = "images"
image_files = sorted(os.listdir(image_folder))

st.subheader("🖼️ Project Images")
image_index = st.slider("Browse Images", 0, len(image_files) - 1, 0)

image_path = os.path.join(image_folder, image_files[image_index])
image_title = image_files[image_index][2:].split(".")[0].replace("_", " ").title()

st.title(image_title)
st.image(image_path, use_container_width=True)

st.markdown("---")
