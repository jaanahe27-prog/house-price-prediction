import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model & Feature Names
# =========================
model = joblib.load("gradient_boosting_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================
# Header
# =========================
st.title("🏠 House Price Prediction")

st.markdown(
    """
    ### Welcome! 👋
    
    Enter the property details below and our **Machine Learning model**
    will estimate the house sale price.
    """
)

st.divider()


# =========================
# Property Information
# =========================
st.subheader("🏡 Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

with col2:
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )

with col3:
    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2000
    )


# =========================
# Area Information
# =========================
st.subheader("📐 Area & Space")

col1, col2, col3 = st.columns(3)

with col1:
    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800
    )

with col3:
    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000
    )


# =========================
# Garage Information
# =========================
st.subheader("🚗 Garage Information")

col1, col2, col3 = st.columns(3)

with col1:
    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2
    )

with col2:
    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=2000,
        value=400
    )

with col3:
    garage_year = st.number_input(
        "Garage Year Built",
        min_value=1800,
        max_value=2026,
        value=2000
    )


# =========================
# Rooms & Bathrooms
# =========================
st.subheader("🛏️ Rooms & Bathrooms")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3
    )

with col2:
    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2
    )

with col3:
    half_bath = st.number_input(
        "Half Bathrooms",
        min_value=0,
        max_value=5,
        value=1
    )


# =========================
# Prediction
# =========================
st.divider()

if st.button("🔮 Predict Sale Price", use_container_width=True):

    # Create input dataframe
    # EXACTLY matching the features used during training
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    # =========================
    # Fill Numerical Features
    # =========================

    input_data["Overall Qual"] = overall_qual
    input_data["Year Built"] = year_built
    input_data["Year Remod/Add"] = year_remod
    input_data["Gr Liv Area"] = gr_liv_area
    input_data["Total Bsmt SF"] = total_bsmt_sf
    input_data["1st Flr SF"] = first_flr_sf
    input_data["Garage Cars"] = garage_cars
    input_data["Garage Area"] = garage_area
    input_data["Garage Yr Blt"] = garage_year
    input_data["Bedroom AbvGr"] = bedrooms
    input_data["Full Bath"] = full_bath
    input_data["Half Bath"] = half_bath

    # =========================
    # Prediction
    # =========================

    prediction = model.predict(input_data)[0]

    # =========================
    # Result
    # =========================

    st.success("Prediction completed successfully! 🎉")

    st.metric(
        label="💰 Estimated Sale Price",
        value=f"${prediction:,.0f}"
    )
