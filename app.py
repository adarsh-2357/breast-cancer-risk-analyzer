import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("svc.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Set page config
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🧬")

# Sidebar
with st.sidebar:
    st.title("👨‍💻 About")
    st.write("**Adarsh Kumar**")
    st.markdown("[🔗 GitHub](https://github.com/adarsh-2357)")
    st.markdown("---")
    st.write("This app predicts whether a breast tumor is **benign** or **malignant** using 10 selected features.")

# Title
st.title("🩺 Breast Cancer Prediction App")

# Initialize session state for reset
if "reset" not in st.session_state:
    st.session_state.reset = False

# Reset button logic
if st.button("🔁 Reset All Inputs"):
    st.session_state.reset = True

# Reset values if reset was clicked
def reset_or_default(value):
    return 0.0 if st.session_state.reset else value

# Input fields
col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input("Mean Radius", value=reset_or_default(14.0))
    mean_area = st.number_input("Mean Area", value=reset_or_default(700.0))
    mean_concavity = st.number_input("Mean Concavity", value=reset_or_default(0.1))
    worst_radius = st.number_input("Worst Radius", value=reset_or_default(17.0))
    worst_concavity = st.number_input("Worst Concavity", value=reset_or_default(0.3))

with col2:
    mean_perimeter = st.number_input("Mean Perimeter", value=reset_or_default(90.0))
    mean_concave_points = st.number_input("Mean Concave Points", value=reset_or_default(0.07))
    worst_perimeter = st.number_input("Worst Perimeter", value=reset_or_default(110.0))
    worst_area = st.number_input("Worst Area", value=reset_or_default(880.0))
    worst_concave_points = st.number_input("Worst Concave Points", value=reset_or_default(0.2))

# Collect features
features = np.array([[mean_radius, mean_perimeter, mean_area, mean_concavity, mean_concave_points,
                      worst_radius, worst_perimeter, worst_area, worst_concavity, worst_concave_points]])

# Prediction
if st.button("🔍 Predict"):
    st.session_state.reset = False

    # Warn if any input is zero
    if np.any(features == 0.0):
        st.warning("⚠️ One or more inputs are zero. Please enter valid, non-zero values for a reliable prediction.")
    else:
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][prediction]

        if prediction == 0:
            st.success(f"🎉 The tumor is **Benign**.\n🧪 Confidence: {probability * 100:.2f}%")
        else:
            st.error(f"⚠️ The tumor is **Malignant**.\n🧪 Confidence: {probability * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "<center><sub>Made with ❤️ using Streamlit</sub></center>",
    unsafe_allow_html=True
)
