import streamlit as st
import numpy as np
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Analyzer",
    page_icon="🍷",
    layout="wide"
)

# --------------------------------------------------
# HIGH CONTRAST CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color: #ffffff;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
    border-right: 6px solid #6d28d9;
}

/* SIDEBAR LABELS (VERY CLEAR) */
section[data-testid="stSidebar"] label {
    color: #f9fafb !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* SIDEBAR HEADER */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
}

/* INPUT BOX */
input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 3px solid #14b8a6 !important;
    border-radius: 8px !important;
    font-weight: 600;
}

/* MAIN HEADINGS */
h1, h2, h3 {
    color: #6d28d9;
    font-weight: 800;
}

/* BUTTON */
.stButton > button {
    background-color: #14b8a6;
    color: #042f2e;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 28px;
    border: 3px solid #0f766e;
}

.stButton > button:hover {
    background-color: #0f766e;
    color: white;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 4px solid #6d28d9;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL & SCALER
# --------------------------------------------------
scaler = pickle.load(open("scaler_model.pkl", "rb"))
model = pickle.load(open("finalized_RFmodel.sav", "rb"))

expected_features = scaler.feature_names_in_

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🍷 Wine Quality Classification System")
st.write("High-contrast dashboard for clear feature visibility")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🍇 Chemical Features")

def user_input():
    values = {}
    for f in expected_features:
        values[f] = st.sidebar.number_input(
            label=f.replace("_", " ").upper(),
            min_value=0.0,
            value=1.0,
            step=0.1
        )
    return pd.DataFrame([values])

input_df = user_input()

# --------------------------------------------------
# SHOW INPUT DATA
# --------------------------------------------------
st.subheader("📊 Entered Feature Values")
st.dataframe(input_df, use_container_width=True)

# --------------------------------------------------
# SCALE DATA
# --------------------------------------------------
scaled_data = scaler.transform(input_df)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🔍 ANALYZE WINE QUALITY"):
    prediction = model.predict(scaled_data)
    quality = int(np.round(prediction[0]))

    st.success(f"🍷 Wine Quality Score: {quality} / 10")
    st.progress(quality / 10)

    if quality >= 7:
        st.markdown("### 🟢 HIGH QUALITY WINE")
    elif quality >= 5:
        st.markdown("### 🟡 MEDIUM QUALITY WINE")
    else:
        st.markdown("### 🔴 LOW QUALITY WINE")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><b>ML Wine Quality Prediction Project</b></center>",
    unsafe_allow_html=True
)
