import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #4facfe, #00f2fe);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    animation: fadeIn 1.5s ease-in;
}
.subtitle {
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.card {
    background-color: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    animation: fadeInUp 1s ease-in-out;
}
.stButton>button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}
.result {
    text-align: center;
    font-size: 24px;
    color: white;
    padding: 15px;
    border-radius: 10px;
    background: rgba(0,0,0,0.3);
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🤖 AI Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Predictions using your ML Model</div>", unsafe_allow_html=True)

# ---------------- INPUT MODE ----------------
option = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

# ---------------- MANUAL INPUT ----------------
if option == "Manual Input":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # 👉 CHANGE THESE FEATURES BASED ON YOUR MODEL
    f1 = st.number_input("Feature 1")
    f2 = st.number_input("Feature 2")
    f3 = st.number_input("Feature 3")

    input_data = np.array([[f1, f2, f3]])

    if st.button("🚀 Predict"):
        try:
            result = model.predict(input_data)
            st.markdown(f"<div class='result'>✅ Prediction: {result[0]}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CSV UPLOAD ----------------
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.write("Preview:", data.head())

        if st.button("📊 Predict from CSV"):
            try:
                predictions = model.predict(data)
                data["Prediction"] = predictions
                st.success("Prediction Completed ✅")
                st.write(data)

                # Download
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Results", csv, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:white;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
