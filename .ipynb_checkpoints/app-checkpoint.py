import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
rf = pickle.load(open("random_forest.pkl", "rb"))

st.set_page_config(page_title="Random Forest Prediction")
st.title("Random Forest Movie Revenue Prediction")

st.write("""
Model ini dibuat menggunakan **Random Forest Regression**
untuk memprediksi pendapatan kotor (Gross) film berdasarkan fitur-fitur penting.
""")

# Input user
budget = st.number_input("Budget (USD)", min_value=0.0, value=5000000.0)
votes = st.number_input("Votes IMDb", min_value=0.0, value=2000.0)
rating = st.slider("Rating IMDb", 0.0, 10.0, 7.0)
duration = st.number_input("Duration (minutes)", min_value=40, value=120)
year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2020)

# Preprocessing input (sesuai dataset)
budget_log = np.log1p(budget)
votes_log = np.log1p(votes)
profit = budget * 0.7
profit_log = np.sign(profit) * np.log1p(abs(profit))

# Buat dataframe untuk prediksi
data_rf = pd.DataFrame({
    "budget_log": [budget_log],
    "votes_log": [votes_log],
    "rating": [rating],
    "duration": [duration],
    "year": [year],
    "profit_log": [profit_log]
})

st.write("### Data yang diprediksi:")
st.write(data_rf)

# Prediksi
if st.button("Prediksi Gross (Log Scale)"):
    pred = rf.predict(data_rf)[0]
    st.success(f"Prediksi Gross Log: {pred:.4f}")
    gross_dollar = np.expm1(pred)
    st.info(f"Estimasi pendapatan kotor: **${gross_dollar:,.0f}**")