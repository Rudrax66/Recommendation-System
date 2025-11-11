import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 🎬 Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="🎥 Movie Rating Predictor", layout="centered")
st.title("🎬 Movie IMDb Rating Prediction App")

# -----------------------------
# 📂 Load Dataset
# -----------------------------
df = pd.read_csv("movies.csv")

# Check if target column exists
if 'IMDB_Rating' not in df.columns:
    st.error("❌ Column 'IMDB_Rating' not found in dataset.")
    st.stop()

# Drop missing values
df = df.dropna()

# Separate features and target
X = df.drop('IMDB_Rating', axis=1)
y = df['IMDB_Rating']

# -----------------------------
# ⚙️ Encode categorical variables
# -----------------------------
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -----------------------------
# 🧠 Train model (or load if exists)
# -----------------------------
model_filename = "random_forest_movie.pkl"

try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.warning("⚠️ Model not found — training a new Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

# -----------------------------
# 🎛️ User Inputs
# -----------------------------
st.subheader("🎞️ Enter Movie Details")

user_input = {}

# Numeric inputs
for col in num_cols:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Categorical inputs
for col in cat_cols:
    options = sorted(df[col].dropna().unique().tolist())
    user_input[col] = st.selectbox(f"{col}", options)

# Encode user input
encoded_input = {}
for col in X.columns:
    if col in cat_cols:
        le = label_encoders[col]
        encoded_input[col] = le.transform([user_input[col]])[0] if user_input[col] in le.classes_ else 0
    else:
        encoded_input[col] = user_input[col]

# Create DataFrame with same feature order
input_df = pd.DataFrame([encoded_input])[X.columns]

# -----------------------------
# 🔮 Predict Button
# -----------------------------
if st.button("⭐ Predict IMDb Rating"):
    prediction = model.predict(input_df)
    st.success(f"🎯 Estimated IMDb Rating: **{prediction[0]:.2f} / 10**")

st.markdown("---")
st.caption("Developed by Rudrax • Powered by Streamlit & Random Forest Regressor")
