import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_data
from modules.model import train_model, predict

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("Prediksi Omzet Lokasi Usaha")

# LOAD DATA
df = load_data()

# MODEL
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Linear Regression"])
model, features = train_model(df, model_choice)

# ==========================================
# INPUT USER
# ==========================================
st.sidebar.header("Input")

input_data = {
    'kemiskinan': st.sidebar.number_input("Kemiskinan", 0.0, 1.0, 0.08),
    'penduduk': st.sidebar.number_input("Penduduk", 0, 1000000, 120000),
    'proporsi_usia_produktif': st.sidebar.number_input("Usia Produktif", 0.0, 1.0, 0.7),
    'umk': st.sidebar.number_input("UMK", 1000000, 10000000, 5000000),
    'lebar_ruko': st.sidebar.number_input("Lebar Ruko", 50, 1000, 300),
    'jumlah_fasilitas_belanja': st.sidebar.number_input("Fasilitas Belanja", 0, 50, 10),
    'jumlah_toko_ponsel': st.sidebar.number_input("Toko Ponsel", 0, 50, 5),
    'dekat_fasilitas_transportasi_publik': st.sidebar.selectbox("Transportasi", [0,1]),
    'jumlah_pasar_tradisional': st.sidebar.number_input("Pasar", 0, 20, 1),
    'jumlah_restoran': st.sidebar.number_input("Restoran", 0, 50, 2),
    'jumlah_kompetitor': st.sidebar.number_input("Kompetitor", 0, 50, 2),
    'dekat_layanan_keuangan': st.sidebar.selectbox("Bank", [0,1]),
    'jumlah_bangunan': st.sidebar.number_input("Bangunan", 1, 50000, 15000),
    'kategori_wilayah': st.sidebar.selectbox("Wilayah", ['Perkotaan','Pedesaan']),
    'jalan': st.sidebar.selectbox("Jalan", ['residential','primary','tertiary']),
    'jarak_pasar': st.sidebar.number_input("Jarak Pasar", 0, 5000, 100),
    'lat': st.sidebar.number_input("Latitude", value=-6.2),
    'lon': st.sidebar.number_input("Longitude", value=106.8)
}

input_df = pd.DataFrame([input_data])

# ==========================================
# PREDIKSI
# ==========================================
pred = predict(model, input_df)

st.subheader("Hasil Prediksi")
st.success(f"Estimasi Omzet: Rp {pred[0]:,.0f}")

# ==========================================
# EVALUASI VALID
# ==========================================
X = df[features]
y = np.log1p(df['avg_omzet'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pred_log = model.predict(X_test)

pred_real = np.expm1(pred_log)
y_real = np.expm1(y_test)

mae = mean_absolute_error(y_real, pred_real)
r2 = r2_score(y_real, pred_real)

# 🔥 CV YANG BENAR
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(model, X, y, cv=kf, scoring='r2')

st.subheader("Evaluasi Model (VALID)")
st.write(f"MAE: Rp {mae:,.0f}")
st.write(f"R2: {r2:.4f}")
st.write(f"CV Mean: {cv.mean():.4f}")
st.write(f"CV Detail: {cv}")

# ==========================================
# DATA PREVIEW
# ==========================================
with st.expander("Lihat Data"):
    st.dataframe(df.head(20))
