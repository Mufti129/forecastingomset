# ==========================================
#  APP PREDIKSI OMZET LOKASI USAHA
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIG UI
# ==========================================
st.set_page_config(page_title="Prediksi Omzet", layout="wide")

st.title("Prediksi Omzet Lokasi Usaha")
st.write("Simulasi prediksi omzet berdasarkan data lokasi dan lingkungan")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel("FIX_mining_prediksi_attribute_jumlah(1).xlsx")
    df.columns = df.columns.str.strip()

    # CLEANING
    df['jumlah_bangunan'] = df['jumlah_bangunan'].astype(str).str.replace(',', '').astype(float)

    # FEATURE ENGINEERING
    df['kepadatan'] = df['penduduk'] / (df['jumlah_bangunan'] + 1)
    df['kompetitor_ratio'] = df['jumlah_kompetitor'] / (df['jumlah_bangunan'] + 1)

    return df

df = load_data()

# ==========================================
# SIDEBAR (INPUT USER)
# ==========================================
st.sidebar.header("Input Data Lokasi")

def user_input():
    data = {
        'kemiskinan': st.sidebar.number_input("Kemiskinan", 0.0, 1.0, 0.08),
        'penduduk': st.sidebar.number_input("Penduduk", 0, 1000000, 120000),
        'proporsi_usia_produktif': st.sidebar.number_input("Usia Produktif", 0.0, 1.0, 0.7),
        'umk': st.sidebar.number_input("UMK", 1000000, 10000000, 5000000),
        'lebar_ruko': st.sidebar.number_input("Lebar Ruko", 50, 1000, 300),
        'jumlah_fasilitas_belanja': st.sidebar.number_input("Fasilitas Belanja", 0, 50, 10),
        'jumlah_toko_ponsel': st.sidebar.number_input("Toko Ponsel", 0, 50, 5),
        'dekat_fasilitas_transportasi_publik': st.sidebar.selectbox("Dekat Transportasi", [0,1]),
        'jumlah_pasar_tradisional': st.sidebar.number_input("Pasar Tradisional", 0, 20, 1),
        'jumlah_restoran': st.sidebar.number_input("Restoran", 0, 50, 2),
        'jumlah_kompetitor': st.sidebar.number_input("Kompetitor", 0, 50, 2),
        'dekat_layanan_keuangan': st.sidebar.selectbox("Layanan Keuangan", [0,1]),
        'jumlah_bangunan': st.sidebar.number_input("Jumlah Bangunan", 1, 50000, 15000),
        'kategori_wilayah': st.sidebar.selectbox("Wilayah", ['Perkotaan','Pedesaan']),
        'jalan': st.sidebar.selectbox("Jenis Jalan", ['residential','primary','tertiary']),
        'jarak_pasar': st.sidebar.number_input("Jarak Pasar", 0, 5000, 100)
    }

    df_input = pd.DataFrame([data])

    # FEATURE ENGINEERING
    df_input['kepadatan'] = df_input['penduduk'] / (df_input['jumlah_bangunan'] + 1)
    df_input['kompetitor_ratio'] = df_input['jumlah_kompetitor'] / (df_input['jumlah_bangunan'] + 1)

    return df_input

input_df = user_input()

# ==========================================
# MODELING
# ==========================================
features = [
    'kemiskinan','penduduk','proporsi_usia_produktif','umk','lebar_ruko',
    'jumlah_fasilitas_belanja','jumlah_toko_ponsel',
    'dekat_fasilitas_transportasi_publik','jumlah_pasar_tradisional',
    'jumlah_restoran','jumlah_kompetitor','dekat_layanan_keuangan',
    'jumlah_bangunan','kategori_wilayah','jalan','jarak_pasar',
    'kepadatan','kompetitor_ratio'
]

target = 'avg_omzet'

X = df[features]
y = df[target]

categorical = ['kategori_wilayah','jalan']
numeric = [f for f in features if f not in categorical]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
])

model.fit(X, y)

# ==========================================
# EVALUASI
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

cv = cross_val_score(model, X, y, cv=5, scoring='r2')

st.subheader("Evaluasi Model")
st.write(f"MAE: Rp {mae:,.0f}")
st.write(f"R2 Score: {r2:.4f}")
st.write(f"CV Mean R2: {cv.mean():.4f}")

# ==========================================
# PREDIKSI
# ==========================================
st.subheader("Hasil Prediksi")

prediction = model.predict(input_df)

st.success(f"Estimasi Omzet: Rp {prediction[0]:,.0f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
st.subheader("Faktor Paling Berpengaruh")

feat_names = numeric + list(
    model.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical)
)

importances = model.named_steps['rf'].feature_importances_

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

st.bar_chart(feat_imp.head(10))
