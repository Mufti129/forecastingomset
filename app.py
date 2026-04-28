import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_data
from modules.model import train_model, predict

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(layout="wide")
st.title("🗺️ Dashboard Prediksi Omzet & Analisis Lokasi")

# ==========================================
# LOAD DATA
# ==========================================
df = load_data()

# ==========================================
# MODEL
# ==========================================
st.sidebar.header("⚙️ Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Random Forest", "Linear Regression"]
)

model, features = train_model(df, model_choice)

# ==========================================
# LAYOUT
# ==========================================
col1, col2 = st.columns([2,1])

# ==========================================
# 🗺️ MAP + HEATMAP + KOMPETITOR
# ==========================================
with col1:
    st.subheader("🗺️ Peta Lokasi & Analisis Kompetitor")

    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # 🔥 HEATMAP OMZET
    heat_data = [
        [row['lat'], row['lon'], row['avg_omzet']]
        for _, row in df.iterrows()
    ]
    HeatMap(heat_data, radius=15).add_to(m)

    # 🔵 MARKER CABANG
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            popup=f"""
            <b>{row.get('nama_cabang','Cabang')}</b><br>
            Omzet: Rp {row['avg_omzet']:,.0f}
            """,
            color='blue',
            fill=True
        ).add_to(m)

    # tampilkan map
    map_data = st_folium(m, width=800, height=500)

    clicked_lat, clicked_lon = None, None

    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]

        st.success(f"📍 Lokasi dipilih: {clicked_lat:.6f}, {clicked_lon:.6f}")

    # ==========================================
    # 🔥 ANALISIS KOMPETITOR TERDEKAT
    # ==========================================
    if clicked_lat is not None:

        # cari cabang terdekat dari titik klik
        df['distance_click'] = (
            (df['lat'] - clicked_lat)**2 + (df['lon'] - clicked_lon)**2
        )

        nearest = df.loc[df['distance_click'].idxmin()]

        cabang_lat = nearest['lat']
        cabang_lon = nearest['lon']

        komp_lat = nearest['lat_komp']
        komp_lon = nearest['lon_komp']

        # marker cabang
        folium.Marker(
            [cabang_lat, cabang_lon],
            popup="Cabang Terpilih",
            icon=folium.Icon(color="blue")
        ).add_to(m)

        # marker kompetitor
        folium.Marker(
            [komp_lat, komp_lon],
            popup="Kompetitor Terdekat",
            icon=folium.Icon(color="red")
        ).add_to(m)

        # garis penghubung
        folium.PolyLine(
            locations=[[cabang_lat, cabang_lon], [komp_lat, komp_lon]],
            color="red",
            weight=3
        ).add_to(m)

        # hitung jarak sederhana
        distance = np.sqrt(
            (cabang_lat - komp_lat)**2 + (cabang_lon - komp_lon)**2
        )

        st.info(f"📏 Jarak ke kompetitor: {distance:.4f} (derajat koordinat)")

        # render ulang map
        st_folium(m, width=800, height=500)

    else:
        clicked_lat, clicked_lon = -6.2, 106.8

# ==========================================
# 📥 INPUT + PREDIKSI
# ==========================================
with col2:
    st.subheader("📥 Input Data Lokasi")

    input_data = {
        'kemiskinan': st.number_input("Kemiskinan", 0.0, 1.0, 0.08),
        'penduduk': st.number_input("Penduduk", 0, 1000000, 120000),
        'proporsi_usia_produktif': st.number_input("Usia Produktif", 0.0, 1.0, 0.7),
        'umk': st.number_input("UMK", 1000000, 10000000, 5000000),
        'lebar_ruko': st.number_input("Lebar Ruko", 50, 1000, 300),
        'jumlah_fasilitas_belanja': st.number_input("Fasilitas Belanja", 0, 50, 10),
        'jumlah_toko_ponsel': st.number_input("Toko Ponsel", 0, 50, 5),
        'dekat_fasilitas_transportasi_publik': st.selectbox("Transportasi", [0,1]),
        'jumlah_pasar_tradisional': st.number_input("Pasar", 0, 20, 1),
        'jumlah_restoran': st.number_input("Restoran", 0, 50, 2),
        'jumlah_kompetitor': st.number_input("Kompetitor", 0, 50, 2),
        'dekat_layanan_keuangan': st.selectbox("Bank", [0,1]),
        'jumlah_bangunan': st.number_input("Bangunan", 1, 50000, 15000),
        'kategori_wilayah': st.selectbox("Wilayah", ['Perkotaan','Pedesaan']),
        'jalan': st.selectbox("Jalan", ['residential','primary','tertiary']),
        'jarak_pasar': st.number_input("Jarak Pasar", 0, 5000, 100),
        'lat': clicked_lat,
        'lon': clicked_lon
    }

    input_df = pd.DataFrame([input_data])

    pred = predict(model, input_df)

    st.subheader("💰 Prediksi Omzet")
    st.success(f"Rp {pred[0]:,.0f}")

# ==========================================
# 📊 EVALUASI MODEL
# ==========================================
st.subheader("📊 Evaluasi Model")

X = df[features]
y = np.log1p(df['avg_omzet'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_val_score(model, X, y, cv=kf, scoring='r2')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pred_log = model.predict(X_test)
pred_real = np.expm1(pred_log)
y_real = np.expm1(y_test)

mae = mean_absolute_error(y_real, pred_real)
r2 = r2_score(y_real, pred_real)

colA, colB, colC = st.columns(3)
colA.metric("MAE", f"Rp {mae:,.0f}")
colB.metric("R2", f"{r2:.3f}")
colC.metric("CV Score", f"{cv.mean():.3f}")

# ==========================================
# 📈 FEATURE IMPORTANCE
# ==========================================
st.subheader("📈 Faktor Penting")

categorical = ['kategori_wilayah','jalan']
numeric = [f for f in features if f not in categorical]

feat_names = numeric + list(
    model.named_steps['prep']
    .named_transformers_['cat']
    .get_feature_names_out(categorical)
)

if model_choice == "Random Forest":
    importances = model.named_steps['model'].feature_importances_
    st.bar_chart(pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10))
else:
    coef = model.named_steps['model'].coef_
    st.bar_chart(pd.Series(coef, index=feat_names).sort_values(ascending=False).head(10))

# ==========================================
# DATA
# ==========================================
with st.expander("📂 Lihat Data"):
    st.dataframe(df.head(50))
