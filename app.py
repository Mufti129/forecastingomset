import streamlit as st
import pandas as pd

from modules.data_loader import load_data
from modules.model import train_model, predict
from modules.map_utils import (
    create_map,
    add_user_marker,
    add_data_points,
    render_map,
    get_clicked_location
)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")

st.title("🗺️ Prediksi Omzet Lokasi Usaha")

# ==========================================
# LOAD DATA
# ==========================================
df = load_data()

# ==========================================
# PILIH MODEL
# ==========================================
st.sidebar.header("⚙️ Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["Random Forest", "Linear Regression"]
)

model, features, preprocessor = train_model(df, model_choice)

# ==========================================
# MAP
# ==========================================
st.subheader("🗺️ Klik Peta untuk Pilih Lokasi")

m = create_map()

# tampilkan cabang
add_data_points(m, df)

map_data = render_map(m)

lat, lon = get_clicked_location(map_data)

# marker user
add_user_marker(m, lat, lon)

st.success(f"📍 Lokasi: {lat:.6f}, {lon:.6f}")

# ==========================================
# INPUT DATA
# ==========================================
st.sidebar.header("📥 Input Data")

input_data = {
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

input_df = pd.DataFrame([input_data])

# ==========================================
# PREDIKSI
# ==========================================
prediction = predict(model, input_df)

st.subheader("💰 Hasil Prediksi")
st.success(f"Estimasi Omzet: Rp {prediction[0]:,.0f}")

# ==========================================
# EVALUASI
# ==========================================
X = df[features]
y = df['avg_omzet']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
cv = cross_val_score(model, X, y, cv=5, scoring='r2')

st.subheader("📊 Evaluasi Model")
st.write(f"Model: {model_choice}")
st.write(f"MAE: Rp {mae:,.0f}")
st.write(f"R2: {r2:.4f}")
st.write(f"CV Rata-rata: {cv.mean():.4f}")

# ==========================================
# OUTPUT MODEL
# ==========================================
st.subheader("📈 Insight Model")

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
# DATA PREVIEW
# ==========================================
with st.expander("📂 Lihat Data Cabang"):
    st.dataframe(df[['nama_cabang','lat','lon','avg_omzet']].head(20))
