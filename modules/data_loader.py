import pandas as pd
import numpy as np

def load_data():
    df = pd.read_excel("data/data.xlsx")
    df.columns = df.columns.str.strip()

    # CLEANING
    df['jumlah_bangunan'] = df['jumlah_bangunan'].astype(str).str.replace(',', '').astype(float)

    # ==========================================
    # FEATURE ENGINEERING (AMAN & TIDAK LEAKAGE)
    # ==========================================
    df['kepadatan'] = df['penduduk'] / (df['jumlah_bangunan'] + 1)
    df['kompetitor_ratio'] = df['jumlah_kompetitor'] / (df['jumlah_bangunan'] + 1)

    # ⚠️ fitur bisnis sederhana (hindari terlalu kompleks dulu)
    df['fasilitas_total'] = df['jumlah_fasilitas_belanja'] + df['jumlah_restoran']

    # ==========================================
    # HANDLE OUTLIER
    # ==========================================
    df = df[df['avg_omzet'] < df['avg_omzet'].quantile(0.99)]

    # ==========================================
    # LOKASI
    # ==========================================
    df = df.dropna(subset=['lat', 'lon'])
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)

    return df
