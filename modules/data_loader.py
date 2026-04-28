import pandas as pd

def load_data():
    df = pd.read_excel("data/data.xlsx")
    df.columns = df.columns.str.strip()

    df['jumlah_bangunan'] = df['jumlah_bangunan'].astype(str).str.replace(',', '').astype(float)

    # feature engineering
    df['kepadatan'] = df['penduduk'] / (df['jumlah_bangunan'] + 1)
    df['kompetitor_ratio'] = df['jumlah_kompetitor'] / (df['jumlah_bangunan'] + 1)

    # lokasi
    df = df.dropna(subset=['lat', 'lon'])
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)

    return df
