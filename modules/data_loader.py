import pandas as pd

def load_data():
    df = pd.read_excel("data/data.xlsx")
    df.columns = df.columns.str.strip()

    df['jumlah_bangunan'] = df['jumlah_bangunan'].astype(str).str.replace(',', '').astype(float)

    df['kepadatan'] = df['penduduk'] / (df['jumlah_bangunan'] + 1)
    df['kompetitor_ratio'] = df['jumlah_kompetitor'] / (df['jumlah_bangunan'] + 1)

    return df
