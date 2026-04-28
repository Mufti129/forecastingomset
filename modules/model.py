from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def train_model(df, model_choice):

    features = [
        'kemiskinan','penduduk','proporsi_usia_produktif','umk','lebar_ruko',
        'jumlah_fasilitas_belanja','jumlah_toko_ponsel',
        'dekat_fasilitas_transportasi_publik','jumlah_pasar_tradisional',
        'jumlah_restoran','jumlah_kompetitor','dekat_layanan_keuangan',
        'jumlah_bangunan','kategori_wilayah','jalan','jarak_pasar',
        'kepadatan','kompetitor_ratio','fasilitas_total',
        'lat','lon'
    ]

    X = df[features]

    # 🔥 LOG TRANSFORM (WAJIB)
    y = np.log1p(df['avg_omzet'])

    categorical = ['kategori_wilayah','jalan']
    numeric = [f for f in features if f not in categorical]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    if model_choice == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,              # 🔥 diturunin biar ga overfit
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
    else:
        model = LinearRegression()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)

    return pipeline, features


# ==========================================
# PREDICT
# ==========================================
def predict(model, input_df):

    input_df['kepadatan'] = input_df['penduduk'] / (input_df['jumlah_bangunan'] + 1)
    input_df['kompetitor_ratio'] = input_df['jumlah_kompetitor'] / (input_df['jumlah_bangunan'] + 1)
    input_df['fasilitas_total'] = input_df['jumlah_fasilitas_belanja'] + input_df['jumlah_restoran']

    pred_log = model.predict(input_df)

    return np.expm1(pred_log)
