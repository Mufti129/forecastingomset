from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_model(df, model_choice):

    features = [
        'kemiskinan','penduduk','proporsi_usia_produktif','umk','lebar_ruko',
        'jumlah_fasilitas_belanja','jumlah_toko_ponsel',
        'dekat_fasilitas_transportasi_publik','jumlah_pasar_tradisional',
        'jumlah_restoran','jumlah_kompetitor','dekat_layanan_keuangan',
        'jumlah_bangunan','kategori_wilayah','jalan','jarak_pasar',
        'kepadatan','kompetitor_ratio'
    ]

    X = df[features]
    y = df['avg_omzet']

    categorical = ['kategori_wilayah','jalan']
    numeric = [f for f in features if f not in categorical]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = LinearRegression()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)

    return pipeline, features, preprocessor


def predict(model, input_df):
    input_df['kepadatan'] = input_df['penduduk'] / (input_df['jumlah_bangunan'] + 1)
    input_df['kompetitor_ratio'] = input_df['jumlah_kompetitor'] / (input_df['jumlah_bangunan'] + 1)

    return model.predict(input_df)
