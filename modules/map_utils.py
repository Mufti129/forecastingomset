import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

def create_map(center_lat=-6.2, center_lon=106.8, zoom=10):
    return folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

def add_user_marker(m, lat, lon):
    folium.Marker(
        location=[lat, lon],
        popup="Lokasi Anda",
        icon=folium.Icon(color="green")
    ).add_to(m)

def add_data_points(m, df):
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        color = "green" if row.get('avg_omzet', 0) > 600_000_000 else "red"

        folium.Marker(
            location=[row['lat'], row['lon']],
            tooltip=row.get('nama_cabang', 'Cabang'),
            popup=f"""
            <b>{row.get('nama_cabang', 'Cabang')}</b><br>
            Omzet: Rp {row.get('avg_omzet', 0):,.0f}<br>
            Kompetitor: {row.get('jumlah_kompetitor', '-')}
            """,
            icon=folium.Icon(color=color)
        ).add_to(cluster)

def render_map(m):
    return st_folium(m, width=700, height=500)

def get_clicked_location(map_data, default_lat=-6.2, default_lon=106.8):
    if map_data and map_data.get("last_clicked"):
        return map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    return default_lat, default_lon
