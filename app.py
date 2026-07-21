import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =============================================================================
# 1. KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Analisis Cacat Produk Manufaktur",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Analisis Clustering Cacat Produk Industri Manufaktur")
st.markdown("Aplikasi interaktif untuk analisis segmentasi cacat produk berbasis **K-Means Clustering** beserta evaluasi dampak bisnisnya.")
st.divider()

# =============================================================================
# 2. LOAD DATASET DARI FILE CSV (`defects_data.csv`)
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('defects_data.csv')
    except FileNotFoundError:
        st.error("File 'defects_data.csv' tidak ditemukan di folder proyek!")
        st.stop()
    return df

df_raw = load_data()
df = df_raw.copy()

# =============================================================================
# 3. FEATURE ENGINEERING
#    Kolom defect_id & product_id adalah ID -> tidak boleh ikut jadi fitur.
#    'severity' di-encode ordinal (Minor < Moderate < Critical) karena ini
#    inti dari "tingkat keparahan cacat" yang ingin disegmentasi.
# =============================================================================
severity_map = {"Minor": 1, "Moderate": 2, "Critical": 3}

if "severity" not in df.columns or "repair_cost" not in df.columns:
    st.error("Dataset harus memiliki kolom 'severity' dan 'repair_cost' untuk analisis ini.")
    st.stop()

df["severity_score"] = df["severity"].map(severity_map)

if df["severity_score"].isnull().any():
    st.warning("Ada nilai 'severity' yang tidak dikenali dan diabaikan dari analisis.")
    df = df.dropna(subset=["severity_score"])

feature_cols = ["severity_score", "repair_cost"]

# =============================================================================
# 4. SIDEBAR - PARAMETER MODEL
# =============================================================================
st.sidebar.header("⚙️ Parameter Model")

k_clusters = st.sidebar.slider(
    "Pilih Jumlah Cluster (K):",
    min_value=2,
    max_value=5,
    value=3,
    help="Pilih jumlah kelompok/cluster."
)

st.sidebar.markdown("**Fitur yang digunakan untuk clustering:**")
st.sidebar.markdown("- `severity_score` (Minor=1, Moderate=2, Critical=3)\n- `repair_cost`")
st.sidebar.caption("Kolom `defect_id` dan `product_id` sengaja tidak diikutkan karena hanya identitas, bukan karakteristik cacat.")

# =============================================================================
# 5. STANDARDISASI + K-MEANS CLUSTERING
# =============================================================================
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, df["Cluster"])

# Urutkan label cluster berdasarkan tingkat risiko (kombinasi severity & repair_cost)
risk_rank = (
    df.groupby("Cluster")[feature_cols]
    .mean()
    .apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-9))
    .mean(axis=1)
    .sort_values()
    .index
)
cluster_mapping = {old_label: f"Cluster {new_idx + 1}" for new_idx, old_label in enumerate(risk_rank)}
df["Cluster_Label"] = df["Cluster"].map(cluster_mapping)

cluster_order_labels = [cluster_mapping[c] for c in risk_rank]

# =============================================================================
# 6. METRICS / DASHBOARD ATAS
# =============================================================================
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Total Sampel / Data", f"{len(df)} Records")
with col_m2:
    st.metric("Jumlah Cluster (K)", f"{k_clusters}")
with col_m3:
    st.metric("Silhouette Score", f"{sil_score:.3f}", help="Mendekati 1 = cluster terpisah dengan baik. Mendekati 0 = cluster tumpang tindih.")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 7. TAB UTAMA
# =============================================================================
tab_visual, tab_insight, tab_data = st.tabs([
    "📊 Hasil Clustering",
    "💡 Interpretasi & Insight Bisnis",
    "📑 Data Mentah"
])

# --- TAB 1: HASIL CLUSTERING ---
with tab_visual:
    col_chart, col_stat = st.columns([2, 1])

    with col_chart:
        st.subheader("Sebaran Cluster: Severity vs Biaya Perbaikan")

        plot_df = df.copy()
        # jitter agar titik severity (nilai diskrit) tidak saling menumpuk
        rng = np.random.default_rng(42)
        plot_df["severity_jitter"] = plot_df["severity_score"] + rng.uniform(-0.15, 0.15, len(plot_df))

        fig = px.scatter(
            plot_df,
            x="severity_jitter",
            y="repair_cost",
            color="Cluster_Label",
            category_orders={"Cluster_Label": cluster_order_labels},
            hover_data=["defect_type", "defect_location", "severity"],
            title="Sebaran Data: Severity vs Repair Cost",
            template="plotly_white",
            height=450,
            labels={"severity_jitter": "Severity (1=Minor, 2=Moderate, 3=Critical)", "repair_cost": "Repair Cost"}
        )
        fig.update_xaxes(tickvals=[1, 2, 3], ticktext=["Minor", "Moderate", "Critical"])
        st.plotly_chart(fig, use_container_width=True)

    with col_stat:
        st.subheader("Rata-Rata Fitur per Cluster")
        summary_df = (
            df.groupby("Cluster_Label")[feature_cols]
            .mean()
            .round(2)
            .reindex(cluster_order_labels)
        )
        summary_df["Jumlah Data"] = df["Cluster_Label"].value_counts().reindex(cluster_order_labels)
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Distribusi Tipe Cacat Dominan")
        dominant_type = (
            df.groupby("Cluster_Label")["defect_type"]
            .agg(lambda s: s.mode().iloc[0])
            .reindex(cluster_order_labels)
        )
        st.dataframe(dominant_type.rename("Tipe Cacat Dominan"), use_container_width=True)

# --- TAB 2: INTERPRETASI & INSIGHT BISNIS (dinamis mengikuti K) ---
with tab_insight:
    st.subheader("Interpretasi Model & Recommendation Strategy")
    st.caption("Cluster diurutkan dari yang paling ringan (repair cost & severity rendah) ke paling kritis.")

    n = len(cluster_order_labels)
    style_map = []
    for i in range(n):
        pos = i / max(n - 1, 1)  # 0 = paling aman, 1 = paling kritis
        if pos < 1 / 3:
            style_map.append(("🟢", "Kategori Aman / Ringan", st.success))
        elif pos < 2 / 3:
            style_map.append(("🟡", "Kategori Waspada / Sedang", st.warning))
        else:
            style_map.append(("🔴", "Kategori Kritis / Berat", st.error))

    cols = st.columns(n)
    for i, cluster_label in enumerate(cluster_order_labels):
        sub = df[df["Cluster_Label"] == cluster_label]
        emoji, kategori, box_fn = style_map[i]
        avg_cost = sub["repair_cost"].mean()
        avg_sev = sub["severity_score"].mean()
        pct = len(sub) / len(df) * 100
        top_type = sub["defect_type"].mode().iloc[0]
        top_loc = sub["defect_location"].mode().iloc[0]

        with cols[i]:
            box_fn(f"{emoji} **{cluster_label} ({kategori})**")
            st.markdown(f"""
            * **Jumlah data:** {len(sub)} ({pct:.1f}% dari total)
            * **Rata-rata repair cost:** {avg_cost:,.2f}
            * **Rata-rata severity score:** {avg_sev:.2f}
            * **Tipe cacat dominan:** {top_type}
            * **Lokasi cacat dominan:** {top_loc}
            """)
            if i / max(n - 1, 1) < 1 / 3:
                st.markdown("**Rekomendasi:** Pertahankan SOP dan lakukan *preventive maintenance* rutin.")
            elif i / max(n - 1, 1) < 2 / 3:
                st.markdown("**Rekomendasi:** Kalibrasi ulang instrumen dan tingkatkan inspeksi *Quality Control* (QC).")
            else:
                st.markdown("**Rekomendasi:** Evaluasi prioritas tinggi — jalankan *Root Cause Analysis* (RCA) dan pertimbangkan *overhaul* mesin.")

# --- TAB 3: DATA MENTAH ---
with tab_data:
    st.subheader("Tabel Data Produk")
    st.dataframe(df, use_container_width=True)
