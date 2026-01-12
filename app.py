import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN, KMeans

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Country Clustering", layout="wide")

st.title("🌍 Country Clustering Predictor (FIXED)")

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler_no_outliers.joblib")
    pca = joblib.load("pca_no_outliers.joblib")
    model = joblib.load("best_dbscan_model.joblib")
    return scaler, pca, model

scaler, pca, cluster_model = load_models()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("World_development_mesurement.csv")
    countries = df["Country"]
    X = df.drop(columns=["Country"])

    X = X.apply(
        lambda col: (
            col.astype(str)
               .str.replace("%", "")
               .str.replace("$", "")
               .str.replace(",", "")
        )
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X.fillna(X.median(), inplace=True)

    return X, countries

X, countries = load_data()

# --------------------------------------------------
# PCA TRANSFORMATION (ONCE)
# --------------------------------------------------
pca_data = pca.transform(scaler.transform(X))

# --------------------------------------------------
# FIT CLUSTER MODEL ONCE (CRITICAL FIX)
# --------------------------------------------------
if "clusters_train" not in st.session_state:
    if isinstance(cluster_model, DBSCAN):
        clusters = cluster_model.fit_predict(pca_data)
        st.session_state["model_type"] = "dbscan"
    else:
        clusters = cluster_model.fit_predict(pca_data)
        st.session_state["model_type"] = "kmeans"

    st.session_state["clusters_train"] = clusters
    st.session_state["pca_data"] = pca_data

clusters_train = st.session_state["clusters_train"]
pca_data = st.session_state["pca_data"]

# --------------------------------------------------
# SHOW CLUSTER SUMMARY
# --------------------------------------------------
st.subheader("📊 Cluster Summary")

unique, counts = np.unique(clusters_train, return_counts=True)
summary = pd.DataFrame({"Cluster": unique, "Count": counts})
st.dataframe(summary, use_container_width=True)

n_clusters = len(unique[unique != -1])
st.info(f"Total clusters (excluding noise): {n_clusters}")

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------
st.subheader("🎯 Predict Cluster for New Country")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(
        col, value=float(X[col].median())
    )

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    pca_input = pca.transform(scaler.transform(input_df))

    distances = np.linalg.norm(pca_data - pca_input, axis=1)

    eps = st.slider("Prediction eps", 0.2, 1.5, 0.5, 0.05)
    min_samples = st.slider("Min neighbors", 1, 10, 3)

    neighbors = distances <= eps

    if neighbors.sum() >= min_samples:
        neighbor_clusters = clusters_train[neighbors]
        valid = neighbor_clusters[neighbor_clusters != -1]

        if len(valid) > 0:
            from collections import Counter
            prediction = Counter(valid).most_common(1)[0][0]
            st.success(f"✅ Assigned to Cluster {prediction}")
        else:
            st.warning("⚠️ Only noise neighbors → classified as Noise")
    else:
        st.warning("⚠️ Not enough neighbors → Noise")

    nearest_idx = np.argmin(distances)
    st.write(
        f"Nearest country: **{countries.iloc[nearest_idx]}**, "
        f"distance = {distances[nearest_idx]:.3f}"
    )

# --------------------------------------------------
# COUNTRY EXPLORER
# --------------------------------------------------
st.subheader("🌍 Countries by Cluster")

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(unique),
    format_func=lambda x: "Noise" if x == -1 else f"Cluster {x}",
)

mask = clusters_train == selected_cluster
st.write(countries[mask].tolist())
