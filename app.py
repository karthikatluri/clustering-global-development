import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Country Clustering Predictor",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
@st.cache_resource
def load_models():
    dbscan = joblib.load("best_dbscan_model.joblib")
    scaler = joblib.load("scaler_no_outliers.joblib")
    pca = joblib.load("pca_no_outliers.joblib")
    return dbscan, scaler, pca


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    countries = None
    if "Country" in df.columns:
        countries = df["Country"].copy()
        df = df.drop(columns=["Country"])

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "")
            .str.replace("$", "")
            .str.replace(",", "")
            .str.strip()
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(df.median(), inplace=True)

    return df, df.columns.tolist(), countries


@st.cache_data
def get_pca_training_data(scaler, pca, df):
    scaled = scaler.transform(df)
    return pca.transform(scaled)


def get_active_model(default_model):
    if "retrained_model" in st.session_state:
        return (
            st.session_state["retrained_model"],
            st.session_state.get("model_type", "dbscan"),
        )
    return default_model, "dbscan"


# --------------------------------------------------
# LOAD MODELS & DATA
# --------------------------------------------------
dbscan_model, scaler, pca = load_models()
df_model, feature_names, countries = load_data(
    "World_development_mesurement.csv"
)

pca_data_train = get_pca_training_data(scaler, pca, df_model)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌍 Country Clustering Prediction System")
st.caption("PCA + DBSCAN / K-Means | Streamlit ML App")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Prediction Settings")

use_custom_eps = st.sidebar.checkbox("Use custom eps (DBSCAN)", False)
if use_custom_eps:
    st.session_state["prediction_eps"] = st.sidebar.slider(
        "eps",
        0.1,
        5.0,
        float(dbscan_model.eps),
        0.1,
    )
else:
    st.session_state["prediction_eps"] = dbscan_model.eps

st.session_state["prediction_min_samples"] = st.sidebar.slider(
    "Min neighbors",
    1,
    10,
    max(1, dbscan_model.min_samples - 2),
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.subheader("Enter Country Development Metrics")

user_inputs = {}
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        user_inputs[feature] = st.number_input(
            feature,
            value=float(df_model[feature].median()),
            min_value=float(df_model[feature].min() * 0.1),
            max_value=float(df_model[feature].max() * 2),
            format="%.4f",
        )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()
predict_btn = st.button("🔮 Predict Cluster", type="primary")

if predict_btn:
    try:
        input_df = pd.DataFrame([user_inputs])
        scaled_input = scaler.transform(input_df)
        pca_input = pca.transform(scaled_input)

        model, model_type = get_active_model(dbscan_model)
        clusters_train = model.labels_

        distances = np.linalg.norm(pca_data_train - pca_input, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]

        # ---------------- K-MEANS ----------------
        if model_type == "kmeans":
            prediction = model.predict(pca_input)[0]

        # ---------------- DBSCAN ----------------
        else:
            eps = st.session_state["prediction_eps"]
            min_samples = st.session_state["prediction_min_samples"]

            neighbors = distances <= eps
            if neighbors.sum() >= min_samples:
                neighbor_clusters = clusters_train[neighbors]
                valid = neighbor_clusters[neighbor_clusters != -1]

                if len(valid) > 0:
                    from collections import Counter

                    prediction = Counter(valid).most_common(1)[0][0]
                else:
                    prediction = -1
            else:
                prediction = -1

        confidence = max(0, min(100, (1 - nearest_distance / 5) * 100))

        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------
        st.success("✅ Prediction Complete")

        c1, c2, c3 = st.columns(3)

        with c1:
            if prediction == -1:
                st.error("⚠️ Noise / Outlier")
            else:
                st.success(f"🎯 Cluster {prediction}")

        with c2:
            st.metric("Nearest Distance", f"{nearest_distance:.4f}")

        with c3:
            st.metric("Confidence", f"{confidence:.1f}%")

        # --------------------------------------------------
        # PCA VISUALIZATION
        # --------------------------------------------------
        viz_df = pd.DataFrame({
            "PC1": pca_data_train[:, 0],
            "PC2": pca_data_train[:, 1],
            "Cluster": clusters_train.astype(str),
            "Type": "Training"
        })

        pred_df = pd.DataFrame({
            "PC1": [pca_input[0, 0]],
            "PC2": [pca_input[0, 1]],
            "Cluster": ["Prediction"],
            "Type": ["Prediction"]
        })

        viz_df = pd.concat([viz_df, pred_df])

        fig = px.scatter(
            viz_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            symbol="Type",
            title="Prediction in PCA Space",
            height=500
        )

        fig.update_traces(marker=dict(size=10))
        fig.data[-1].marker.size = 18

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------
        # SIMILAR COUNTRIES
        # --------------------------------------------------
        if countries is not None and prediction != -1:
            same_cluster = clusters_train == prediction
            indices = np.where(same_cluster)[0]
            nearest = indices[np.argsort(distances[indices])[:10]]

            st.subheader("🌍 Similar Countries")

            st.dataframe(
                pd.DataFrame({
                    "Country": countries.iloc[nearest].values,
                    "Distance": distances[nearest].round(4),
                }),
                use_container_width=True,
            )

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "🌍 Country Clustering Prediction System | PCA + DBSCAN / K-Means | Streamlit"
)
