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
from collections import Counter

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Country Clustering Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .metric-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #1f77b4; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_model_components():
    try:
        dbscan_model = joblib.load('best_dbscan_model.joblib')
        scaler = joblib.load('scaler_no_outliers.joblib')
        pca = joblib.load('pca_no_outliers.joblib')
        return dbscan_model, scaler, pca, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_and_prepare_data(file_path):
    try:
        original_df = pd.read_csv(file_path)
        countries = original_df['Country'].copy() if 'Country' in original_df.columns else None
        df_model_temp = original_df.drop(columns=['Country']) if 'Country' in original_df.columns else original_df.copy()
        
        # Numeric Cleaning
        for col in df_model_temp.columns:
            df_model_temp[col] = (df_model_temp[col].astype(str)
                                  .str.replace('%', '', regex=False)
                                  .str.replace('$', '', regex=False)
                                  .str.replace(',', '', regex=False).str.strip())
        df_model_temp = df_model_temp.apply(pd.to_numeric, errors='coerce')
        df_model_temp.fillna(df_model_temp.median(), inplace=True)
        return df_model_temp, df_model_temp.columns.tolist(), countries, original_df, None
    except Exception as e:
        return None, None, None, None, str(e)

# --- APP LAYOUT ---
st.markdown('<p class="main-header">🌍 Country Clustering Predictor</p>', unsafe_allow_html=True)

dbscan_model, scaler, pca, model_err = load_model_components()
df_model, feature_names, countries, raw_df, data_err = load_and_prepare_data("World_development_mesurement.csv")

if model_err or data_err:
    st.error("Deployment Error: Ensure your .joblib and .csv files are in the repository.")
    st.stop()

# Sidebar Settings
st.sidebar.header("🔧 Prediction Parameters")
pred_eps = st.sidebar.slider("Neighborhood Radius (eps)", 0.1, 5.0, float(dbscan_model.eps))
pred_min_samples = st.sidebar.slider("Min Samples for Cluster", 1, 10, int(dbscan_model.min_samples))

# --- TABS ---
tab1, tab2 = st.tabs(["🎯 Predict", "📊 Visualization"])

with tab1:
    st.subheader("Enter Country Metrics")
    col_a, col_b, col_c = st.columns(3)
    user_inputs = {}
    
    for idx, feat in enumerate(feature_names):
        target_col = [col_a, col_b, col_c][idx % 3]
        user_inputs[feat] = target_col.number_input(f"{feat}", value=float(df_model[feat].median()))

    if st.button("🔮 Predict Cluster", type="primary"):
        # Processing
        input_df = pd.DataFrame([user_inputs])
        scaled_input = scaler.transform(input_df)
        pca_input = pca.transform(scaled_input)
        
        # Get training PCA space
        scaled_train = scaler.transform(df_model)
        pca_train = pca.transform(scaled_train)
        train_labels = dbscan_model.fit_predict(pca_train)
        
        # Custom Prediction Logic (DBSCAN Neighbor Search)
        distances = np.linalg.norm(pca_train - pca_input, axis=1)
        nearby_mask = distances <= pred_eps
        num_neighbors = nearby_mask.sum()
        
        st.divider()
        
        if num_neighbors >= pred_min_samples:
            nearby_labels = train_labels[nearby_mask]
            valid_labels = [l for l in nearby_labels if l != -1]
            
            if valid_labels:
                prediction = Counter(valid_labels).most_common(1)[0][0]
                st.balloons()
                st.markdown(f"""<div class="metric-card"><h3>🎯 Result: Cluster {prediction}</h3>
                            This country matches the development profile of Cluster {prediction}.</div>""", unsafe_allow_html=True)
            else:
                prediction = -1
        else:
            prediction = -1

        if prediction == -1:
            st.warning("⚠️ Classified as Noise (Unique Profile)")
            nearest_idx = np.argmin(distances)
            st.info(f"Closest match: **{countries.iloc[nearest_idx]}** (Dist: {distances[nearest_idx]:.2f})")

with tab2:
    st.subheader("Cluster Distribution (PCA Space)")
    scaled_train = scaler.transform(df_model)
    pca_train = pca.transform(scaled_train)
    train_labels = dbscan_model.fit_predict(pca_train)
    
    viz_df = pd.DataFrame(pca_train[:, :2], columns=['PC1', 'PC2'])
    viz_df['Cluster'] = train_labels.astype(str)
    viz_df['Country'] = countries
    
    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', hover_data=['Country'],
                     title="Countries mapped to 2D PCA Space")
    st.plotly_chart(fig, use_container_width=True)
