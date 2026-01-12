import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from collections import Counter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Development Predictor", page_icon="🌍", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_assets():
    try:
        dbscan = joblib.load('best_dbscan_model.joblib')
        scaler = joblib.load('scaler_no_outliers.joblib')
        pca = joblib.load('pca_no_outliers.joblib')
        return dbscan, scaler, pca
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Clean non-numeric characters for processing
    df_clean = df.copy()
    cols_to_fix = df_clean.columns.drop('Country') if 'Country' in df_clean.columns else df_clean.columns
    for col in cols_to_fix:
        df_clean[col] = (df_clean[col].astype(str).str.replace(r'[%$,]', '', regex=True).str.strip())
    df_clean[cols_to_fix] = df_clean[cols_to_fix].apply(pd.to_numeric, errors='coerce')
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    return df, df_clean

# --- INITIALIZE ---
dbscan_model, scaler, pca = load_assets()
raw_df, df_numeric = load_data("World_development_mesurement.csv")

if not dbscan_model: st.stop()

# --- SIDEBAR: COUNTRY SELECTION ---
st.sidebar.header("📍 Data Selection")
selection_mode = st.sidebar.radio("Input Method", ["Manual Entry", "Select Existing Country"])

selected_country_data = None
if selection_mode == "Select Existing Country":
    country_list = sorted(raw_df['Country'].unique())
    choice = st.sidebar.selectbox("Choose a Country", country_list)
    selected_country_data = df_numeric[raw_df['Country'] == choice].iloc[0]
    st.sidebar.success(f"Loaded data for {choice}")

# --- MAIN INTERFACE ---
st.markdown('<p class="main-header">🌍 Country Clustering & Development Explorer</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction Engine", "🔎 Cluster Explorer", "📈 Visual Analysis"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.subheader("Predict Cluster Membership")
    features = df_numeric.columns.drop('Country') if 'Country' in df_numeric.columns else df_numeric.columns
    
    # Create input fields
    cols = st.columns(3)
    user_inputs = {}
    for i, feat in enumerate(features):
        default_val = float(selected_country_data[feat]) if selected_country_data is not None else float(df_numeric[feat].median())
        user_inputs[feat] = cols[i % 3].number_input(f"{feat}", value=default_val)

    if st.button("🔮 Run Analysis", type="primary"):
        input_df = pd.DataFrame([user_inputs])
        pca_input = pca.transform(scaler.transform(input_df))
        
        # DBSCAN Logic: Find neighbors in PCA space
        pca_train = pca.transform(scaler.transform(df_numeric.drop(columns=['Country'] if 'Country' in df_numeric.columns else [])))
        train_labels = dbscan_model.fit_predict(pca_train)
        
        distances = np.linalg.norm(pca_train - pca_input, axis=1)
        nearby_indices = np.where(distances <= dbscan_model.eps)[0]
        
        if len(nearby_indices) >= dbscan_model.min_samples:
            votes = [train_labels[i] for i in nearby_indices if train_labels[i] != -1]
            if votes:
                res = Counter(votes).most_common(1)[0][0]
                st.success(f"### Result: This profile belongs to **Cluster {res}**")
            else:
                st.warning("### Result: Classified as **Noise** (Unique Profile)")
        else:
            st.warning("### Result: Classified as **Noise** (Outlier)")

# --- TAB 2: CLUSTER EXPLORER ---
with tab2:
    st.subheader("Browse Countries by Cluster")
    # Pre-calculate clusters for the whole dataset
    pca_all = pca.transform(scaler.transform(df_numeric.drop(columns=['Country'] if 'Country' in df_numeric.columns else [])))
    labels = dbscan_model.fit_predict(pca_all)
    explorer_df = raw_df.copy()
    explorer_df['Cluster'] = labels
    
    cluster_ids = sorted(explorer_df['Cluster'].unique())
    cluster_choice = st.selectbox("Select Cluster to View", cluster_ids, format_func=lambda x: f"Cluster {x}" if x != -1 else "Noise/Outliers")
    
    filtered_countries = explorer_df[explorer_df['Cluster'] == cluster_choice]
    
    st.write(f"Showing **{len(filtered_countries)}** countries in this group:")
    st.dataframe(filtered_countries[['Country'] + list(features)], use_container_width=True)

# --- TAB 3: VISUALIZATION ---
with tab3:
    st.subheader("PCA Cluster Mapping")
    pca_df = pd.DataFrame(pca_all[:, :2], columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels.astype(str)
    pca_df['Country'] = raw_df['Country']
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                     hover_data=['Country'], template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Safe)
    st.plotly_chart(fig, use_container_width=True)
