import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from collections import Counter
from sklearn.metrics import pairwise_distances

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Development Predictor", page_icon="🌍", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; margin-top: 10px;}
    .similarity-box { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 5px solid #2196f3; margin-bottom: 5px; }
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

# --- SIDEBAR: SELECTION ---
st.sidebar.header("📍 Data Selection")
selection_mode = st.sidebar.radio("Input Method", ["Manual Entry", "Select Existing Country"])

selected_country_data = None
current_country_name = "User Input"
if selection_mode == "Select Existing Country":
    country_list = sorted(raw_df['Country'].unique())
    current_country_name = st.sidebar.selectbox("Choose a Country", country_list)
    selected_country_data = df_numeric[raw_df['Country'] == current_country_name].iloc[0]

# --- MAIN INTERFACE ---
st.markdown('<p class="main-header">🌍 Country Clustering & Similarity Engine</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Prediction & Top 10", "🔎 Cluster Explorer", "📈 Visual Analysis"])

# Pre-calculate whole dataset PCA & Clusters for reference
features = df_numeric.columns.drop('Country') if 'Country' in df_numeric.columns else df_numeric.columns
pca_train = pca.transform(scaler.transform(df_numeric[features]))
train_labels = dbscan_model.fit_predict(pca_train)

with tab1:
    st.subheader(f"Analyzing: {current_country_name}")
    cols = st.columns(3)
    user_inputs = {}
    for i, feat in enumerate(features):
        default_val = float(selected_country_data[feat]) if selected_country_data is not None else float(df_numeric[feat].median())
        user_inputs[feat] = cols[i % 3].number_input(f"{feat}", value=default_val)

    if st.button("🔮 Run Similarity Analysis", type="primary"):
        # Process Input
        input_df = pd.DataFrame([user_inputs])
        pca_input = pca.transform(scaler.transform(input_df))
        
        # Determine Cluster
        distances = np.linalg.norm(pca_train - pca_input, axis=1)
        nearby_indices = np.where(distances <= dbscan_model.eps)[0]
        
        assigned_cluster = -1
        if len(nearby_indices) >= dbscan_model.min_samples:
            votes = [train_labels[i] for i in nearby_indices if train_labels[i] != -1]
            if votes: assigned_cluster = Counter(votes).most_common(1)[0][0]

        # Results Display
        if assigned_cluster != -1:
            st.success(f"### Result: Cluster {assigned_cluster}")
            
            # --- TOP 10 LOGIC ---
            st.markdown("#### 🏆 Top 10 Most Similar Countries")
            st.write("Based on closest distance in the multi-dimensional development space:")
            
            # Filter training data to only this cluster
            cluster_mask = (train_labels == assigned_cluster)
            cluster_pca = pca_train[cluster_mask]
            cluster_countries = raw_df['Country'][cluster_mask].values
            
            # Calculate distances only to members of the same cluster
            cluster_dist = np.linalg.norm(cluster_pca - pca_input, axis=1)
            
            # Combine, sort, and pick top 10 (excluding itself if it's already in the data)
            sim_df = pd.DataFrame({'Country': cluster_countries, 'Distance': cluster_dist})
            sim_df = sim_df[sim_df['Country'] != current_country_name].sort_values('Distance').head(10)
            
            for idx, row in sim_df.iterrows():
                st.markdown(f"""<div class="similarity-box">
                    <b>{row['Country']}</b> <span style='float:right; color:gray;'>Match Score: {max(0, 100 - row['Distance']*10):.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("### Result: Noise/Outlier")
            st.info("This country has a unique development profile that doesn't closely match any major groups.")

with tab2:
    # (Explorer code remains same as previous version)
    cluster_choice = st.selectbox("Select Cluster to View", sorted(np.unique(train_labels)))
    st.dataframe(raw_df[train_labels == cluster_choice], use_container_width=True)

with tab3:
    # Visualization with current point highlighted
    viz_df = pd.DataFrame(pca_train[:, :2], columns=['PC1', 'PC2'])
    viz_df['Cluster'] = train_labels.astype(str)
    viz_df['Country'] = raw_df['Country']
    
    fig = px.scatter(viz_df, x='PC1', y='PC2', color='Cluster', hover_data=['Country'], title="Development Space")
    st.plotly_chart(fig, use_container_width=True)
