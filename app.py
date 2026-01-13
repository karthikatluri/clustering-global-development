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
from sklearn.cluster import DBSCAN

# Page configuration
st.set_page_config(
    page_title="Country Clustering Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model and preprocessing objects
@st.cache_resource
def load_model_components():
    try:
        dbscan_model = joblib.load('best_dbscan_model.joblib')
        scaler = joblib.load('scaler_no_outliers.joblib')
        pca = joblib.load('pca_no_outliers.joblib')
        return dbscan_model, scaler, pca, None
    except Exception as e:
        return None, None, None, str(e)

# Load original data and prepare feature names
@st.cache_data
def load_and_prepare_data(file_path):
    try:
        original_df = pd.read_csv(file_path)
        
        # Store country names
        countries = original_df['Country'].copy() if 'Country' in original_df.columns else None
        
        # Prepare model dataframe
        df_model_temp = original_df.drop(columns=['Country']) if 'Country' in original_df.columns else original_df.copy()
        
        # Clean data
        for col in df_model_temp.columns:
            df_model_temp[col] = (
                df_model_temp[col]
                .astype(str)
                .str.replace('%', '', regex=False)
                .str.replace('$', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
        
        df_model_temp = df_model_temp.apply(pd.to_numeric, errors='coerce')
        df_model_temp.fillna(df_model_temp.median(), inplace=True)
        
        feature_names = df_model_temp.columns.tolist()
        
        return df_model_temp, feature_names, countries, original_df, None
    except Exception as e:
        return None, None, None, None, str(e)

# Load trained reference data for visualization
@st.cache_data
def get_trained_data(_scaler, _pca, df_model_temp):
    try:
        scaled_data = _scaler.transform(df_model_temp)
        pca_data = _pca.transform(scaled_data)
        return pca_data
    except Exception as e:
        st.warning(f"Could not prepare reference data: {e}")
        return None

# Header
st.markdown('<p class="main-header">🌍 Country Clustering Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict cluster membership for country development metrics using DBSCAN</p>', unsafe_allow_html=True)

# Load models
dbscan_model, scaler, pca, model_error = load_model_components()

if model_error:
    st.error(f"❌ Error loading models: {model_error}")
    st.info("Please ensure the following files exist in the repository:")
    st.code("best_dbscan_model.joblib\nscaler_no_outliers.joblib\npca_no_outliers.joblib\nWorld_development_mesurement.csv")
    st.stop()

st.success("✅ Models loaded successfully!")

# Sidebar for file upload and configuration
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Training Data (CSV)", 
    type=['csv'],
    help="Upload the World_development_measurement.csv file"
)

# Use default path or uploaded file
if uploaded_file is not None:
    file_path = uploaded_file
else:
    file_path = "World_development_mesurement.csv"
    st.sidebar.info("Using default file: World_development_mesurement.csv")

# Load data
df_model_temp, feature_names, countries, original_df, data_error = load_and_prepare_data(file_path)

if data_error:
    st.error(f"❌ Error loading data: {data_error}")
    st.info("Please make sure 'World_development_mesurement.csv' is in the repository")
    st.stop()

st.sidebar.success(f"✅ Data loaded: {len(feature_names)} features")

# Show DBSCAN parameters and allow adjustments
with st.sidebar.expander("🔧 Model Parameters", expanded=True):
    st.write("**Current Model Settings:**")
    st.write(f"- DBSCAN eps: {dbscan_model.eps:.4f}")
    st.write(f"- DBSCAN min_samples: {dbscan_model.min_samples}")
    st.write(f"- PCA components: {pca.n_components_}")
    
    # Check if model is problematic
    pca_data_check = get_trained_data(scaler, pca, df_model_temp)
    if pca_data_check is not None:
        clusters_check = dbscan_model.fit_predict(pca_data_check)
        n_clusters_check = len(set(clusters_check)) - (1 if -1 in clusters_check else 0)
        
        if n_clusters_check <= 1:
            st.error(f"⚠️ Model has only {n_clusters_check} cluster(s)!")
            st.write("Consider retraining with better parameters.")
    
    st.divider()
    
    # Option to retrain model
    if st.checkbox("🔄 Retrain Model", value=False):
        st.write("**New DBSCAN Parameters:**")
        
        new_eps = st.slider(
            "eps (distance threshold)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Smaller = more clusters, larger = fewer clusters"
        )
        
        new_min_samples = st.slider(
            "min_samples",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Higher = stricter clustering"
        )
        
        if st.button("🚀 Apply New Parameters", type="primary"):
            with st.spinner("Retraining DBSCAN..."):
                from sklearn.cluster import DBSCAN
                new_dbscan = DBSCAN(eps=new_eps, min_samples=new_min_samples)
                new_clusters = new_dbscan.fit_predict(pca_data_check)
                
                n_new_clusters = len(set(new_clusters)) - (1 if -1 in new_clusters else 0)
                
                if n_new_clusters > 1:
                    st.session_state['retrained_model'] = new_dbscan
                    st.success(f"✅ Retrained! Found {n_new_clusters} clusters")
                    st.rerun()
                else:
                    st.error(f"❌ Still only {n_new_clusters} cluster(s). Try different parameters.")
    
    st.divider()
    
    st.write("**Adjust Prediction Threshold:**")
    use_custom_eps = st.checkbox("Use custom eps for prediction", value=False)
    
    if use_custom_eps:
        custom_eps = st.slider(
            "Prediction eps (larger = more lenient)",
            min_value=0.1,
            max_value=5.0,
            value=float(dbscan_model.eps * 1.5),
            step=0.1,
            help="Increase this if too many points are classified as noise"
        )
        st.session_state['prediction_eps'] = custom_eps
    else:
        st.session_state['prediction_eps'] = dbscan_model.eps
    
    custom_min_samples = st.slider(
        "Minimum neighbors required",
        min_value=1,
        max_value=10,
        value=max(1, dbscan_model.min_samples - 2),
        help="Decrease this if too many points are classified as noise"
    )
    st.session_state['prediction_min_samples'] = custom_min_samples
    
    st.divider()
    
    use_fallback = st.checkbox(
        "Use fallback for all-noise neighbors", 
        value=True,
        help="If all neighbors are noise, use the nearest non-noise point instead"
    )
    st.session_state['use_fallback'] = use_fallback
    
    st.info(f"Current: Need {st.session_state['prediction_min_samples']} neighbors within {st.session_state['prediction_eps']:.3f} distance")

# Use retrained model if available
if 'retrained_model' in st.session_state:
    dbscan_model = st.session_state['retrained_model']
    st.sidebar.success("✅ Using retrained model")

# Display data info
with st.expander("📊 Training Data Overview", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Countries", len(df_model_temp))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        variance_explained = sum(pca.explained_variance_ratio_) * 100
        st.metric("PCA Variance", f"{variance_explained:.1f}%")
    
    st.dataframe(df_model_temp.head(10), use_container_width=True)
    
    if st.checkbox("Show all features"):
        st.write("**All Features:**")
        st.write(", ".join(feature_names))
    
    # Check cluster quality
    if st.button("🔍 Analyze Training Clusters"):
        with st.spinner("Analyzing clusters..."):
            pca_data_train = get_trained_data(scaler, pca, df_model_temp)
            if pca_data_train is not None:
                clusters_train = dbscan_model.fit_predict(pca_data_train)
                
                unique_clusters = np.unique(clusters_train)
                n_clusters = len(unique_clusters[unique_clusters != -1])
                n_noise = sum(clusters_train == -1)
                
                st.subheader("🎯 Cluster Quality Analysis")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Clusters Found", n_clusters)
                with col_b:
                    st.metric("Noise Points", f"{n_noise} ({n_noise/len(clusters_train)*100:.1f}%)")
                with col_c:
                    largest = max([sum(clusters_train == i) for i in unique_clusters if i != -1], default=0)
                    st.metric("Largest Cluster Size", f"{largest} ({largest/len(clusters_train)*100:.1f}%)")
                
                # Warning if only one cluster
                if n_clusters == 1:
                    st.error("""
                    ⚠️ **CRITICAL: Only 1 cluster found!**
                    
                    Your DBSCAN model has eps={:.4f} which is TOO LARGE, causing all points to merge into one cluster.
                    
                    **Solutions:**
                    1. Retrain model with smaller eps (try 0.5-2.0)
                    2. Increase min_samples (try 5-10)
                    3. Check PCA transformation quality
                    """.format(dbscan_model.eps))
                
                elif n_clusters == 0:
                    st.error("""
                    ⚠️ **CRITICAL: No clusters found!**
                    
                    Your DBSCAN model has eps={:.4f} which is TOO SMALL, all points are noise.
                    
                    **Solutions:**
                    1. Retrain model with larger eps
                    2. Decrease min_samples
                    """.format(dbscan_model.eps))
                
                # Show cluster distribution
                cluster_counts = pd.DataFrame({
                    'Cluster': [f'Cluster {i}' if i != -1 else 'Noise' for i in unique_clusters],
                    'Count': [sum(clusters_train == i) for i in unique_clusters],
                    'Percentage': [sum(clusters_train == i)/len(clusters_train)*100 for i in unique_clusters]
                })
                cluster_counts = cluster_counts.sort_values('Count', ascending=False)
                
                st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict Cluster", "📈 Visualizations", "📋 Feature Analysis"])

with tab1:
    st.subheader("Enter Country Development Metrics")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Load from Existing Country", "Upload CSV"],
        horizontal=True
    )
    
    user_inputs = {}
    
    if input_method == "Manual Input":
        st.info("💡 Tip: Default values are set to median values from the training data")
        
        # Organize inputs in columns
        num_cols = 3
        
        cols = st.columns(num_cols)
        
        for idx, feature in enumerate(feature_names):
            col_idx = idx % num_cols
            with cols[col_idx]:
                default_value = float(df_model_temp[feature].median())
                min_val = float(df_model_temp[feature].min())
                max_val = float(df_model_temp[feature].max())
                
                user_inputs[feature] = st.number_input(
                    f'{feature}',
                    value=default_value,
                    min_value=min_val * 0.1,
                    max_value=max_val * 2.0,
                    format="%.4f",
                    key=f"input_{feature}",
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
    
    elif input_method == "Load from Existing Country":
        if countries is not None:
            selected_country = st.selectbox(
                "Select a country to load its data:",
                countries.tolist()
            )
            
            country_idx = countries[countries == selected_country].index[0]
            
            st.info(f"📍 Loading data for: **{selected_country}**")
            
            for feature in feature_names:
                user_inputs[feature] = float(df_model_temp.iloc[country_idx][feature])
            
            # Display loaded values in expandable section
            with st.expander("👁️ View loaded values", expanded=True):
                display_df = pd.DataFrame([user_inputs]).T
                display_df.columns = ['Value']
                display_df.index.name = 'Feature'
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Country column not found in dataset")
            input_method = "Manual Input"
    
    elif input_method == "Upload CSV":
        uploaded_predict = st.file_uploader(
            "Upload CSV with feature values",
            type=['csv'],
            key="predict_upload",
            help="CSV should contain columns matching the feature names"
        )
        
        if uploaded_predict:
            predict_df = pd.read_csv(uploaded_predict)
            st.success(f"✅ Loaded {len(predict_df)} rows")
            st.dataframe(predict_df.head(), use_container_width=True)
            
            if st.button("🎯 Use first row for prediction"):
                for feature in feature_names:
                    if feature in predict_df.columns:
                        user_inputs[feature] = float(predict_df.iloc[0][feature])
                    else:
                        user_inputs[feature] = float(df_model_temp[feature].median())
                        st.warning(f"Feature '{feature}' not found in CSV. Using median value.")
    
    # Prediction button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button('🔮 Predict Cluster', type="primary", use_container_width=True)
    
    if predict_button and len(user_inputs) == len(feature_names):
        with st.spinner("Processing prediction..."):
            # Create DataFrame from user inputs
            input_df = pd.DataFrame([user_inputs])
            
            # Preprocess the input data
            try:
                scaled_input = scaler.transform(input_df)
                pca_input = pca.transform(scaled_input)
                
                # Get training data in PCA space
                pca_data_train = get_trained_data(scaler, pca, df_model_temp)
                
                if pca_data_train is not None:
                    # Fit DBSCAN on training data
                    clusters_train = dbscan_model.fit_predict(pca_data_train)
                    
                    # Find k nearest neighbors in training data
                    distances = np.linalg.norm(pca_data_train - pca_input, axis=1)
                    
                    # Get prediction parameters from session state
                    eps = st.session_state.get('prediction_eps', dbscan_model.eps)
                    min_samples_pred = st.session_state.get('prediction_min_samples', max(1, dbscan_model.min_samples - 2))
                    use_fallback = st.session_state.get('use_fallback', True)
                    
                    # Find all neighbors within eps distance
                    neighbors_within_eps = distances <= eps
                    num_neighbors = neighbors_within_eps.sum()
                    
                    if num_neighbors >= min_samples_pred:
                        # Get cluster labels of neighbors within eps
                        neighbor_clusters = clusters_train[neighbors_within_eps]
                        # Remove noise points (-1)
                        valid_clusters = neighbor_clusters[neighbor_clusters != -1]
                        
                        if len(valid_clusters) > 0:
                            # Use majority voting among valid clusters
                            from collections import Counter
                            cluster_counts = Counter(valid_clusters)
                            prediction = cluster_counts.most_common(1)[0][0]
                            confidence_votes = cluster_counts.most_common(1)[0][1]
                        else:
                            # All neighbors are noise
                            if use_fallback:
                                # Use nearest non-noise point
                                non_noise_mask = clusters_train != -1
                                if non_noise_mask.any():
                                    non_noise_distances = distances.copy()
                                    non_noise_distances[~non_noise_mask] = np.inf
                                    nearest_non_noise_idx = np.argmin(non_noise_distances)
                                    prediction = clusters_train[nearest_non_noise_idx]
                                    confidence_votes = 1
                                    nearest_idx = nearest_non_noise_idx
                                    nearest_distance = distances[nearest_non_noise_idx]
                                    st.info(f"ℹ️ All {num_neighbors} neighbors are noise points. Using nearest non-noise point (Cluster {prediction}) at distance {nearest_distance:.4f}")
                                else:
                                    prediction = -1
                                    confidence_votes = 0
                            else:
                                prediction = -1
                                confidence_votes = 0
                    else:
                        # Not enough neighbors, classify as noise
                        prediction = -1
                        confidence_votes = 0
                    
                    nearest_idx = np.argmin(distances)
                    nearest_distance = distances[nearest_idx]
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        if prediction == -1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #ff4b4b;">⚠️ Noise (Outlier)</h3>
                                <p>This data point doesn't fit into any cluster</p>
                                <p style="font-size: 0.9em; margin-top: 0.5rem;">
                                    Found {num_neighbors} neighbors (need {min_samples_pred})<br>
                                    Within eps={eps:.3f}<br>
                                    Nearest distance: {nearest_distance:.3f}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #00cc00;">🎯 Cluster {prediction}</h3>
                                <p>Successfully assigned to a cluster</p>
                                <p style="font-size: 0.9em; margin-top: 0.5rem;">
                                    {num_neighbors} neighbors found within eps={eps:.3f}<br>
                                    {confidence_votes} voted for this cluster
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with result_col2:
                        st.metric("Nearest Neighbor Distance", f"{nearest_distance:.4f}")
                    
                    with result_col3:
                        confidence = max(0, min(100, (1 - nearest_distance / 5) * 100))
                        st.metric("Confidence Score", f"{confidence:.1f}%")
                    
                    # Show explanation if noise
                    if prediction == -1:
                        st.warning(f"""
                        **Why is this classified as Noise?**
                        
                        This point has only **{num_neighbors} neighbors** within distance **{eps:.3f}**, but needs **{min_samples_pred}** to form/join a cluster.
                        The nearest training point is **{nearest_distance:.3f}** away.
                        
                        **Possible reasons:**
                        - The input values are significantly different from training data
                        - The country has a unique development profile
                        - Some feature values may be outliers
                        
                        **Solutions:**
                        - ✅ **Adjust prediction threshold** in the sidebar (increase eps or decrease min neighbors)
                        - Check if input values are realistic and within normal ranges
                        - Compare with similar countries to identify unusual features
                        - Consider that this might represent a genuinely unique development pattern
                        """)
                        
                        # Show neighbor distribution
                        st.subheader("📊 Neighbor Analysis")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Show distance distribution
                            distances_sorted = np.sort(distances)[:20]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=distances_sorted,
                                mode='lines+markers',
                                name='Distance to nearest 20 points',
                                line=dict(color='blue')
                            ))
                            fig.add_hline(y=eps, line_dash="dash", line_color="red", 
                                         annotation_text=f"eps threshold = {eps:.3f}")
                            fig.update_layout(
                                title="Distance to Nearest Training Points",
                                yaxis_title="Distance",
                                xaxis_title="Rank",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            st.metric("Neighbors within eps", num_neighbors)
                            st.metric("Neighbors needed", min_samples_pred)
                            st.metric("Nearest neighbor distance", f"{nearest_distance:.4f}")
                            if num_neighbors > 0:
                                st.metric("Average neighbor distance", f"{distances[neighbors_within_eps].mean():.4f}")
                                # Show cluster distribution of neighbors
                                neighbor_clusters = clusters_train[neighbors_within_eps]
                                noise_neighbors = (neighbor_clusters == -1).sum()
                                valid_neighbors = num_neighbors - noise_neighbors
                                st.metric("Neighbors that are noise", f"{noise_neighbors}/{num_neighbors}")
                                st.metric("Neighbors in clusters", f"{valid_neighbors}/{num_neighbors}")
                        
                        # Show which features are most different
                        if countries is not None:
                            st.subheader("📊 Comparison with Nearest Country")
                            nearest_country = countries.iloc[nearest_idx]
                            
                            comparison_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Your Input': [user_inputs[f] for f in feature_names],
                                f'{nearest_country}': df_model_temp.iloc[nearest_idx].values
                            })
                            comparison_df['Difference'] = comparison_df['Your Input'] - comparison_df[f'{nearest_country}']
                            comparison_df['% Difference'] = (comparison_df['Difference'] / comparison_df[f'{nearest_country}'] * 100).round(2)
                            comparison_df['Abs % Diff'] = comparison_df['% Difference'].abs()
                            
                            # Sort by absolute difference
                            comparison_df = comparison_df.sort_values('Abs % Diff', ascending=False)
                            
                            st.write(f"**Nearest country:** {nearest_country} (distance: {nearest_distance:.3f})")
                            st.write("**Top 10 most different features:**")
                            st.dataframe(
                                comparison_df[['Feature', 'Your Input', f'{nearest_country}', '% Difference']].head(10),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Visualization of prediction point
                    st.subheader("📍 Prediction Point in PCA Space")
                    
                    # Create visualization dataframe
                    viz_df = pd.DataFrame({
                        'PC1': pca_data_train[:, 0],
                        'PC2': pca_data_train[:, 1],
                        'Cluster': clusters_train,
                        'Type': 'Training Data'
                    })
                    
                    # Add prediction point
                    pred_point = pd.DataFrame({
                        'PC1': [pca_input[0, 0]],
                        'PC2': [pca_input[0, 1]],
                        'Cluster': [prediction],
                        'Type': ['Prediction']
                    })
                    
                    viz_df = pd.concat([viz_df, pred_point], ignore_index=True)
                    viz_df['Cluster_Label'] = viz_df['Cluster'].apply(lambda x: f'Noise' if x == -1 else f'Cluster {x}')
                    
                    fig = px.scatter(
                        viz_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster_Label',
                        symbol='Type',
                        title='Prediction Point in PCA Space',
                        labels={
                            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                        },
                        height=500
                    )
                    
                    fig.update_traces(marker=dict(size=10))
                    if len(fig.data) > 0:
                        fig.data[-1].marker.size = 20
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show similar countries
                    if countries is not None and prediction != -1:
                        st.subheader("📍 Similar Countries in Same Cluster")
                        
                        same_cluster_mask = clusters_train == prediction
                        same_cluster_indices = np.where(same_cluster_mask)[0]
                        
                        if len(same_cluster_indices) > 0:
                            # Calculate distances to all countries in same cluster
                            cluster_distances = distances[same_cluster_indices]
                            sorted_indices = same_cluster_indices[np.argsort(cluster_distances)[:10]]
                            
                            similar_df = pd.DataFrame({
                                'Rank': range(1, len(sorted_indices) + 1),
                                'Country': countries.iloc[sorted_indices].values,
                                'Distance': distances[sorted_indices].round(4),
                                'Cluster': clusters_train[sorted_indices]
                            })
                            
                            st.dataframe(similar_df, use_container_width=True, hide_index=True)
                            
                            # Show feature comparison with nearest country
                            st.subheader("📊 Feature Comparison with Nearest Country")
                            
                            nearest_country = countries.iloc[nearest_idx]
                            comparison_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Your Input': [user_inputs[f] for f in feature_names],
                                f'{nearest_country}': df_model_temp.iloc[nearest_idx].values
                            })
                            comparison_df['Difference'] = comparison_df['Your Input'] - comparison_df[f'{nearest_country}']
                            comparison_df['Difference %'] = (comparison_df['Difference'] / comparison_df[f'{nearest_country}'] * 100).round(2)
                            
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Download prediction results
                    st.divider()
                    result_df = pd.DataFrame({
                        'Feature': feature_names + ['Predicted_Cluster', 'Confidence', 'Distance'],
                        'Value': list(user_inputs.values()) + [prediction, f"{confidence:.1f}%", f"{nearest_distance:.4f}"]
                    })
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction Results",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.exception(e)
    
    elif predict_button:
        st.warning("⚠️ Please fill in all feature values")

with tab2:
    st.subheader("Cluster Visualization in PCA Space")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_viz = st.button("🔄 Generate Visualization", type="primary", use_container_width=True)
    
    if show_viz:
        with st.spinner("Generating visualizations..."):
            pca_data_train = get_trained_data(scaler, pca, df_model_temp)
            
            if pca_data_train is not None:
                clusters_train = dbscan_model.fit_predict(pca_data_train)
                
                # Create visualization dataframe
                viz_df = pd.DataFrame({
                    'PC1': pca_data_train[:, 0],
                    'PC2': pca_data_train[:, 1],
                    'Cluster': clusters_train,
                    'Cluster_Label': [f'Noise' if c == -1 else f'Cluster {c}' for c in clusters_train]
                })
                
                if countries is not None:
                    viz_df['Country'] = countries.values
                
                # Plot type selection
                plot_type = st.radio("Visualization Type:", ["Interactive (Plotly)", "Static (Matplotlib)"], horizontal=True)
                
                if plot_type == "Interactive (Plotly)":
                    # 2D Plotly scatter
                    fig = px.scatter(
                        viz_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster_Label',
                        hover_data=['Country'] if 'Country' in viz_df.columns else None,
                        title=f'Country Clusters in PCA Space (Total: {len(set(clusters_train)) - (1 if -1 in clusters_train else 0)} clusters)',
                        labels={
                            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
                        },
                        height=600
                    )
                    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3D plot if available
                    if pca_data_train.shape[1] >= 3:
                        st.subheader("3D Cluster Visualization")
                        viz_df['PC3'] = pca_data_train[:, 2]
                        
                        fig_3d = px.scatter_3d(
                            viz_df,
                            x='PC1',
                            y='PC2',
                            z='PC3',
                            color='Cluster_Label',
                            hover_data=['Country'] if 'Country' in viz_df.columns else None,
                            title='Country Clusters in 3D PCA Space',
                            labels={
                                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)'
                            },
                            height=700
                        )
                        fig_3d.update_traces(marker=dict(size=5))
                        st.plotly_chart(fig_3d, use_container_width=True)
                
                else:
                    # Matplotlib static plot
                    fig, ax = plt.subplots(figsize=(14, 9))
                    
                    unique_clusters = sorted(set(clusters_train))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
                    
                    for cluster, color in zip(unique_clusters, colors):
                        mask = clusters_train == cluster
                        label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
                        ax.scatter(
                            pca_data_train[mask, 0],
                            pca_data_train[mask, 1],
                            c=[color],
                            label=label,
                            s=100,
                            alpha=0.6,
                            edgecolors='k',
                            linewidths=0.5
                        )
                    
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
                    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
                    ax.set_title("Country Clusters in PCA Space", fontsize=16, fontweight='bold')
                    ax.legend(loc='best', framealpha=0.9, fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                # Cluster statistics
                st.subheader("📊 Cluster Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    n_clusters = len(set(clusters_train)) - (1 if -1 in clusters_train else 0)
                    st.metric("Total Clusters", n_clusters)
                
                with col2:
                    n_noise = sum(clusters_train == -1)
                    noise_pct = (n_noise / len(clusters_train)) * 100
                    st.metric("Noise Points", f"{n_noise} ({noise_pct:.1f}%)")
                
                with col3:
                    largest_cluster = max([sum(clusters_train == i) for i in set(clusters_train) if i != -1], default=0)
                    st.metric("Largest Cluster", largest_cluster)
                
                with col4:
                    avg_cluster_size = np.mean([sum(clusters_train == i) for i in set(clusters_train) if i != -1])
                    st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
                
                # Cluster size distribution
                st.subheader("Cluster Size Distribution")
                
                cluster_sizes = pd.DataFrame({
                    'Cluster': [f'Cluster {i}' if i != -1 else 'Noise' for i in sorted(set(clusters_train))],
                    'Size': [sum(clusters_train == i) for i in sorted(set(clusters_train))]
                })
                
                fig = px.bar(
                    cluster_sizes,
                    x='Cluster',
                    y='Size',
                    title='Number of Countries per Cluster',
                    color='Size',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔍 Feature Analysis")
    
    # PCA explained variance
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PCA Explained Variance**")
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Variance (%)': (pca.explained_variance_ratio_ * 100).round(2),
            'Cumulative (%)': (np.cumsum(pca.explained_variance_ratio_) * 100).round(2)
        })
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=variance_df['Component'],
            y=variance_df['Variance (%)'],
            name='Individual',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=variance_df['Component'],
            y=variance_df['Cumulative (%)'],
            name='Cumulative',
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            yaxis='y2'
        ))
        fig.update_layout(
            title='Scree Plot - PCA Variance Explained',
            xaxis_title='Principal Component',
            yaxis_title='Variance Explained (%)',
            yaxis2=dict(title='Cumulative Variance (%)', overlaying='y', side='right'),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feature statistics
    st.subheader("📈 Feature Statistics")
    stats_df = df_model_temp.describe().T
    stats_df = stats_df.round(4)
    stats_df['range'] = stats_df['max'] - stats_df['min']
    st.dataframe(stats_df, use_container_width=True)
    
    # Feature distribution
    st.subheader("📊 Feature Distribution")
    selected_feature = st.selectbox("Select feature to visualize:", feature_names)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df_model_temp,
            x=selected_feature,
            nbins=30,
            title=f'Distribution of {selected_feature}',
            color_discrete_sequence=['steelblue']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_model_temp,
            y=selected_feature,
            title=f'Box Plot of {selected_feature}',
            color_discrete_sequence=['coral']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feature correlations
    st.subheader("🔗 Feature Correlation Analysis")
    
    if st.checkbox("Show Feature Correlation Heatmap"):
        with st.spinner("Generating correlation heatmap..."):
            fig, ax = plt.subplots(figsize=(16, 14))
            correlation = df_model_temp.corr()
            
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(
                correlation,
                mask=mask,
                annot=False,
                cmap='coolwarm',
                center=0,
                ax=ax,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
    
    # PCA loadings
    st.subheader("🎯 PCA Feature Loadings")
    
    n_components_to_show = min(5, pca.n_components_)
    loadings = pca.components_[:n_components_to_show, :]
    
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i+1}' for i in range(n_components_to_show)],
        index=feature_names
    )
    
    # Show top features for each PC
    selected_pc = st.selectbox(
        "Select Principal Component:",
        [f'PC{i+1}' for i in range(n_components_to_show)]
    )
    
    pc_loadings = loadings_df[selected_pc].abs().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=pc_loadings.values,
        y=pc_loadings.index,
        orientation='h',
        title=f'Top 10 Features for {selected_pc}',
        labels={'x': 'Absolute Loading', 'y': 'Feature'},
        color=pc_loadings.values,
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🌍 Country Clustering Prediction System</strong></p>
    <p style='font-size: 0.9rem;'>Powered by DBSCAN & PCA | Built with Streamlit</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
        Upload your data or use existing countries to predict cluster membership based on development metrics
    </p>
</div>
""", unsafe_allow_html=True)
