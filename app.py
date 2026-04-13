import streamlit as st
import pandas as pd
import io
import plotly.express as px
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Relative imports from structured packages
from model.supervised_pipeline import preprocess_data, train_and_evaluate
from model.unsupervised_pipeline import preprocess_unsupervised, auto_kmeans_clustering, auto_tune_clustering
from agent.agent import analyze_dataset_initial, self_critique_models, chat_with_copilot
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Self-Improving AI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-Mode Compatible Glassmorphism CSS + Font Size Improvements
st.markdown("""
<style>
    /* Improve Tabs Text Size and Clarity */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem !important; 
        font-weight: 600 !important;
        padding: 5px 10px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-title { font-size: 1.2rem; color: #a1a1aa; margin-bottom: 5px; font-weight: 500; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #3b82f6; }
    h1 { font-size: 3rem !important; font-weight: 800; }
    h2 { font-size: 2.2rem !important; }
    h3 { font-size: 1.8rem !important; }
    /* Increase base text size slightly */
    .stMarkdown p { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Chat History and ML Context Memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hello! I am your AI Copilot. Upload a dataset and run the ML pipelines, then ask me to explain the models, preprocessing steps, or data patterns!")
    ]

if "ml_context" not in st.session_state:
    st.session_state["ml_context"] = "No pipeline has been run yet. Tell the user to run the Supervised or Unsupervised pipeline first."

st.title("🧠 Multi-AI Agent Workstation")
st.markdown("Upload a dataset, run comprehensive ML pipelines, and chat dynamically with the AI Copilot to understand the results.")

# Sidebar Configuration
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Populate the Sidebar with Stats to keep it active
    st.sidebar.divider()
    st.sidebar.subheader("Dataset Dashboard")
    st.sidebar.write(f"**Rows:** {df.shape[0]}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")
    missing_data = df.isnull().sum().sum()
    st.sidebar.write(f"**Total Missing Values:** {missing_data}")
    st.sidebar.write("**Column Types:**")
    st.sidebar.dataframe(df.dtypes.astype(str).rename("Type"), use_container_width=True)
    
    # Create the Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data & Analyst Preview", 
        "🎯 Supervised Leaderboard", 
        "🧩 Unsupervised Clustering",
        "💬 AI Copilot Chat"
    ])
    
    # --- TAB 1: DATA & ANALYST ---
    with tab1:
        st.header("Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.divider()
        if st.button("🔍 Request AI Data Analyst Review"):
            with st.spinner("Agent is analyzing the dataset structure..."):
                buffer = io.StringIO()
                df.info(buf=buffer)
                s_info = buffer.getvalue()
                
                analysis = analyze_dataset_initial(df.head(5).to_string(), s_info)
                st.info(analysis)
                
                # Update memory context quietly
                st.session_state["ml_context"] += f"\n\nInitial Dataset Analysis:\n{analysis}"
            
            # --- CLEAN DATASET SECTION ---
            st.divider()
            st.subheader("🧹 Cleaned Dataset")
            st.write("Missing values filled, categorical columns encoded, and ready for download.")
            
            clean_df = df.copy()
            
            # Impute numerical missing values with mean
            num_cols = clean_df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = clean_df.select_dtypes(exclude=['int64', 'float64']).columns
            
            if len(num_cols) > 0:
                from sklearn.impute import SimpleImputer
                num_imputer = SimpleImputer(strategy='mean')
                clean_df[num_cols] = num_imputer.fit_transform(clean_df[num_cols])
            
            if len(cat_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                clean_df[cat_cols] = cat_imputer.fit_transform(clean_df[cat_cols])
            
            # Show cleaning stats
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><div class='metric-title'>Original Missing</div><div class='metric-value'>{df.isnull().sum().sum()}</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-title'>After Cleaning</div><div class='metric-value'>{clean_df.isnull().sum().sum()}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-title'>Total Rows</div><div class='metric-value'>{len(clean_df)}</div></div>", unsafe_allow_html=True)
            
            st.dataframe(clean_df.head(10), use_container_width=True)
            
            # Download button
            csv_data = clean_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
    # --- TAB 2: SUPERVISED ---
    with tab2:
        st.header("Supervised Machine Learning")
        st.write("Select a target column to predict and find the best algorithm.")
        target_col = st.selectbox("Select Target Column", options=df.columns, index=len(df.columns)-1)
        
        # Dual Buttons for Basic vs Advanced
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            run_basic = st.button("🚀 Run Baseline Pipeline", type="primary", use_container_width=True)
        with c_btn2:
            run_advanced = st.button("🔥 Auto-Tune & Apply AI Suggestions (Slower)", type="secondary", use_container_width=True)
            
        if run_basic or run_advanced:
            if run_advanced:
                st.info("Applying AI Suggestions: SMOTE Class Balancing and RandomizedSearchCV Hyperparameter Tuning initialized...")
                
            spinner_msg = "Hyper-Tuning Models (This might take 30-60s)..." if run_advanced else "Training Baseline Models..."
            with st.spinner(spinner_msg):
                try:
                    X_scaled, y, task_type, label_encoder = preprocess_data(df, target_col)
                    
                    results_df, best_model_name, best_model, trained_models, X_test, y_test, tuning_info = train_and_evaluate(X_scaled, y, task_type, apply_improvements=run_advanced)
                    
                    st.success(f"Pipeline Execution Complete! Identified Task: **{task_type.capitalize()}**")
                    st.markdown(f"### Best Performing Model: `{best_model_name}`")
                    
                    if task_type == 'classification':
                        best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>Accuracy</div><div class='metric-value'>{best_row['Accuracy']:.3f}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>Precision</div><div class='metric-value'>{best_row['Precision']:.3f}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>Recall</div><div class='metric-value'>{best_row['Recall']:.3f}</div></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='metric-card'><div class='metric-title'>F1 Score</div><div class='metric-value'>{best_row['F1 Score']:.3f}</div></div>", unsafe_allow_html=True)
                        
                        fig = px.bar(results_df, x='Model', y='Accuracy', color='Model', title="Model Accuracy Comparison")
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        best_row = results_df[results_df['Model'] == best_model_name].iloc[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>RMSE</div><div class='metric-value'>{best_row['RMSE']:.3f}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>MSE</div><div class='metric-value'>{best_row['MSE']:.3f}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>MAE</div><div class='metric-value'>{best_row['MAE']:.3f}</div></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='metric-card'><div class='metric-title'>R2 Score</div><div class='metric-value'>{best_row['R2 Score']:.3f}</div></div>", unsafe_allow_html=True)
                        
                        fig = px.bar(results_df, x='Model', y='RMSE', color='Model', title="Model RMSE Comparison (Lower is Better)")
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if task_type == 'classification':
                        styled_df = results_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
                    else:
                        # RdYlGn_r reverses the colormap so lower error = green
                        styled_df = results_df.style.background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MSE', 'MAE'])
                        
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # --- PIPELINE CONFIGURATION & PARAMETERS USED ---
                    st.divider()
                    st.markdown("### ⚙️ Pipeline Configuration & Parameters Used")
                    
                    # Preprocessing summary
                    with st.expander("🔧 Preprocessing Steps Applied", expanded=True):
                        preprocess_details = f"""
| Step | Detail | Why / Impact |
|---|---|---|
| **Missing Values (Numeric)** | Mean Imputation | Fills gaps with column average — preserves distribution shape |
| **Missing Values (Categorical)** | Mode Imputation | Fills with most frequent value — avoids data loss from row drops |
| **Categorical Encoding** | One-Hot Encoding (drop_first) | Converts text categories to binary columns — required for ML algorithms |
| **Feature Scaling** | RobustScaler (outlier-resistant) | Scales using median/IQR instead of mean/std — handles outliers better than StandardScaler |
| **Low-Variance Filter** | VarianceThreshold (0.01) | Removes near-constant features — reduces noise and speeds up training |
| **Correlation Filter** | Drop features with >0.95 correlation | Removes redundant features — prevents multicollinearity issues |
| **Train/Test Split** | 80/20 {"(Stratified)" if task_type == "classification" else ""} | {"Stratified split preserves class ratios in both sets — critical for imbalanced data" if task_type == "classification" else "Random split — ensures unbiased evaluation"} |
| **SMOTE Applied** | {"✅ Yes (auto strategy)" if run_advanced and task_type == "classification" else "❌ No"} | {"Generates synthetic minority samples — balances class distribution for fairer training" if run_advanced and task_type == "classification" else "No oversampling — baseline performance on original data distribution"} |
| **Features Used** | {X_scaled.shape[1]} columns | Total features after encoding, filtering, and correlation removal |
| **Training Samples** | {X_scaled.shape[0]} rows | Total data points used for model training |
"""
                        st.markdown(preprocess_details)
                    
                    # Model parameters for each model
                    with st.expander("📊 Model Hyperparameters (Per Model)", expanded=True):
                        for model_name in results_df['Model'].tolist():
                            info = tuning_info.get(model_name, {})
                            if info.get('tuned'):
                                st.markdown(f"**{model_name}** — 🔥 Auto-Tuned")
                                params_data = []
                                for k, v in info['best_params'].items():
                                    params_data.append({"Parameter": k, "Best Value": str(v)})
                                params_df = pd.DataFrame(params_data)
                                st.dataframe(params_df, use_container_width=True, hide_index=True)
                                st.caption(f"Best CV Score: {info['best_cv_score']}")
                            else:
                                st.markdown(f"**{model_name}** — Default Parameters")
                                default_params = info.get('default_params', {})
                                # Show only the most important params
                                key_params = {k: v for k, v in default_params.items() 
                                              if k in ['n_estimators', 'max_depth', 'learning_rate', 'C', 'kernel', 
                                                        'n_neighbors', 'weights', 'hidden_layer_sizes', 'activation',
                                                        'alpha', 'max_iter', 'min_samples_split', 'min_samples_leaf',
                                                        'criterion', 'solver', 'penalty', 'max_features',
                                                        'subsample', 'colsample_bytree', 'eval_metric', 'algorithm']}
                                if key_params:
                                    params_data = [{"Parameter": k, "Value": str(v)} for k, v in key_params.items()]
                                    st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)
                                else:
                                    st.caption("Using sklearn defaults")
                            st.markdown("---")
                    
                    # --- IMPROVEMENT SUGGESTIONS ---
                    st.divider()
                    tuning_summary = "\n".join([
                        f"- {name}: {'Auto-Tuned with params ' + str(info.get('best_params', {})) if info.get('tuned') else 'Default params'}"
                        for name, info in tuning_info.items()
                    ])
                    
                    with st.spinner("Generating Agent Critique based on results..."):
                        critique = self_critique_models(
                            results_df.to_string() + f"\n\nParameters Used:\n{tuning_summary}",
                            best_model_name
                        )
                        st.markdown("### 💡 AI Improvement Suggestions")
                        st.write(critique)
                    
                    # Store model + preprocessing info in session for prediction
                    st.session_state["trained_best_model"] = best_model
                    st.session_state["trained_best_model_name"] = best_model_name
                    st.session_state["trained_feature_cols"] = list(X_scaled.columns)
                    st.session_state["trained_task_type"] = task_type
                    st.session_state["trained_label_encoder"] = label_encoder
                    st.session_state["trained_original_df"] = df
                    st.session_state["trained_target_col"] = target_col
                    
                    # Update Chat Context Memory
                    st.session_state["ml_context"] = f"Supervised Pipeline Run for '{target_col}'. Advanced Auto-Tuning Applied: {run_advanced}. Best Model: {best_model_name}. \nResults:\n{results_df.to_string()} \n\nParameters Used:\n{tuning_summary}\n\nCritique:\n{critique} \n\nPreprocessing used: Dropped NaN targets, Mean Imputer for nums, Mode Imputer+GetDummies for cats, RobustScaler, VarianceThreshold, Correlation Filter. SMOTE Active: {run_advanced}"
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        # --- TRY THE MODEL: Live Prediction ---
        if "trained_best_model" in st.session_state:
            st.divider()
            st.markdown("### 🎯 Try the Model — Live Prediction")
            st.write(f"Enter values below to get a prediction from **{st.session_state['trained_best_model_name']}**.")
            
            original_df = st.session_state["trained_original_df"]
            target_col_saved = st.session_state["trained_target_col"]
            feature_cols_saved = st.session_state["trained_feature_cols"]
            task_type_saved = st.session_state["trained_task_type"]
            label_enc_saved = st.session_state["trained_label_encoder"]
            model_saved = st.session_state["trained_best_model"]
            
            # Get original feature columns (before preprocessing)
            original_features = [c for c in original_df.columns if c != target_col_saved]
            
            with st.form("prediction_form"):
                st.markdown("**Enter feature values:**")
                input_values = {}
                
                # Create input fields in a grid (3 columns)
                cols_per_row = 3
                for i in range(0, len(original_features), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, col_name in enumerate(original_features[i:i+cols_per_row]):
                        with row_cols[j]:
                            col_data = original_df[col_name]
                            if col_data.dtype in ['int64', 'float64']:
                                # Numeric: show number input with min/max/mean
                                col_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                                col_max = float(col_data.max()) if not pd.isna(col_data.max()) else 100.0
                                col_mean = float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0
                                input_values[col_name] = st.number_input(
                                    f"{col_name}", 
                                    min_value=col_min, max_value=col_max, value=col_mean,
                                    help=f"Range: {col_min:.2f} – {col_max:.2f}"
                                )
                            else:
                                # Categorical: show selectbox with unique values
                                unique_vals = col_data.dropna().unique().tolist()
                                input_values[col_name] = st.selectbox(
                                    f"{col_name}", options=unique_vals
                                )
                
                predict_btn = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)
            
            if predict_btn:
                try:
                    # Build input DataFrame
                    input_df = pd.DataFrame([input_values])
                    
                    # Apply same preprocessing as training
                    from sklearn.impute import SimpleImputer
                    from sklearn.preprocessing import RobustScaler
                    from sklearn.feature_selection import VarianceThreshold
                    
                    num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
                    cat_cols = input_df.select_dtypes(exclude=['int64', 'float64']).columns
                    
                    if len(num_cols) > 0:
                        num_imputer = SimpleImputer(strategy='mean')
                        num_imputer.fit(original_df[num_cols])
                        input_df[num_cols] = num_imputer.transform(input_df[num_cols])
                    
                    if len(cat_cols) > 0:
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        cat_imputer.fit(original_df[cat_cols])
                        input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])
                        input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
                    
                    # Align columns with training features
                    for col in feature_cols_saved:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[feature_cols_saved]
                    
                    # Scale
                    scaler = RobustScaler()
                    scaler.fit(original_df.drop(columns=[target_col_saved]).select_dtypes(include=['int64', 'float64']))
                    # Scale only numeric columns that exist
                    num_in_features = [c for c in input_df.columns if c in original_df.select_dtypes(include=['int64', 'float64']).columns]
                    if len(num_in_features) > 0:
                        input_df[num_in_features] = scaler.transform(input_df[num_in_features])
                    
                    prediction = model_saved.predict(input_df)[0]
                    
                    # Decode if classification with label encoder
                    if task_type_saved == 'classification' and label_enc_saved is not None:
                        prediction = label_enc_saved.inverse_transform([int(prediction)])[0]
                    
                    st.markdown(f"""
                    <div class='metric-card' style='border: 2px solid #3b82f6;'>
                        <div class='metric-title'>🔮 Predicted {target_col_saved}</div>
                        <div class='metric-value' style='font-size: 2.5rem;'>{prediction}</div>
                        <div style='color: #a1a1aa; margin-top: 8px;'>Model: {st.session_state['trained_best_model_name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # --- TAB 3: UNSUPERVISED ---
    with tab3:
        st.header("Unsupervised Clustering")
        st.write("Find hidden patterns. Choose **Baseline** for quick KMeans or **Auto-Tune** to compare 5 algorithms with hyperparameter optimization.")
        
        # Dual Buttons
        uc1, uc2 = st.columns(2)
        with uc1:
            run_baseline_clust = st.button("🧩 Run Baseline KMeans", type="primary", use_container_width=True)
        with uc2:
            run_autotune_clust = st.button("🔥 Auto-Tune All Algorithms (Slower)", type="secondary", use_container_width=True)
        
        if run_baseline_clust or run_autotune_clust:
            spinner_msg = "Auto-Tuning 5 Clustering Algorithms..." if run_autotune_clust else "Finding Optimal K for KMeans..."
            with st.spinner(spinner_msg):
                try:
                    X_scaled = preprocess_unsupervised(df)
                    
                    if run_autotune_clust:
                        (best_k, best_score, best_algo_name, best_config,
                         comparison_df, scores_df, inertias_df, pca_df, explained_variance) = auto_tune_clustering(X_scaled)
                        
                        st.success(f"Auto-Tuning Complete! Best Algorithm: **{best_algo_name}**")
                        
                        # Metric cards
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>Best Algorithm</div><div class='metric-value' style='font-size:1.4rem;'>{best_algo_name}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>Optimal Clusters</div><div class='metric-value'>{best_k}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>Silhouette Score</div><div class='metric-value'>{best_score:.3f}</div></div>", unsafe_allow_html=True)
                        total_var = sum(explained_variance[:2]) * 100
                        c4.markdown(f"<div class='metric-card'><div class='metric-title'>PCA Variance</div><div class='metric-value'>{total_var:.1f}%</div></div>", unsafe_allow_html=True)
                        
                        # Algorithm Comparison Table
                        st.subheader("📊 Algorithm Comparison Leaderboard")
                        styled_comp = comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Silhouette Score'])
                        st.dataframe(styled_comp, use_container_width=True, hide_index=True)
                        
                        # Best Config Details
                        with st.expander(f"⚙️ Best Config: {best_algo_name}", expanded=True):
                            config_data = [{"Parameter": k, "Value": str(v)} for k, v in best_config.items()]
                            st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)
                        
                        # Bar chart comparing algorithms
                        fig_comp = px.bar(comparison_df, x='Algorithm', y='Silhouette Score', color='Algorithm',
                                         title='Algorithm Silhouette Score Comparison',
                                         color_discrete_sequence=px.colors.qualitative.Bold)
                        fig_comp.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    else:
                        best_k, best_score, scores_df, inertias_df, pca_df, best_model, explained_variance = auto_kmeans_clustering(X_scaled)
                        best_algo_name = 'KMeans'
                        
                        st.success("KMeans Clustering Complete!")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>Optimal Clusters (K)</div><div class='metric-value'>{best_k}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>Silhouette Score</div><div class='metric-value'>{best_score:.3f}</div></div>", unsafe_allow_html=True)
                        total_var = sum(explained_variance[:2]) * 100
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>PCA Variance</div><div class='metric-value'>{total_var:.1f}%</div></div>", unsafe_allow_html=True)
                    
                    # Side by side: Silhouette Score + Elbow Method
                    chart_c1, chart_c2 = st.columns(2)
                    with chart_c1:
                        fig_sil = px.line(scores_df, x='K', y='Silhouette Score', markers=True, title='Silhouette Score vs K')
                        fig_sil.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_sil, use_container_width=True)
                    with chart_c2:
                        fig_elbow = px.line(inertias_df, x='K', y='Inertia', markers=True, title='Elbow Method (Inertia vs K)')
                        fig_elbow.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_elbow, use_container_width=True)
                    
                    st.divider()
                    
                    # PCA Visualization
                    st.subheader("PCA Visualization of Clusters")
                    if 'PCA3' in pca_df.columns:
                        fig_pca = px.scatter_3d(
                            pca_df, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
                            title=f'3D PCA Cluster Visualization ({best_algo_name}, K={best_k})',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_pca.update_layout(template="plotly_dark", height=600)
                    else:
                        fig_pca = px.scatter(
                            pca_df, x='PCA1', y='PCA2', color='Cluster',
                            title=f'2D PCA Cluster Visualization ({best_algo_name}, K={best_k})',
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_pca.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Cluster size distribution
                    st.subheader("Cluster Size Distribution")
                    cluster_counts = pca_df['Cluster'].value_counts().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster',
                                    title='Number of Samples per Cluster',
                                    color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_bar.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Update Chat Context Memory
                    algo_info = f"Algorithm: {best_algo_name}. Auto-Tuned: {run_autotune_clust}."
                    st.session_state["ml_context"] = f"Unsupervised Pipeline Run. {algo_info} Optimal K: {best_k}. Silhouette Score: {best_score:.3f}. PCA Explained Variance: {[f'{v:.2%}' for v in explained_variance]}. Cluster Sizes: {cluster_counts.to_string()}. \nPreprocessing: Mean Imputer, Mode Imputer+GetDummies, RobustScaler."
                    
                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")

    # --- TAB 4: AI COPILOT CHAT ---
    with tab4:
        st.header("💬 Talk to your AI Copilot")
        st.markdown("Ask the agent to explain the best model, why a specific cluster was chosen, or what mathematical preprocessing happened under the hood.")
        
        # Display chat messages from history
        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                st.chat_message("assistant", avatar="🧠").write(msg.content)

        # Chat Input
        if prompt := st.chat_input("Ask me about the pipeline results..."):
            # Display user prompt
            st.chat_message("user").write(prompt)
            # Add to history
            st.session_state.messages.append(HumanMessage(content=prompt))
            
            # Prepare backend prompt with hidden context
            system_role = SystemMessage(content=(
                "You are the AI Copilot for a Machine Learning application. "
                "Answer the user's questions based on this underlying dataset & pipeline framework context:\n"
                f"Context: {st.session_state['ml_context']}\n\n"
                "Explain things simply, clearly, and mathematically. "
                "The user just asked you a question. Reference their earlier conversation if necessary."
            ))
            
            # Submit to Agent
            with st.spinner("AI is thinking..."):
                full_history_for_llm = [system_role] + st.session_state.messages
                response = chat_with_copilot(full_history_for_llm)
                
            # Display and save response
            st.chat_message("assistant", avatar="🧠").write(response)
            st.session_state.messages.append(AIMessage(content=response))

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin.")
