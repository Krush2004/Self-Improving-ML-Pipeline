import streamlit as st
import pandas as pd
import io
import plotly.express as px
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    .metric-card, .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }
    .feature-card { min-height: 260px; }
    .metric-card { min-height: 150px; }
    .feature-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(59, 130, 246, 0.5);
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.2);
    }
    .feature-icon { font-size: 3rem; margin-bottom: 15px; }
    .feature-title { font-size: 1.5rem; font-weight: 700; color: #fff; margin-bottom: 10px; }
    .feature-desc { font-size: 1rem; color: #a1a1aa; line-height: 1.6; }
    .metric-title { font-size: 1.2rem; color: #a1a1aa; margin-bottom: 5px; font-weight: 500; }
    .metric-value { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #3b82f6; 
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .hero-section {
        padding: 60px 0;
        text-align: center;
        background: radial-gradient(circle at top, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        border-radius: 30px;
        margin-bottom: 40px;
    }
    h1 { 
        font-size: 4rem !important; 
        font-weight: 900 !important; 
        background: linear-gradient(90deg, #fff, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem !important;
    }
    h2 { font-size: 2.5rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.8rem !important; font-weight: 600 !important; }
    .stMarkdown p { font-size: 1.15rem; color: #d4d4d8; }
    
    /* Animation for the cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade {
        animation: fadeIn 0.8s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Chat History and ML Context Memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hello! I am your AI Copilot. Upload a dataset and run the ML pipelines, then ask me to explain the models, preprocessing steps, or data patterns!")
    ]

if "ml_context" not in st.session_state:
    st.session_state["ml_context"] = "No pipeline has been run yet. Tell the user to run the Supervised or Unsupervised pipeline first."


# Sidebar Configuration
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file is None:
    st.sidebar.divider()
    st.sidebar.markdown("### 🛠️ Quick Start")
    st.sidebar.info("1. **Upload** your dataset (.csv)\n2. **Choose** a target for ML\n3. **Analyze** with AI Analyst")

    st.sidebar.divider()
    st.sidebar.caption("v2.1.0 • Built with Pinecone & Langchain")

if uploaded_file is not None:
    st.markdown("""
    <div style='background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%); padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6; margin-bottom: 25px;'>
        <h1 style='font-size: 2.5rem !important; margin: 0;'>🧠 Multi-AI Agent Workstation</h1>
        <p style='margin: 5px 0 0 0; color: #a1a1aa;'>Advanced ML Pipeline & Agentic Self-Critique System</p>
    </div>
    """, unsafe_allow_html=True)
    df = pd.read_csv(uploaded_file)
    
    # Populate the Sidebar with Stats to keep it active
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Dataset Dashboard")
    
    # Custom Sidebar Metric Cards
    st.sidebar.markdown(f"""
    <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #3b82f6;'>
        <p style='margin:0; font-size: 0.9rem; color: #a1a1aa;'>Total Rows</p>
        <p style='margin:0; font-size: 1.2rem; font-weight: bold;'>🔢 {df.shape[0]:,}</p>
    </div>
    <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #10b981;'>
        <p style='margin:0; font-size: 0.9rem; color: #a1a1aa;'>Total Columns</p>
        <p style='margin:0; font-size: 1.2rem; font-weight: bold;'>📐 {df.shape[1]}</p>
    </div>
    <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #f59e0b;'>
        <p style='margin:0; font-size: 0.9rem; color: #a1a1aa;'>Missing Values</p>
        <p style='margin:0; font-size: 1.2rem; font-weight: bold;'>⚠️ {df.isnull().sum().sum():,}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("🔍 Detailed Column Types", expanded=False):
        st.dataframe(df.dtypes.astype(str).rename("Type"), use_container_width=True)
    
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
        target_col = st.selectbox("Select Target Column", options=df.columns, index=len(df.columns)-1)
        
        # Feature Selection
        feature_options = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Select Feature Columns", options=feature_options, default=feature_options, help="Choose which columns the model should use for learning.")
        
        # Dual Buttons for Basic vs Advanced
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            run_basic = st.button("🚀 Run Baseline Pipeline", type="primary", use_container_width=True, disabled=len(selected_features) == 0)
        with c_btn2:
            run_advanced = st.button("🔥 Auto-Tune & Apply AI Suggestions (Slower)", type="secondary", use_container_width=True, disabled=len(selected_features) == 0)
            
        if len(selected_features) == 0:
            st.warning("⚠️ Please select at least one feature column to continue training models.")
            
        if (run_basic or run_advanced) and len(selected_features) > 0:
            if run_advanced:
                st.info("Applying AI Suggestions: SMOTE Class Balancing and RandomizedSearchCV Hyperparameter Tuning initialized...")
                
            spinner_msg = "Hyper-Tuning Models (This might take 30-60s)..." if run_advanced else "Training Baseline Models..."
            with st.spinner(spinner_msg):
                try:
                    X_scaled, y, task_type, label_encoder, preprocessors = preprocess_data(df, target_col, feature_cols=selected_features)
                    
                    results_df, best_model_name, best_model, trained_models, X_test, y_test, tuning_info = train_and_evaluate(X_scaled, y, task_type, apply_improvements=run_advanced)
                    
                    st.success(f"Pipeline Execution Complete! Identified Task: **{task_type.capitalize()}**")
                    st.markdown(f"### Best Performing Model: `{best_model_name}`")
                    
                    if task_type == 'classification':
                        # Sort by Accuracy descending
                        results_df = results_df.sort_values(by='Accuracy', ascending=False)
                        best_row = results_df.iloc[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>Accuracy</div><div class='metric-value'>{best_row['Accuracy']:.4f}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>Precision</div><div class='metric-value'>{best_row['Precision']:.4f}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>Recall</div><div class='metric-value'>{best_row['Recall']:.4f}</div></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='metric-card'><div class='metric-title'>F1 Score</div><div class='metric-value'>{best_row['F1 Score']:.4f}</div></div>", unsafe_allow_html=True)
                        
                        fig = px.bar(results_df, x='Model', y='Accuracy', color='Model', title="Model Accuracy Comparison")
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Sort by RMSE ascending (lower is better)
                        results_df = results_df.sort_values(by='RMSE', ascending=True)
                        best_row = results_df.iloc[0]
                        c1, c2, c3, c4 = st.columns(4)
                        # Use commas for thousands in large error values
                        c1.markdown(f"<div class='metric-card'><div class='metric-title'>RMSE</div><div class='metric-value'>{best_row['RMSE']:,.2f}</div></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='metric-card'><div class='metric-title'>MSE</div><div class='metric-value'>{best_row['MSE']:,.0f}</div></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='metric-card'><div class='metric-title'>MAE</div><div class='metric-value'>{best_row['MAE']:,.2f}</div></div>", unsafe_allow_html=True)
                        c4.markdown(f"<div class='metric-card'><div class='metric-title'>R2 Score</div><div class='metric-value'>{best_row['R2 Score']:.3f}</div></div>", unsafe_allow_html=True)
                        
                        fig = px.bar(results_df, x='Model', y='RMSE', color='Model', title="Model RMSE Comparison (Lower is Better)")
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if task_type == 'classification':
                        styled_df = results_df.round(4).style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
                    else:
                        # RdYlGn_r reverses the colormap so lower error = green
                        styled_df = results_df.round(2).style.background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MSE', 'MAE'])
                        
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
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
                            
                    # Preview Preprocessed Data
                    with st.expander("📦 View Preprocessed Features (After Transformations)", expanded=False):
                        st.markdown("**Note:** This shows the data after imputation, encoding, and RobustScaling.")
                        st.dataframe(X_scaled.head(100), use_container_width=True)
                        st.caption(f"Showing first 100 rows of {X_scaled.shape[1]} total features.")
                    
                    # --- IMPROVEMENT SUGGESTIONS ---
                    st.divider()
                    tuning_summary = "\n".join([
                        f"- {name}: {'Auto-Tuned with params ' + str(info.get('best_params', {})) if info.get('tuned') else 'Default params'}"
                        for name, info in tuning_info.items()
                    ])
                    
                    with st.spinner("Generating Agent Critique based on results..."):
                        # Add a metric guide to prevent AI hallucination on RMSE/Error metrics
                        metric_guide = "NOTE: For RMSE, MSE, and MAE, LOWER values are BETTER. For Accuracy and R2 Score, HIGHER values are BETTER."
                        critique = self_critique_models(
                            f"{metric_guide}\n\n{results_df.to_string()}\n\nParameters Used:\n{tuning_summary}",
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
                    st.session_state["trained_preprocessed_df"] = X_scaled # Store for preview
                    st.session_state["trained_target_col"] = target_col
                    st.session_state["trained_selected_features"] = selected_features
                    st.session_state["trained_preprocessors"] = preprocessors # Key fix: store memory
                    
                    # Update Chat Context Memory
                    metric_guide_context = "CRITICAL: In this dataset, RMSE/MSE/MAE are error metrics (Lower is Better). Accuracy/R2 are performance metrics (Higher is Better)."
                    st.session_state["ml_context"] = f"{metric_guide_context} Supervised Pipeline Run for '{target_col}'. Advanced Auto-Tuning Applied: {run_advanced}. Best Model: {best_model_name}. \nResults:\n{results_df.to_string()} \n\nParameters Used:\n{tuning_summary}\n\nCritique:\n{critique}"
                        
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
            selected_features_saved = st.session_state["trained_selected_features"]
            
            # Get original feature columns (those selected during training)
            original_features = selected_features_saved
            
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
                            if col_data.dtype == 'int64':
                                # Integer: show number input without decimals
                                col_min = int(col_data.min()) if not pd.isna(col_data.min()) else 0
                                col_max = int(col_data.max()) if not pd.isna(col_data.max()) else 100
                                col_mean = int(col_data.mean()) if not pd.isna(col_data.mean()) else 0
                                input_values[col_name] = st.number_input(
                                    f"{col_name}", 
                                    min_value=col_min, max_value=col_max, value=col_mean,
                                    step=1,
                                    help=f"Range: {col_min} – {col_max} (Integer)"
                                )
                            elif col_data.dtype == 'float64':
                                # Numeric: show number input with decimals
                                col_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                                col_max = float(col_data.max()) if not pd.isna(col_data.max()) else 100.0
                                col_mean = float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0
                                input_values[col_name] = st.number_input(
                                    f"{col_name}", 
                                    min_value=col_min, max_value=col_max, value=col_mean,
                                    help=f"Range: {col_min:.2f} – {col_max:.2f} (Float)"
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
                    # Retrieve saved preprocessors
                    prep = st.session_state["trained_preprocessors"]
                    input_df = pd.DataFrame([input_values])
                    
                    # 1. Impute Nums
                    if prep['num_imputer']:
                        input_df[prep['num_cols']] = prep['num_imputer'].transform(input_df[prep['num_cols']])
                    
                    # 2. Impute & Manual One-Hot Encode
                    if prep['cat_imputer']:
                        # Impute first
                        input_df[prep['cat_cols']] = prep['cat_imputer'].transform(input_df[prep['cat_cols']])
                        
                        # Instead of get_dummies (which fails on single rows with drop_first=True),
                        # manually construct the dummy columns expected by the model.
                        for cat_col in prep['cat_cols']:
                            current_val = str(input_df[cat_col].iloc[0])
                            # Identify dummy columns in 'final_columns' matching this feature
                            for final_col in prep['final_columns']:
                                if final_col.startswith(f"{cat_col}_"):
                                    # If the dummy column name suffix matches our current value, set 1, else 0
                                    # This correctly handles the 'drop_first' case (if value was dropped, none will match)
                                    suffix = final_col[len(f"{cat_col}_"):]
                                    input_df[final_col] = 1 if suffix == current_val else 0
                        
                        # Drop original string columns
                        input_df = input_df.drop(columns=prep['cat_cols'])
                    
                    # 3. Align order and handle missing (remaining) columns
                    for col in prep['final_columns']:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[prep['final_columns']]
                    
                    # 4. Scale
                    input_df = pd.DataFrame(prep['scaler'].transform(input_df), columns=input_df.columns)
                    
                    # 5. Predict
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
        st.write("Find hidden patterns. Choose **Baseline** for quick KMeans or **Auto-Tune** to compare 5 algorithms with hyperparameter optimization.")
        
        # Feature Selection for Clustering
        selected_unsupervised_features = st.multiselect("Select Columns for Clustering", options=df.columns.tolist(), default=df.columns.tolist(), help="Choose which features should be used to group the data.")
        
        # Dual Buttons
        uc1, uc2 = st.columns(2)
        with uc1:
            run_baseline_clust = st.button("🧩 Run Baseline KMeans", type="primary", use_container_width=True, disabled=len(selected_unsupervised_features) == 0)
        with uc2:
            run_autotune_clust = st.button("🔥 Auto-Tune All Algorithms (Slower)", type="secondary", use_container_width=True, disabled=len(selected_unsupervised_features) == 0)
        
        if len(selected_unsupervised_features) == 0:
            st.warning("⚠️ Please select at least one column for clustering analysis.")
            
        if run_baseline_clust or run_autotune_clust:
            spinner_msg = "Auto-Tuning 5 Clustering Algorithms..." if run_autotune_clust else "Finding Optimal K for KMeans..."
            with st.spinner(spinner_msg):
                try:
                    X_scaled = preprocess_unsupervised(df, feature_cols=selected_unsupervised_features)
                    
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
    # --- PREMIUM LANDING PAGE (NO FILE UPLOADED) ---
    st.markdown("""
    <div class='hero-section animate-fade'>
        <h1>🤖 Self-Improving AI Multi-Agent Workstation</h1>
        <h3 style='margin-bottom: 1rem;'>The ultimate machine learning workstation that builds, tunes, and critiques itself.</h3>
        <p>Unlock the power of automated ML pipelines with vector memory and agentic self-critique.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Grid
    st.markdown("<h2 style='text-align: center; margin-bottom: 40px;'>Exploration Unleashed</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='feature-card animate-fade'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-title'>Supervised ML</div>
            <div class='feature-desc'>
                Train 10+ algorithms. Advanced <b>Auto-Tuning</b> and class balancing find the absolute best model for your data.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='feature-card animate-fade' style='animation-delay: 0.2s;'>
            <div class='feature-icon'>🧩</div>
            <div class='feature-title'>Unsupervised Discovery</div>
            <div class='feature-desc'>
                Find hidden patterns automatically. Compare multiple algorithms with interactive <b>3D visualizations</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='feature-card animate-fade' style='animation-delay: 0.4s;'>
            <div class='feature-icon'>🧠</div>
            <div class='feature-title'>AI Copilot & Critique</div>
            <div class='feature-desc'>
                Chat with an AI that knows your data. Get <b>Self-Critique</b> reports and mathematical improvement tips.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # "How it Works" or "Why it Matters" section
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 🚀 Why this Workstation?")
        st.markdown("""
        - **Vector Memory Enabled**: Remembers past results to avoid repeating mistakes.
        - **Automated Preprocessing**: Handles outliers, missing values, and high correlation automatically.
        - **Production-Ready**: Export cleaned datasets and test models with live prediction forms.
        """)
    with c2:
        st.info("👈 **Ready to start?** Upload a CSV file in the sidebar to unlock the full potential of your data.")
        st.markdown("""
        <div style='background: rgba(59, 130, 146, 0.1); padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6;'>
            <b>Pro Tip:</b> Use the "Auto-Tune" buttons in the tabs once you upload a file for the most powerful results.
        </div>
        """, unsafe_allow_html=True)
