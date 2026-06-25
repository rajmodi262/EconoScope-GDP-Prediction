# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from pharma_analytics import render_pharma_tab

# ----------------------------
# PAGE & STYLES
# ----------------------------
st.set_page_config(page_title="🌍 Global GDP Predictor", page_icon="🌏", layout="wide")

# Modern, vibrant design with better color grading and full-width layout
st.markdown(
    """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Header styles */
    .app-header {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(120deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        color: #a0aec0;
        font-size: 15px;
        margin-bottom: 24px;
        font-weight: 400;
    }
    
    /* Card styles with glassmorphism */
    .card {
        background: rgba(25, 32, 56, 0.6);
        backdrop-filter: blur(10px);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .card-compact {
        background: rgba(25, 32, 56, 0.6);
        backdrop-filter: blur(10px);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
    }
    
    .metric-label {
        color: #10b981;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Section headers */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-title::before {
        content: '';
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #00d4ff 0%, #7b2ff7 100%);
        border-radius: 2px;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Button styling override */
    .stButton > button {
        background: linear-gradient(120deg, #00d4ff 0%, #7b2ff7 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 16px rgba(123, 47, 247, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(123, 47, 247, 0.4);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        color: #a0aec0;
        padding: 12px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
        color: #ffffff;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    
    /* Compact spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    /* Input label styling */
    .stNumberInput label {
        color: #e2e8f0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="app-header">📊 EconoScope — Economic & Life-Sciences Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">US biopharma commercial analytics • ML-driven forecasting • patent-cliff &amp; competitive intelligence • real-time scenario analysis</div>', unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS & ARTIFACTS
# ----------------------------
models_dir = "models"
if not os.path.exists(models_dir):
    st.error("⚠️ models/ directory not found. Run the notebook training cell first to create models/.")
    st.stop()

# Load mandatory artifacts with safe guards
def load_safe(path, default=None):
    try:
        return joblib.load(path)
    except Exception:
        return default

encoder = load_safe(os.path.join(models_dir, "encoder.pkl"))
scaler = load_safe(os.path.join(models_dir, "scaling_pipeline.pkl"))
df_imputed = load_safe(os.path.join(models_dir, "df_imputed.pkl"))
columns_index = load_safe(os.path.join(models_dir, "X_columns.pkl"))
metrics_csv_path = os.path.join(models_dir, "model_metrics.csv")
metrics_df = pd.read_csv(metrics_csv_path) if os.path.exists(metrics_csv_path) else pd.DataFrame()
log_flag = load_safe(os.path.join(models_dir, "log_target_flag.pkl"), False)

# Load models
models = {}
for f in os.listdir(models_dir):
    if f.endswith(".pkl") and f not in ("encoder.pkl", "scaling_pipeline.pkl", "df_imputed.pkl", "X_columns.pkl", "log_target_flag.pkl"):
        try:
            name = f.replace(".pkl", "").replace("_", " ").title()
            models[name] = joblib.load(os.path.join(models_dir, f))
        except Exception:
            continue

if df_imputed is None:
    st.error("⚠️ Could not load df_imputed.pkl — preprocessing reference missing.")
    st.stop()

# ----------------------------
# HELPERS
# ----------------------------
def format_gdp_full(x):
    """Return full USD number with commas and no decimals."""
    try:
        if x >= 1e12:
            return f"${x/1e12:.2f}T"
        elif x >= 1e9:
            return f"${x/1e9:.2f}B"
        elif x >= 1e6:
            return f"${x/1e6:.2f}M"
        else:
            return f"${int(x):,}"
    except Exception:
        return str(x)

def safe_predict(model, X_df):
    """Return real-scale GDP prediction."""
    pred = model.predict(X_df)[0]
    if log_flag:
        pred = np.expm1(np.clip(pred, -20, 36))
    return float(pred)

# Define the 25 selected features
SELECTED_FEATURES = [
    'Official exchange rate (LCU per US$, period average)',
    'Households and NPISHs Final consumption expenditure (current US$)',
    'Services, value added (current US$)',
    'Imports of goods and services (current US$)',
    'Agriculture, forestry, and fishing, value added (current US$)',
    'Individuals using the Internet (% of population)',
    'Exports of goods and services (current US$)',
    'Gross capital formation (% of GDP)',
    'Industry (including construction), value added (current US$)',
    'School enrollment, primary and secondary (gross), gender parity index (GPI)',
    'Life expectancy at birth, total (years)',
    'Population, total',
    'General government final consumption expenditure (current US$)',
    'Urban population',
    'Mobile cellular subscriptions (per 100 people)',
    'Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)',
    'GDP deflator (base year varies by country)',
    'External debt stocks, total (DOD, current US$)',
    'Foreign direct investment, net inflows (% of GDP)',
    'Unemployment, total (% of total labor force) (modeled ILO estimate)',
    'Gross capital formation (current US$)',
    'Current account balance (BoP, current US$)'
]

# ----------------------------
# MAIN LAYOUT - TABS
# ----------------------------
tab_pharma, tab1, tab2, tab3 = st.tabs([
    "💊 Pharma Commercial Analytics",
    "🎯 GDP Prediction Dashboard",
    "📈 Model Performance",
    "⚙️ System Information",
])

with tab_pharma:
    render_pharma_tab()

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Scenario Configuration</div>', unsafe_allow_html=True)
    
    # Top row: Country and Scenario name
    col_select1, col_select2, col_select3 = st.columns([2, 2, 1])
    with col_select1:
        country = st.selectbox("Select Country", sorted(df_imputed["Country Name"].unique()), 
                              help="Choose a country to analyze")
    
    with col_select2:
        scenario_name = st.text_input("Scenario Name", value=f"{country} Analysis", 
                                     placeholder="Name your scenario...")
    
    with col_select3:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="status-badge">✓ 25 Features</div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Get baseline data
    baseline_row = df_imputed[df_imputed["Country Name"] == country].iloc[-1].drop(["GDP (current US$)"], errors="ignore")
    
    # Display all 25 indicators in a grid layout
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Economic Indicators (25 Selected Features)</div>', unsafe_allow_html=True)
    
    # Create a grid of 5 columns for indicators
    edits = {}
    cols_per_row = 5
    available_features = [f for f in SELECTED_FEATURES if f in baseline_row.index]
    
    for i in range(0, len(available_features), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(available_features):
                indicator = available_features[idx]
                with col:
                    value = float(baseline_row[indicator]) if pd.notna(baseline_row[indicator]) else 0.0
                    # Shorten label for display
                    short_label = indicator[:30] + "..." if len(indicator) > 30 else indicator
                    edits[indicator] = st.number_input(
                        short_label,
                        value=value,
                        format="%.4f",
                        key=f"ind_{idx}",
                        help=indicator
                    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action button
    st.markdown('<div class="card-compact">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns([3, 1, 1, 1, 1])
    
    with col_btn1:
        run_predict = st.button("🚀 Run Prediction Models", use_container_width=True, type="primary")
    
    with col_btn2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    
    with col_btn3:
        st.button("💾 Save", use_container_width=True, disabled=True, help="Coming soon")
    
    with col_btn4:
        st.button("📁 Load", use_container_width=True, disabled=True, help="Coming soon")
    
    with col_btn5:
        st.button("📊 Compare", use_container_width=True, disabled=True, help="Coming soon")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------
    # PREDICTION RESULTS SECTION
    # ----------------------------
    if run_predict:
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 GDP Prediction Results</div>', unsafe_allow_html=True)
        
        # Build input
        numeric_cols = [c for c in df_imputed.columns if c not in ["Country Name", "GDP (current US$)"]]
        input_series = baseline_row[numeric_cols].copy()
        
        # Apply edits
        for k, v in edits.items():
            if k in input_series.index:
                input_series[k] = v
        
        # Preprocess
        encoded_country = None
        if encoder is not None:
            try:
                encoded_country = pd.DataFrame(encoder.transform([[country]]), 
                                             columns=encoder.get_feature_names_out(["Country Name"]))
            except Exception:
                pass
        
        scaled_df = None
        if scaler is not None:
            try:
                to_scale = pd.DataFrame([input_series[numeric_cols]])
                scaled_arr = scaler.transform(to_scale)
                scaled_df = pd.DataFrame(scaled_arr, columns=scaler.get_feature_names_out(), index=[0])
            except Exception:
                pass
        
        # Combine
        if scaled_df is not None and encoded_country is not None:
            model_input = pd.concat([scaled_df, encoded_country], axis=1)
            model_input.columns = model_input.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True)
            if columns_index is not None:
                try:
                    model_input = model_input.reindex(columns=list(columns_index), fill_value=0)
                except Exception:
                    pass
        else:
            model_input = pd.DataFrame([input_series[numeric_cols]])
        
        # Run predictions
        if models:
            results = []
            for name, model in models.items():
                try:
                    pred_val = safe_predict(model, model_input)
                    results.append({
                        "Model": name,
                        "Predicted_GDP_USD": pred_val,
                        "Display": format_gdp_full(pred_val)
                    })
                except Exception:
                    pass
            
            if results:
                results_df = pd.DataFrame(results).sort_values("Predicted_GDP_USD", ascending=False)
                
                # Merge metrics
                if not metrics_df.empty:
                    metrics_copy = metrics_df.copy()
                    metrics_copy["Model"] = metrics_copy["Model"].astype(str)
                    results_df = results_df.merge(metrics_copy, on="Model", how="left")
                
                # Top 3 models in metric cards
                col_top1, col_top2, col_top3, col_top4 = st.columns([1, 1, 1, 2])
                
                for idx, col in enumerate([col_top1, col_top2, col_top3]):
                    if idx < len(results_df):
                        row = results_df.iloc[idx]
                        medal = ["🥇", "🥈", "🥉"][idx]
                        with col:
                            st.markdown(f"""
                            <div class='metric-card'>
                                <div class='metric-label'>{medal} {row['Model']}</div>
                                <div class='metric-value'>{row['Display']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Comparison with baseline
                with col_top4:
                    try:
                        top_model = models[results_df.iloc[0]["Model"]]
                        baseline_to_scale = pd.DataFrame([baseline_row[numeric_cols]])
                        if scaler is not None:
                            base_scaled = scaler.transform(baseline_to_scale)
                            base_scaled_df = pd.DataFrame(base_scaled, columns=scaler.get_feature_names_out())
                            base_input = pd.concat([base_scaled_df, encoded_country], axis=1) if encoded_country is not None else base_scaled_df
                        else:
                            base_input = baseline_to_scale
                        
                        if columns_index is not None:
                            try:
                                base_input = base_input.reindex(columns=list(columns_index), fill_value=0)
                            except Exception:
                                pass
                        
                        base_pred = safe_predict(top_model, base_input)
                        new_pred = float(results_df.iloc[0]["Predicted_GDP_USD"])
                        change_pct = (new_pred - base_pred) / base_pred * 100 if base_pred != 0 else 0
                        
                        st.markdown(f"""
                        <div style='background: rgba(123, 47, 247, 0.1); border: 1px solid rgba(123, 47, 247, 0.3); border-radius: 12px; padding: 16px;'>
                            <div style='font-size: 13px; color: #7b2ff7; font-weight: 600; margin-bottom: 8px;'>CHANGE FROM BASELINE</div>
                            <div style='font-size: 28px; color: #ffffff; font-weight: 700;'>{change_pct:+.2f}%</div>
                            <div style='font-size: 12px; color: #a0aec0; margin-top: 4px;'>Baseline: {format_gdp_full(base_pred)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception:
                        pass
                
                # Detailed results
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    st.markdown('<div class="section-title">📋 All Model Predictions</div>', unsafe_allow_html=True)
                    display_cols = ["Model", "Display"]
                    if "GDP_R2" in results_df.columns:
                        results_df["R²"] = results_df["GDP_R2"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
                        display_cols.append("R²")
                    if "GDP_RMSE" in results_df.columns:
                        results_df["RMSE"] = results_df["GDP_RMSE"].map(lambda x: format_gdp_full(x) if pd.notna(x) else "")
                        display_cols.append("RMSE")
                    
                    st.dataframe(
                        results_df[display_cols].rename(columns={"Display": "Predicted GDP"}),
                        use_container_width=True,
                        height=350
                    )
                
                with col_res2:
                    st.markdown('<div class="section-title">📈 Performance Overview</div>', unsafe_allow_html=True)
                    
                    if "GDP_R2" in results_df.columns:
                        avg_r2 = results_df["GDP_R2"].mean()
                        max_r2 = results_df["GDP_R2"].max()
                        min_r2 = results_df["GDP_R2"].min()
                        
                        st.metric("Average R²", f"{avg_r2:.3f}")
                        st.metric("Best R²", f"{max_r2:.3f}")
                        st.metric("Range", f"{max_r2 - min_r2:.3f}")
                    
                    if "GDP_RMSE" in results_df.columns:
                        avg_rmse = results_df["GDP_RMSE"].mean()
                        st.metric("Average RMSE", format_gdp_full(avg_rmse))
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Adjust the economic indicators above and click 'Run Prediction Models' to see GDP forecasts from all trained models.")

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
    
    if not metrics_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            if "GDP_R2" in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                ax.set_facecolor('#0a0e27')
                sorted_df = metrics_df.sort_values("GDP_R2", ascending=True)
                bars = ax.barh(sorted_df["Model"], sorted_df["GDP_R2"], color='#10b981')
                ax.set_xlabel("R² Score", color='white', fontsize=12, fontweight='600')
                ax.set_title("Model R² Performance", color='white', fontsize=14, fontweight='bold', pad=15)
                ax.tick_params(colors='white', labelsize=10)
                ax.spines['bottom'].set_color('#ffffff')
                ax.spines['left'].set_color('#ffffff')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.2, color='white', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            if "GDP_RMSE" in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                ax.set_facecolor('#0a0e27')
                sorted_df = metrics_df.sort_values("GDP_RMSE", ascending=True)
                bars = ax.barh(sorted_df["Model"], sorted_df["GDP_RMSE"] / 1e9, color='#06b6d4')
                ax.set_xlabel("RMSE (Billion USD)", color='white', fontsize=12, fontweight='600')
                ax.set_title("Model Error (RMSE)", color='white', fontsize=14, fontweight='bold', pad=15)
                ax.tick_params(colors='white', labelsize=10)
                ax.spines['bottom'].set_color('#ffffff')
                ax.spines['left'].set_color('#ffffff')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', alpha=0.2, color='white', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.markdown("---")
        st.markdown('<div class="section-title">📈 Complete Metrics Table</div>', unsafe_allow_html=True)
        display_metrics = metrics_df.copy()
        for col in ["GDP_RMSE", "GDP_MAE"]:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].map(lambda x: format_gdp_full(x) if pd.notna(x) else "")
        st.dataframe(display_metrics, use_container_width=True, height=400)
        
        # Additional insights
        st.markdown("---")
        st.markdown('<div class="section-title">🎯 Model Insights</div>', unsafe_allow_html=True)
        
        col_ins1, col_ins2, col_ins3, col_ins4 = st.columns(4)
        
        if "GDP_R2" in metrics_df.columns:
            best_model = metrics_df.loc[metrics_df["GDP_R2"].idxmax(), "Model"]
            best_r2 = metrics_df["GDP_R2"].max()
            with col_ins1:
                st.metric("Best R² Model", best_model, f"{best_r2:.3f}")
        
        if "GDP_RMSE" in metrics_df.columns:
            lowest_error = metrics_df.loc[metrics_df["GDP_RMSE"].idxmin(), "Model"]
            lowest_rmse = metrics_df["GDP_RMSE"].min()
            with col_ins2:
                st.metric("Lowest RMSE Model", lowest_error, format_gdp_full(lowest_rmse))
        
        with col_ins3:
            st.metric("Total Models", len(metrics_df))
        
        if "GDP_R2" in metrics_df.columns:
            avg_r2 = metrics_df["GDP_R2"].mean()
            with col_ins4:
                st.metric("Average R²", f"{avg_r2:.3f}")
    else:
        st.warning("⚠️ No metrics available. Train models first.")
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ System Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Loaded", len(models))
    with col2:
        st.metric("Countries Available", len(df_imputed["Country Name"].unique()) if df_imputed is not None else 0)
    with col3:
        st.metric("Total Features", len(df_imputed.columns) - 2 if df_imputed is not None else 0)
    with col4:
        st.metric("Selected Features", len(SELECTED_FEATURES))
    
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Feature Information</div>', unsafe_allow_html=True)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("**Selected Features for Prediction:**")
        for i, feature in enumerate(SELECTED_FEATURES, 1):
            st.markdown(f"{i}. {feature}")
    
    with col_feat2:
        st.markdown("**Model Details:**")
        if models:
            for model_name in models.keys():
                st.markdown(f"✓ {model_name}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Data Processing:**")
        st.markdown(f"- Encoder: {'✓ Loaded' if encoder is not None else '✗ Not loaded'}")
        st.markdown(f"- Scaler: {'✓ Loaded' if scaler is not None else '✗ Not loaded'}")
        st.markdown(f"- Log Transform: {'✓ Enabled' if log_flag else '✗ Disabled'}")
    
    st.markdown("---")
    st.markdown('<div class="section-title">ℹ️ About This Application</div>', unsafe_allow_html=True)
    st.markdown("""
    This GDP prediction dashboard uses machine learning models trained on World Bank economic indicators. 
    
    **Features:**
    - 🎯 Real-time scenario analysis with 25 selected economic indicators
    - 📊 Multiple model comparison (Linear Regression, Random Forest, XGBoost, etc.)
    - 📈 Interactive parameter adjustment with full-width grid layout
    - 💡 Automatic log-scale conversion for accurate USD predictions
    - 🔍 Comprehensive performance metrics and visualizations
    
    **Workflow:**
    1. Select a country and adjust economic indicators in the Prediction Dashboard
    2. Click "Run Prediction Models" to generate GDP forecasts
    3. Compare predictions from multiple models with baseline scenarios
    4. Analyze model performance in the Model Performance tab
    
    **Technical Details:**
    - Models are trained on historical World Bank data
    - Predictions use 25 carefully selected features for optimal accuracy
    - Log transformation is applied where appropriate and automatically converted back to USD
    - All monetary values are displayed in readable formats (T/B/M suffixes)
    
    **Note:** Models trained on log(1+GDP) are automatically converted to real USD values for display.
    """)