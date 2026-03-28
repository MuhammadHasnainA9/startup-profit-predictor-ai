"""
AI-Powered Startup Profit Prediction - Streamlit Web Application
This app provides an interactive interface for predicting startup profits
with AI-powered business suggestions and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚀 AI Startup Profit Predictor",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
def set_custom_css():
    st.markdown("""
    <style>
        .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
        .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0; }
        .suggestion-box { background: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 0.5rem 0; }
        .profit-display { font-size: 2rem; font-weight: bold; color: #2ecc71; text-align: center; padding: 1rem; background: #e8f5e8; border-radius: 10px; margin: 1rem 0; }
        .stButton > button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; padding: 0.5rem 2rem; border-radius: 25px; border: none; }
        .stButton > button:hover { background: linear-gradient(90deg, #764ba2 0%, #667eea 100%); }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'train_model.py' first.")
        return None

# ---------------- AI SUGGESTIONS ----------------
def generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, state):
    suggestions = []
    total_spend = rd_spend + admin_spend + marketing_spend

    # R&D Analysis
    if rd_spend < total_spend * 0.3:
        suggestions.append("💡 **Increase R&D Investment**: Innovation drives long-term growth")
    elif rd_spend > total_spend * 0.6:
        suggestions.append("⚖️ **Balance R&D Spending**: Very high R&D investment")

    # Marketing Analysis
    if marketing_spend < total_spend * 0.2:
        suggestions.append("📢 **Boost Marketing Budget**: Low marketing spend may limit growth")
    elif marketing_spend > total_spend * 0.5:
        suggestions.append("🎯 **Optimize Marketing ROI**: High marketing spend")

    # Admin Analysis
    if admin_spend > total_spend * 0.4:
        suggestions.append("🏢 **Reduce Administrative Costs**: High overhead")

    # Balance Analysis
    rd_ratio = rd_spend / total_spend
    marketing_ratio = marketing_spend / total_spend
    if abs(rd_ratio - marketing_ratio) > 0.3:
        suggestions.append("⚖️ **Balance Growth Investments**: Align R&D and marketing spend")

    # State-specific
    state_suggestions = {
        'California': '🌴 **California Strategy**: Focus on tech innovation',
        'New York': '🏙️ **New York Approach**: Leverage diverse opportunities',
        'Florida': '🌊 **Florida Opportunity**: Capitalize on growing market'
    }
    if state in state_suggestions:
        suggestions.append(state_suggestions[state])

    # Total investment analysis
    if total_spend < 100000:
        suggestions.append("💰 **Scaling Strategy**: Modest investment - phased growth")
    elif total_spend > 500000:
        suggestions.append("🚀 **High-Stakes Execution**: Significant investment - focus on ROI")

    # Profit optimization tip
    if rd_spend > 100000 and marketing_spend > 100000:
        suggestions.append("🔄 **Synergy Opportunity**: Strong R&D and marketing")

    if not suggestions:
        suggestions.append("✅ Spending strategy looks balanced.")
    return suggestions

# ---------------- FEATURE IMPORTANCE ----------------
def create_feature_importance_plot(model_data):
    model = model_data['model']
    feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
    if model_data['model_name'] == 'Random Forest':
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(10,6))
        indices = np.argsort(importances)
        bars = ax.barh(range(len(importances)), importances[indices], color='#1f77b4')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance')
        for i, bar in enumerate(bars):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,f'{bar.get_width():.3f}', ha='left', va='center')
        plt.tight_layout()
        return fig
    else:
        coefficients = model.coef_
        fig, ax = plt.subplots(figsize=(10,6))
        indices = np.argsort(np.abs(coefficients))
        colors = ['#1f77b4' if c>0 else '#e74c3c' for c in coefficients[indices]]
        bars = ax.barh(range(len(coefficients)), np.abs(coefficients[indices]), color=colors)
        ax.set_yticks(range(len(coefficients)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Linear Regression Feature Impact')
        for i, bar in enumerate(bars):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,f'{coefficients[indices][i]:.3f}', ha='left', va='center')
        plt.tight_layout()
        return fig

# ---------------- SPENDING CHART ----------------
def create_spending_analysis_chart(rd_spend, admin_spend, marketing_spend):
    categories = ['R&D','Administration','Marketing']
    values = [rd_spend, admin_spend, marketing_spend]
    colors = ['#3498db','#e74c3c','#2ecc71']
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Spending Distribution')
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Amount ($)')
    ax2.set_title('Spending by Category')
    ax2.grid(True, alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    return fig

# ---------------- MAIN APP ----------------
def main():
    set_custom_css()
    model_data = load_model()
    if model_data is None:
        st.stop()

    # HEADER
    st.markdown('<h1 class="main-header">🚀 AI Startup Profit Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict your startup profit with AI-powered insights and recommendations</p>', unsafe_allow_html=True)

    # MODEL METRICS
    col1,col2,col3 = st.columns(3)
    col1.metric("Model", f"{model_data['model_name']}")
    col2.metric("R² Score", f"{model_data['r2_score']:.3f}")
    col3.metric("MAE", f"${model_data['mae']:,.0f}")

    st.divider()

    # INPUTS
    st.sidebar.header("📥 Enter Your Startup Parameters")
    rd_spend = st.sidebar.number_input("💡 R&D Spend ($)", 0, 1000000, 150000, step=5000)
    admin_spend = st.sidebar.number_input("🏢 Administration Spend ($)", 0, 1000000, 120000, step=5000)
    marketing_spend = st.sidebar.number_input("📢 Marketing Spend ($)", 0, 1000000, 80000, step=5000)
    state = st.sidebar.selectbox("🌍 State", ['California','New York','Florida'])

    # SPENDING OVERVIEW
    total_spend = rd_spend + admin_spend + marketing_spend
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Investment", f"${total_spend:,.0f}")
    col2.metric("R&D Ratio", f"{(rd_spend/total_spend)*100:.1f}%")
    col3.metric("Marketing Ratio", f"{(marketing_spend/total_spend)*100:.1f}%")

    fig = create_spending_analysis_chart(rd_spend, admin_spend, marketing_spend)
    st.pyplot(fig)

    # PREDICTION
    if st.button("🔮 Predict Profit"):
        state_encoded = model_data['label_encoder'].transform([state])[0]
        features = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
        if model_data['model_name'] == 'Linear Regression':
            features_scaled = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(features_scaled)[0]
        else:
            prediction = model_data['model'].predict(features)[0]

        st.markdown(f'<div class="profit-display">${prediction:,.2f}</div>', unsafe_allow_html=True)

        # AI suggestions
        suggestions = generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, state)
        for suggestion in suggestions:
            st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)

    # FEATURE IMPORTANCE
    st.subheader("📊 Model Insights")
    fig2 = create_feature_importance_plot(model_data)
    st.pyplot(fig2)

    # DOWNLOAD CSV
    st.subheader("📥 Download Your Report")
    df_report = pd.DataFrame({
        'R&D Spend':[rd_spend], 'Administration':[admin_spend], 'Marketing':[marketing_spend],
        'State':[state], 'Predicted Profit':[prediction if 'prediction' in locals() else 0]
    })
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="startup_report.csv", mime='text/csv')

    # FOOTER
    st.markdown("---")
    st.markdown("🚀 Built with Machine Learning & Streamlit | GitHub Repo")

if __name__ == "__main__":
    main()
