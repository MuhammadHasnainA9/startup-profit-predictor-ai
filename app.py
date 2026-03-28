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
import warnings
warnings.filterwarnings('ignore')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚀 Startup Profit Predictor",
    layout="wide",
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
def set_custom_css():
    st.markdown("""
    <style>
        .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
        .sub-header { font-size: 1.3rem; color: #555; text-align: center; margin-bottom: 2rem; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0; font-size:1.2rem; text-align:center; }
        .suggestion-box { background: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 6px solid #1f77b4; margin: 0.5rem 0; font-size:1.1rem; }
        .profit-display { font-size: 3rem; font-weight: bold; color: #2ecc71; text-align: center; padding: 1.2rem; background: #e8f5e8; border-radius: 12px; margin: 1rem 0; }
        .stButton > button { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; padding: 0.7rem 2.5rem; border-radius: 25px; border: none; font-size:1.2rem; }
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
        st.error("❌ Model file not found! Please train your model first.")
        return None

# ---------------- SIMPLE AI SUGGESTIONS ----------------
def generate_ai_suggestions(rd, admin, marketing, location):
    suggestions = []
    total = rd + admin + marketing

    if rd < total * 0.3:
        suggestions.append("💡 Spend more on R&D to grow your business.")
    elif rd > total * 0.6:
        suggestions.append("⚖️ Too much R&D spend, balance it with other costs.")

    if marketing < total * 0.2:
        suggestions.append("📢 Increase marketing to reach more customers.")
    elif marketing > total * 0.5:
        suggestions.append("🎯 Marketing is high, check if it's giving good results.")

    if admin > total * 0.4:
        suggestions.append("🏢 Admin costs are high. Try to reduce overhead.")

    # Balance
    rd_ratio = rd / total
    marketing_ratio = marketing / total
    if abs(rd_ratio - marketing_ratio) > 0.3:
        suggestions.append("⚖️ Balance R&D and marketing spending for better results.")

    # Location-specific
    location_suggestions = {
        'Pakistan - Sindh': '🌴 Focus on tech hubs like Karachi.',
        'Pakistan - Punjab': '🏙️ Focus on business opportunities in Lahore.',
        'Pakistan - Balochistan': '🌄 Explore untapped local markets.',
        'Pakistan - Khyber Pakhtunkhwa': '🏞️ Small businesses can grow here.',
        'Pakistan - Islamabad Capital Territory': '🏛️ Work with government and companies.'
    }
    if location in location_suggestions:
        suggestions.append(location_suggestions[location])

    if total < 100000:
        suggestions.append("💰 Small investment, grow slowly step by step.")
    elif total > 500000:
        suggestions.append("🚀 Big investment! Focus on getting good returns.")

    if rd > 100000 and marketing > 100000:
        suggestions.append("🔄 Strong R&D and marketing together can boost profit.")

    if not suggestions:
        suggestions.append("✅ Your spending looks balanced.")

    return suggestions

# ---------------- FEATURE IMPORTANCE ----------------
def create_feature_plot(model_data):
    model = model_data['model']
    feature_names = ['R&D', 'Admin', 'Marketing', 'Location']
    if model_data['model_name'] == 'Random Forest':
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(10,6))
        indices = np.argsort(importances)
        bars = ax.barh(range(len(importances)), importances[indices], color='#1f77b4')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Which input matters most?')
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
        ax.set_xlabel('Effect size')
        ax.set_title('How each input affects profit')
        for i, bar in enumerate(bars):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,f'{coefficients[indices][i]:.3f}', ha='left', va='center')
        plt.tight_layout()
        return fig

# ---------------- SPENDING CHART ----------------
def create_spending_chart(rd, admin, marketing):
    categories = ['R&D','Admin','Marketing']
    values = [rd, admin, marketing]
    colors = ['#3498db','#e74c3c','#2ecc71']
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,6))
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':14})
    ax1.set_title('Your spending ratio', fontsize=16)
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_ylabel('Amount ($)', fontsize=14)
    ax2.set_title('Spending by type', fontsize=16)
    ax2.grid(True, alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    return fig

# ---------------- MAIN APP ----------------
def main():
    set_custom_css()
    model_data = load_model()
    if model_data is None:
        st.stop()

    # HEADER
    st.markdown('<h1 class="main-header">🚀 Startup Profit Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter your numbers below and see predicted profit with simple tips</p>', unsafe_allow_html=True)

    # MODEL METRICS
    col1,col2,col3 = st.columns(3)
    col1.metric("Model Used", f"{model_data['model_name']}")
    col2.metric("R² Score", f"{model_data['r2_score']:.2f}")
    col3.metric("MAE", f"${model_data['mae']:,.0f}")

    st.divider()

    # SIDEBAR INPUTS
    st.sidebar.header("📥 Enter your details")
    rd_spend = st.sidebar.number_input("💡 R&D Spend ($)", 0, 1000000, 150000, step=5000)
    admin_spend = st.sidebar.number_input("🏢 Admin Spend ($)", 0, 1000000, 120000, step=5000)
    marketing_spend = st.sidebar.number_input("📢 Marketing Spend ($)", 0, 1000000, 80000, step=5000)

    locations = [
        'Pakistan - Sindh',
        'Pakistan - Punjab',
        'Pakistan - Balochistan',
        'Pakistan - Khyber Pakhtunkhwa',
        'Pakistan - Islamabad Capital Territory'
    ]
    location = st.sidebar.selectbox("🌍 Choose your province", locations)

    # SPENDING OVERVIEW
    total_spend = rd_spend + admin_spend + marketing_spend
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Investment 💰", f"${total_spend:,.0f}")
    col2.metric("R&D Ratio 📊", f"{(rd_spend/total_spend)*100:.1f}%")
    col3.metric("Marketing Ratio 📈", f"{(marketing_spend/total_spend)*100:.1f}%")

    fig = create_spending_chart(rd_spend, admin_spend, marketing_spend)
    st.pyplot(fig)

    # PREDICTION BUTTON
    if st.button("🔮 Predict Profit"):
        try:
            state_encoded = model_data['label_encoder'].transform([location])[0]
        except ValueError:
            st.warning(f"⚠️ Location '{location}' not in model training. Using default 0.")
            state_encoded = 0

        features = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
        if model_data['model_name'] == 'Linear Regression':
            features_scaled = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(features_scaled)[0]
        else:
            prediction = model_data['model'].predict(features)[0]

        st.markdown(f'<div class="profit-display">💵 ${prediction:,.2f}</div>', unsafe_allow_html=True)

        # AI suggestions
        tips = generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, location)
        for tip in tips:
            st.markdown(f'<div class="suggestion-box">{tip}</div>', unsafe_allow_html=True)

    # FEATURE IMPORTANCE
    st.subheader("📊 How each input affects profit")
    fig2 = create_feature_plot(model_data)
    st.pyplot(fig2)

    # DOWNLOAD CSV
    st.subheader("📥 Download your report")
    df_report = pd.DataFrame({
        'R&D Spend':[rd_spend],
        'Admin Spend':[admin_spend],
        'Marketing Spend':[marketing_spend],
        'Province':[location],
        'Predicted Profit':[prediction if 'prediction' in locals() else 0]
    })
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="startup_report.csv", mime='text/csv')

    # FOOTER
    st.markdown("---")
st.markdown('<p style="text-align:center;">🚀 Built with Machine Learning & Streamlit | GitHub Repo</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">👨‍💻 Made by Husnain</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
