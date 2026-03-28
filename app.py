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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Startup Profit Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
def set_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .suggestion-box {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .profit-display {
            font-size: 2rem;
            font-weight: bold;
            color: #2ecc71;
            text-align: center;
            padding: 1rem;
            background: #e8f5e8;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            padding: 0.5rem 2rem;
            border-radius: 25px;
            border: none;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'train_model.py' first.")
        return None

def generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, state):
    """Generate AI-powered business suggestions based on input parameters"""
    suggestions = []
    
    total_spend = rd_spend + admin_spend + marketing_spend
    
    # R&D Analysis
    if rd_spend < total_spend * 0.3:
        suggestions.append("💡 **Increase R&D Investment**: Innovation drives long-term growth and competitive advantage")
    elif rd_spend > total_spend * 0.6:
        suggestions.append("⚖️ **Balance R&D Spending**: Very high R&D investment - ensure balanced allocation across departments")
    
    # Marketing Analysis
    if marketing_spend < total_spend * 0.2:
        suggestions.append("📢 **Boost Marketing Budget**: Low marketing spend may limit customer acquisition and brand awareness")
    elif marketing_spend > total_spend * 0.5:
        suggestions.append("🎯 **Optimize Marketing ROI**: High marketing spend - ensure campaign tracking and conversion optimization")
    
    # Administration Analysis
    if admin_spend > total_spend * 0.4:
        suggestions.append("🏢 **Reduce Administrative Costs**: High overhead - explore automation and operational efficiencies")
    
    # Balance Analysis
    rd_ratio = rd_spend / total_spend
    marketing_ratio = marketing_spend / total_spend
    
    if abs(rd_ratio - marketing_ratio) > 0.3:
        suggestions.append("⚖️ **Balance Growth Investments**: Consider aligning R&D and marketing spend for optimal growth trajectory")
    
    # State-specific suggestions
    state_suggestions = {
        'California': '🌴 **California Strategy**: Focus on tech innovation and differentiation in this competitive market',
        'New York': '🏙️ **New York Approach**: Leverage diverse opportunities through market segmentation and networking',
        'Florida': '🌊 **Florida Opportunity**: Capitalize on growing market, tourism, and service sector potential'
    }
    
    if state in state_suggestions:
        suggestions.append(state_suggestions[state])
    
    # Total investment analysis
    if total_spend < 100000:
        suggestions.append("💰 **Scaling Strategy**: Modest investment - develop strategic scaling plan with phased growth")
    elif total_spend > 500000:
        suggestions.append("🚀 **High-Stakes Execution**: Significant investment - focus on efficient execution and measurable ROI")
    
    # Profit optimization tip
    if rd_spend > 100000 and marketing_spend > 100000:
        suggestions.append("🔄 **Synergy Opportunity**: Strong R&D and marketing - create integrated product-market strategy")
    
    return suggestions

def create_feature_importance_plot(model_data):
    """Create feature importance visualization"""
    if model_data['model_name'] == 'Random Forest':
        model = model_data['model']
        feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
        importances = model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importances)
        bars = ax.barh(range(len(importances)), importances[indices], color='#1f77b4')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    else:
        # For Linear Regression, show coefficients
        model = model_data['model']
        feature_names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        coefficients = model.coef_
        indices = np.argsort(np.abs(coefficients))
        colors = ['#1f77b4' if coef > 0 else '#e74c3c' for coef in coefficients[indices]]
        bars = ax.barh(range(len(coefficients)), np.abs(coefficients[indices]), color=colors)
        ax.set_yticks(range(len(coefficients)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_title('Linear Regression Feature Impact', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{coefficients[indices][i]:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig

def create_spending_analysis_chart(rd_spend, admin_spend, marketing_spend):
    """Create a spending breakdown chart"""
    categories = ['R&D', 'Administration', 'Marketing']
    values = [rd_spend, admin_spend, marketing_spend]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Spending Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(categories, values, color=colors)
    ax2.set_title('Spending by Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Amount ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main application function"""
    # Apply custom styling
    set_custom_css()
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Startup Profit Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict your startup profit with AI-powered insights and recommendations</p>', unsafe_allow_html=True)
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Used</h3>
            <p style="font-size: 1.5rem; font-weight: bold;">{model_data['model_name']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy (R²)</h3>
            <p style="font-size: 1.5rem; font-weight: bold;">{model_data['r2_score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Mean Absolute Error</h3>
            <p style="font-size: 1.5rem; font-weight: bold;">${model_data['mae']:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Enter Your Startup Parameters")
        
        # Create two columns for inputs
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            rd_spend = st.number_input(
                "💡 R&D Spend ($)",
                min_value=0,
                max_value=1000000,
                value=150000,
                step=5000,
                help="Investment in research and development"
            )
            
            admin_spend = st.number_input(
                "🏢 Administration Spend ($)",
                min_value=0,
                max_value=1000000,
                value=120000,
                step=5000,
                help="Administrative and operational costs"
            )
        
        with input_col2:
            marketing_spend = st.number_input(
                "📢 Marketing Spend ($)",
                min_value=0,
                max_value=1000000,
                value=80000,
                step=5000,
                help="Marketing and advertising expenses"
            )
            
            state = st.selectbox(
                "🌍 State",
                options=['California', 'New York', 'Florida'],
                help="Select your business location"
            )
    
    with col2:
        st.subheader("📈 Spending Overview")
        total_spend = rd_spend + admin_spend + marketing_spend
        
        st.metric("Total Investment", f"${total_spend:,.0f}")
        st.metric("R&D Ratio", f"{(rd_spend/total_spend)*100:.1f}%")
        st.metric("Marketing Ratio", f"{(marketing_spend/total_spend)*100:.1f}%")
        
        # Create spending chart
        if total_spend > 0:
            fig = create_spending_analysis_chart(rd_spend, admin_spend, marketing_spend)
            st.pyplot(fig)
    
    # Prediction button
    predict_button = st.button("🔮 Predict Profit", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'rd_spend': [rd_spend],
            'administration': [admin_spend],
            'marketing_spend': [marketing_spend],
            'state': [state]
        })
        
        # Encode state
        state_encoded = model_data['label_encoder'].transform([state])[0]
        
        # Create feature array
        features = np.array([[rd_spend, admin_spend, marketing_spend, state_encoded]])
        
        # Scale features if using Linear Regression
        if model_data['model_name'] == 'Linear Regression':
            features_scaled = model_data['scaler'].transform(features)
            prediction = model_data['model'].predict(features_scaled)[0]
        else:
            prediction = model_data['model'].predict(features)[0]
        
        # Display prediction
        st.divider()
        st.subheader("💰 Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="profit-display">
                ${prediction:,.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            confidence = model_data['r2_score'] * 100
            if confidence > 90:
                confidence_color = "🟢"
                confidence_text = "Very High"
            elif confidence > 80:
                confidence_color = "🟡"
                confidence_text = "High"
            else:
                confidence_color = "🟠"
                confidence_text = "Moderate"
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <p>{confidence_color} Confidence: {confidence_text} ({confidence:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🤖 AI-Powered Business Suggestions")
            suggestions = generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, state)
            
            for suggestion in suggestions:
                st.markdown(f"""
                <div class="suggestion-box">
                    {suggestion}
                </div>
                """, unsafe_allow_html=True)
    
    # Feature importance section
    st.divider()
    st.subheader("📊 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Importance/Impact**")
        fig = create_feature_importance_plot(model_data)
        st.pyplot(fig)
    
    with col2:
        st.write("**How to Use This Tool**")
        st.markdown("""
        1. **Enter your spending data** in the input fields
        2. **Select your state** for location-based insights
        3. **Click "Predict Profit"** to get your forecast
        4. **Review AI suggestions** for business optimization
        5. **Analyze feature importance** to understand key drivers
        
        **Tips for Better Predictions:**
        - Ensure balanced spending across departments
        - Higher R&D investment generally correlates with better profits
        - Marketing spend should align with your growth stage
        - Consider state-specific market conditions
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚀 AI Startup Profit Predictor | Built with Machine Learning & Streamlit</p>
        <p>Model trained on 50 startups dataset with {model_data['model_name']} algorithm</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
