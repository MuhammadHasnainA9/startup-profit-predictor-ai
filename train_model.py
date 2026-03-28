"""
AI-Powered Startup Profit Prediction - Model Training Script
This script loads the 50_Startups dataset, preprocesses it, trains multiple models,
and saves the best performing model for deployment.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the 50_Startups dataset"""
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.read_csv('50_Startups.csv')
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Clean column names (remove spaces and lowercase)
    df.columns = df.columns.str.replace(' ', '_').str.replace('&', 'and').str.lower()
    
    # Handle missing values if any
    df = df.dropna()
    
    # Encode categorical variable 'State'
    le = LabelEncoder()
    df['state_encoded'] = le.fit_transform(df['state'])
    
    # Select features and target
    features = ['randd_spend', 'administration', 'marketing_spend', 'state_encoded']
    target = 'profit'
    
    X = df[features]
    y = df[target]
    
    print(f"Features selected: {features}")
    print(f"Target variable: {target}")
    
    return X, y, le, df

def train_and_evaluate_models(X, y):
    """Train multiple models and select the best one"""
    print("\nTraining and evaluating models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {
            'model': model,
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    
    # Select best model based on R² score
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
    best_model = results[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best R² Score: {best_model['r2_score']:.4f}")
    
    return results, best_model_name, scaler, X_test, y_test

def save_model(best_model, scaler, label_encoder, best_model_name):
    """Save the best model and preprocessing objects"""
    print("\nSaving model and preprocessing objects...")
    
    model_data = {
        'model': best_model['model'],
        'scaler': scaler,
        'label_encoder': label_encoder,
        'model_name': best_model_name,
        'r2_score': best_model['r2_score'],
        'mae': best_model['mae']
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'model.pkl'")

def create_visualizations(df, results, X_test, y_test):
    """Create and save visualization plots"""
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Feature correlations heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['randd_spend', 'administration', 'marketing_spend', 'profit']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
    y_pred = results[best_model_name]['predictions']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.title(f'Actual vs Predicted Profit - {best_model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature importance for Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_names = X_test.columns
        importances = rf_model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved as PNG files")

def generate_ai_suggestions(rd_spend, admin_spend, marketing_spend, state):
    """Generate AI-powered business suggestions based on input parameters"""
    suggestions = []
    
    total_spend = rd_spend + admin_spend + marketing_spend
    
    # R&D Analysis
    if rd_spend < total_spend * 0.3:
        suggestions.append("💡 Consider increasing R&D investment - innovation drives long-term growth")
    elif rd_spend > total_spend * 0.6:
        suggestions.append("⚖️ R&D spending is very high - ensure balanced investment across departments")
    
    # Marketing Analysis
    if marketing_spend < total_spend * 0.2:
        suggestions.append("📢 Marketing budget seems low - consider increasing to reach more customers")
    elif marketing_spend > total_spend * 0.5:
        suggestions.append("🎯 High marketing spend - ensure ROI tracking and optimize campaigns")
    
    # Administration Analysis
    if admin_spend > total_spend * 0.4:
        suggestions.append("🏢 Administration costs are high - look for operational efficiencies")
    
    # Balance Analysis
    rd_ratio = rd_spend / total_spend
    marketing_ratio = marketing_spend / total_spend
    
    if abs(rd_ratio - marketing_ratio) > 0.3:
        suggestions.append("⚖️ Consider balancing R&D and marketing spend for optimal growth")
    
    # State-specific suggestions
    state_suggestions = {
        'California': '🌴 California market is competitive - focus on innovation and differentiation',
        'New York': '🏙️ New York offers diverse opportunities - consider market segmentation',
        'Florida': '🌊 Florida market growing - leverage tourism and service sectors'
    }
    
    if state in state_suggestions:
        suggestions.append(state_suggestions[state])
    
    # Total investment analysis
    if total_spend < 100000:
        suggestions.append("💰 Total investment is modest - consider strategic scaling plan")
    elif total_spend > 500000:
        suggestions.append("🚀 High investment level - focus on efficient execution and ROI")
    
    return suggestions

def main():
    """Main function to run the complete training pipeline"""
    print("Starting AI-Powered Startup Profit Prediction Model Training")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        X, y, label_encoder, df = load_and_preprocess_data()
        
        # Train and evaluate models
        results, best_model_name, scaler, X_test, y_test = train_and_evaluate_models(X, y)
        
        # Save the best model
        best_model = results[best_model_name]
        save_model(best_model, scaler, label_encoder, best_model_name)
        
        # Create visualizations
        create_visualizations(df, results, X_test, y_test)
        
        # Test AI suggestions function
        print("\nTesting AI suggestions function...")
        test_suggestions = generate_ai_suggestions(150000, 120000, 80000, 'California')
        print("Sample AI suggestions:")
        for suggestion in test_suggestions:
            print(f"  {suggestion}")
        
        print("\nTraining pipeline completed successfully!")
        print("=" * 60)
        print("Files created:")
        print("  - model.pkl (trained model)")
        print("  - correlation_heatmap.png")
        print("  - actual_vs_predicted.png")
        print("  - feature_importance.png")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
