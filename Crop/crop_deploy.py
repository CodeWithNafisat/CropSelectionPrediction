import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Styles & Custom CSS ---
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {width: 100%; background-color: #4CAF50; color: white; font-weight: bold;}
    .stButton>button:hover {background-color: #45a049;}
    .metric-card {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center;}
    h1, h2, h3 {color: #2E7D32;}
    </style>
""", unsafe_allow_html=True)

# --- 3. Model Architecture ---
class ClassificationModel(nn.Module):
    def __init__(self, in_features=7, h1=64, h2=32, output_target=22):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.output = nn.Linear(h2, output_target)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.output(x)
        return x

# --- 4. Loading Resources ---
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load('Crop/scaler.pkl')
        le = joblib.load('Crop/le.pkl')
        model = ClassificationModel(output_target=len(le.classes_))
        model.load_state_dict(torch.load('Crop/best_model.pth'))
        model.eval()
        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Ensure 'Crop/best_model.pth', 'Crop/scaler.pkl', and 'Crop/le.pkl' exist.")
        return None, None, None


@st.cache_data
def get_shap_background(csv_path='Crop/bg.csv'):
    """Load background data for SHAP"""
    try:
        df = pd.read_csv(csv_path)
        num_cols = df.select_dtypes(include=['int', 'float']).columns
        for col in num_cols:
            df[col] = winsorize(df[col], limits=[0.075, 0.075])
        
        if 'label' in df.columns:
            X = df.drop('label', axis=1).values
        else:
            X = df.values
            
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)
        background = X_scaled[:100].astype(np.float32)
        return background, df.columns.drop('label') if 'label' in df.columns else df.columns
    except Exception as e:
        st.warning(f"Could not load background data for SHAP: {e}")
        return None, None

# --- 5. Feature Data & Sidebar Logic ---

def get_feature_descriptions():
    """Return descriptions for each feature"""
    return {
        'N': {
            'description': 'Nitrogen content in soil',
            'importance': 'Essential for leaf growth and chlorophyll production',
            'ideal_range': '50-120 mg/kg',
            'unit': 'mg/kg'
        },
        'P': {
            'description': 'Phosphorus content in soil',
            'importance': 'Crucial for root development and energy transfer',
            'ideal_range': '30-70 mg/kg',
            'unit': 'mg/kg'
        },
        'K': {
            'description': 'Potassium content in soil',
            'importance': 'Helps with flower/fruit formation and disease resistance',
            'ideal_range': '100-250 mg/kg',
            'unit': 'mg/kg'
        },
        'temperature': {
            'description': 'Average temperature',
            'importance': 'Affects germination, growth rate, and crop maturity',
            'ideal_range': '15-30°C (varies by crop)',
            'unit': '°C'
        },
        'humidity': {
            'description': 'Relative humidity',
            'importance': 'Influences transpiration and disease occurrence',
            'ideal_range': '40-80% (varies by crop)',
            'unit': '%'
        },
        'ph': {
            'description': 'Soil pH level',
            'importance': 'Affects nutrient availability and microbial activity',
            'ideal_range': '5.5-7.0 (most crops)',
            'unit': 'pH units'
        },
        'rainfall': {
            'description': 'Annual rainfall',
            'importance': 'Determines water availability for crop growth',
            'ideal_range': '500-1500 mm (varies by crop)',
            'unit': 'mm'
        }
    }

def sidebar_input_features():
    st.sidebar.header("🌱 Soil & Climate Inputs")
    
    feature_desc = get_feature_descriptions()

    # --- A. Feature Reference Guide (Expander) ---
    with st.sidebar.expander("📖 Feature Reference Guide"):
        st.markdown("### Feature Details & Ideal Ranges")
        for feature, details in feature_desc.items():
            st.markdown(f"""
            **{feature} ({details['unit']})**
            - *{details['description']}*
            - Range: `{details['ideal_range']}`
            ---
            """)

    st.sidebar.write("Adjust the sliders below to match your soil data:")

    # --- B. Interactive Sliders with Tooltips ---
    def get_tooltip(key):
        data = feature_desc[key]
        return f"{data['description']}\n\n⚠️ Importance: {data['importance']}"

    N = st.sidebar.slider('Nitrogen (N)', 0, 150, 50, help=get_tooltip('N'))
    P = st.sidebar.slider('Phosphorus (P)', 0, 150, 50, help=get_tooltip('P'))
    K = st.sidebar.slider('Potassium (K)', 0, 210, 50, help=get_tooltip('K'))
    temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0, help=get_tooltip('temperature'))
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 70.0, help=get_tooltip('humidity'))
    ph = st.sidebar.slider('pH Level', 0.0, 14.0, 6.5, help=get_tooltip('ph'))
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 300.0, 100.0, help=get_tooltip('rainfall'))

    # --- C. Return Single-Row DataFrame ---
    data = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    return pd.DataFrame(data, index=[0])

# Load resources
model, scaler, le = load_artifacts()
background_data, feature_names = get_shap_background('bg.csv')
feature_desc = get_feature_descriptions()

# --- 6. Main Interface ---
input_df = sidebar_input_features()

st.title("🌾 Intelligent Crop Recommendation System")
st.markdown("### Utilize Deep Learning to find the best crop for your soil.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Inputs")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    if st.button("Predict Optimal Crop"):
        if model is not None and scaler is not None:
            # Preprocess input for model (scaled)
            scaled_input = scaler.transform(input_df.values).astype(np.float32)
            input_tensor = torch.FloatTensor(scaled_input)
            
            # Prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            prediction = le.inverse_transform([pred_idx.item()])[0]
            confidence_score = conf.item() * 100
            
            # Display result
            st.success("Analysis Complete!")
            st.markdown(f"""
            <div class='metric-card'>
                <h2>Recommended Crop</h2>
                <h1 style='color: #4CAF50; font-size: 3em;'>{prediction.upper()}</h1>
                <p>Confidence: <b>{confidence_score:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save to session state
            st.session_state.update({
                'probs': probs.numpy()[0],
                'scaled_input': scaled_input,
                'prediction_idx': pred_idx.item(),
                'made_prediction': True
            })
        else:
            st.error("Model artifacts not found. Please check your files.")

with col2:
    if st.session_state.get('made_prediction', False):
        
        # --- SHAP Explanation Section ---
        if background_data is not None:
            st.subheader("💡 Explanation: Why this prediction?")
            
            def predict_wrapper(x):
                with torch.no_grad():
                    tensor_x = torch.from_numpy(x.astype(np.float32))
                    logits = model(tensor_x)
                    return F.softmax(logits, dim=1).numpy()
            
            explainer = shap.Explainer(predict_wrapper, shap.maskers.Independent(background_data))
            shap_values = explainer(st.session_state['scaled_input'])
            pred_idx = st.session_state['prediction_idx']
            instance_shap = shap_values[0, :, pred_idx].values
            
            # Horizontal bar chart (contribution %)
            abs_vals = np.abs(instance_shap)
            percent = (abs_vals / abs_vals.sum()) * 100
            sorted_idx = np.argsort(percent)[::-1]
            
            # Match feature names safely
            features_sorted_keys = [feature_names[i] for i in sorted_idx]
            percent_sorted = percent[sorted_idx]
            shap_sorted = instance_shap[sorted_idx]
            colors = ['#FF6B6B' if v > 0 else '#4D96FF' for v in shap_sorted]
            
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            ax.barh(features_sorted_keys, percent_sorted, color=colors, alpha=0.8)
            ax.set_xlabel("Contribution (%)")
            ax.set_title(f"Impact on prediction: {le.classes_[pred_idx]}")
            ax.invert_yaxis()
            st.pyplot(fig_shap)
            
            # --- UPDATED TEXTUAL INTERPRETATION (All Features) ---
            st.markdown("### 📝 Detailed Feature Impact")
            st.markdown("Below is the breakdown of how every feature influenced this specific prediction:")
            
            for feature, p, sval in zip(features_sorted_keys, percent_sorted, shap_sorted):
                # Determine direction
                dir_icon = "🟢 Positive" if sval > 0 else "🔴 Negative"
                dir_text = "increased likelihood" if sval > 0 else "decreased likelihood"
                
                # Retrieve importance from dictionary
                feat_info = feature_desc.get(feature, {})
                importance_note = feat_info.get('importance', 'plays a role in growth')
                
                # Render as a styled list item
                st.markdown(f"""
                **{feature}** — *{dir_icon} Influence ({p:.1f}%)*
                - **Biological Role:** {importance_note}
                - **Effect:** The current value {dir_text} of predicting **{le.classes_[pred_idx]}**.
                """)
                st.markdown("---")

    else:
        st.info("👈 Enter soil parameters in the sidebar and click 'Predict' to see results.")
