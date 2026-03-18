import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="Manipur Conflict & Humanitarian Impact Analyzer",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Unique UI (Dark Glassmorphism) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f, #15151e);
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff4b4b;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Metrics Box - Glassmorphism */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #ff4b4b;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 47, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 75, 75, 0.1);
        border-radius: 5px;
        color: #ff4b4b;
    }
    
    /* DataFrame */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Models ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/manipur_processed.csv')
        df['event_date'] = pd.to_datetime(df['event_date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        with open('models/random_forest_conflict_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        return None, None

df = load_data()
model, encoders = load_model()

# --- App Header ---
st.title("🛡️ Manipur Conflict & Humanitarian Impact Analyzer")
st.markdown("*An interactive geospatial and temporal analysis of conflict events in Manipur.*")

if df.empty:
    st.error("Processed data not found! Please run the data pipelines first.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Controls")
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("Navigation", ["Overview Dashboard", "Predictive Risk Model"])
districts = ["All"] + list(df['district'].dropna().unique())
selected_district = st.sidebar.selectbox("Filter District (Overview Only)", districts)

year_range = st.sidebar.slider(
    "Select Year Range", 
    min_value=int(df['year'].min()), 
    max_value=int(df['year'].max()), 
    value=(int(df['year'].min()), int(df['year'].max()))
)

# Filter dataframe
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df['district'] == selected_district]

# --- View 1: Overview Dashboard ---
if view_mode == "Overview Dashboard":
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_events = len(filtered_df)
    total_fatalities = filtered_df['fatalities'].sum()
    most_affected = filtered_df['district'].value_counts().index[0] if not filtered_df.empty else "N/A"
    peak_year = filtered_df['year'].value_counts().index[0] if not filtered_df.empty else "N/A"
    
    col1.metric("Total Events", f"{total_events:,}")
    col2.metric("Total Fatalities", f"{int(total_fatalities):,}")
    col3.metric("Most Affected District", most_affected)
    col4.metric("Peak Conflict Year", peak_year)
    
    st.markdown("---")
    
    # Visuals Row 1
    v_col1, v_col2 = st.columns([1, 1])
    
    with v_col1:
        st.subheader("🗺️ Conflict Heatmap")
        st.markdown("Geospatial distribution of events.")
        
        # Center map on Manipur
        m = folium.Map(location=[24.8170, 93.9368], zoom_start=7, tiles='CartoDB dark_matter')
        
        # Add points (sample to avoid lag if too large)
        map_data = filtered_df.dropna(subset=['latitude', 'longitude']).tail(1000)
        for idx, row in map_data.iterrows():
            color = 'red' if row['fatalities'] > 0 else 'orange'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{row['event_date'].strftime('%Y-%m-%d')} - {row['event_type']} (Fatalities: {row['fatalities']})"
            ).add_to(m)
            
        st_folium(m, width=500, height=400)
        
    with v_col2:
        st.subheader("📈 Monthly Event Trend")
        # Ensure we group by year_month properly for plotting
        trend_df = filtered_df.groupby('year_month').size().reset_index(name='count')
        trend_df['date'] = pd.to_datetime(trend_df['year_month'])
        trend_df = trend_df.sort_values('date')
        
        fig = px.line(trend_df, x='date', y='count', 
                      line_shape='spline', render_mode='svg')
        fig.update_traces(line=dict(color='#ff4b4b', width=3))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis_title="Timeline",
            yaxis_title="Event Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    # Visuals Row 2
    st.subheader("📊 Event Types Breakdown")
    event_pie = filtered_df['event_type'].value_counts().reset_index()
    event_pie.columns = ['Event Type', 'Count']
    fig_pie = px.pie(event_pie, names='Event Type', values='Count', hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Reds_r)
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    st.plotly_chart(fig_pie, use_container_width=True)

# --- View 2: Predictive Risk Model ---
elif view_mode == "Predictive Risk Model":
    st.header("🔮 Conflict Risk Prediction")
    st.markdown("Use the trained **Random Forest** model to predict conflict intensity for a given district and month.")
    
    if model is None or encoders is None:
        st.warning("Model not found! Please run the training script.")
    else:
        with st.container():
            st.markdown("<div style='background-color: rgba(255, 75, 75, 0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(255, 75, 75, 0.2);'>", unsafe_allow_html=True)
            p_col1, p_col2, p_col3 = st.columns(3)
            
            with p_col1:
                input_district = st.selectbox("Select District", encoders['district'].classes_)
                input_year = st.number_input("Year", min_value=2010, max_value=2030, value=2024)
                
            with p_col2:
                input_season = st.selectbox("Season", encoders['season'].classes_)
                lag_events = st.number_input("Last Month's Event Count", min_value=0, max_value=200, value=5)
                
            with p_col3:
                lag_fatals = st.number_input("Last Month's Fatalities", min_value=0, max_value=200, value=0)
                st.markdown("<br>", unsafe_allow_html=True)
                predict_btn = st.button("Predict Risk Level", use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        if predict_btn:
            # Transform inputs
            d_enc = encoders['district'].transform([input_district])[0]
            s_enc = encoders['season'].transform([input_season])[0]
            
            # Predict
            X_pred = pd.DataFrame({
                'district_encoded': [d_enc],
                'season_encoded': [s_enc],
                'lag_event_count': [lag_events],
                'lag_fatalities': [lag_fatals],
                'year': [input_year]
            })
            
            # Ensure order matches features
            X_pred = X_pred[encoders['feature_names']]
            
            pred_class = model.predict(X_pred)[0]
            pred_label = encoders['target'].inverse_transform([pred_class])[0]
            
            st.markdown("---")
            if pred_label == 'High':
                st.error(f"### 🚨 Predicted Risk Level: {pred_label} Intensity")
                st.markdown(f"**Warning**: High probability of conflict escalation in **{input_district}** during **{input_season} {input_year}**.")
            else:
                st.success(f"### ✅ Predicted Risk Level: {pred_label} Intensity")
                st.markdown(f"**Stable**: Low probability of major conflict escalation in **{input_district}** during **{input_season} {input_year}**.")
            
        # Feature Importance
        st.markdown("---")
        st.subheader("Model Feature Importance")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': encoders['feature_names'], 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=True)
        
        fig_bar = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance',
                         color_continuous_scale='Reds')
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
        st.plotly_chart(fig_bar, use_container_width=True)
