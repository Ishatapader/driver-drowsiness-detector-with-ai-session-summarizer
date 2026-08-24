import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

# Load configuration dynamically to match main.py filename
try:
    with open("config.json", 'r') as file:
        config = json.load(file)
    CSV_FILE = config['system']['csv_log_file']
except Exception:
    CSV_FILE = "driver_telemetry.csv"

EAR_THRESH = 0.20
MAR_THRESH = 0.65
PITCH_THRESH = -15.0

# Securely load the API key from environment variables 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Driver Monitoring Dashboard", layout="wide")

st.title("Driver Monitoring System Dashboard")
st.write("Session logs tracking eye closure episodes, yawning durations, and head posture drops.")

@st.cache_data
def load_data():
    if not os.path.exists(CSV_FILE):
        return None
    try:
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except Exception as e:
        print(f"[ERROR loading CSV]: {e}")
        return None

df = load_data()

if df is None or df.empty:
    st.warning("No telemetry data found. Run python main.py first to generate logs.")
    st.stop()

df_starts = df[df['Event'] == 'START'] if 'Event' in df.columns else df

# BASIC METRICS
st.subheader("Session Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Alert Episodes", len(df_starts))
col2.metric("Micro-sleep Episodes", len(df_starts[df_starts['Warning_Type'] == 'eyes']))
col3.metric("Yawn Episodes", len(df_starts[df_starts['Warning_Type'] == 'yawn']))
col4.metric("Head Nod Episodes", len(df_starts[df_starts['Warning_Type'] == 'nod']))

st.markdown("---")

# RAW CSV DATA
st.subheader("Logged Telemetry Data")
if st.checkbox("Do you want to view the raw CSV episode log table?"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")

# EVENT-BASED TIMELINE CHARTS
st.subheader("Telemetry Episode Timeline")
st.write("Detailed breakdown of when safety thresholds were crossed during the drive.")

def plot_event_metric(df_start_subset, warning_type, title, threshold, y_label, description):
    st.markdown(f"**{title}**")
    st.caption(description)
    
    fig = go.Figure()
    subset = df_start_subset[df_start_subset['Warning_Type'] == warning_type].copy()
    
    if not subset.empty:
        y_column = 'EAR' if warning_type == 'eyes' else ('MAR' if warning_type == 'yawn' else 'Pitch_Delta')

        fig.add_trace(go.Scatter(
            x=subset['Timestamp'], 
            y=subset[y_column], 
            mode='markers+lines', 
            name='Episode Start',
            marker=dict(color='#E74C3C', size=10, symbol='x'),
            line=dict(color='#E74C3C', width=2, dash='dot')
        ))
    
    fig.add_hline(
        y=threshold, 
        line_dash="dash", 
        line_color="#F1C40F", 
        annotation_text=f"Safety Limit ({threshold})",
        annotation_position="bottom right"
    )    
    
    fig.update_layout(
        xaxis_title="Timestamp", 
        yaxis_title=y_label, 
        height=300, 
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

plot_event_metric(df_starts, 'eyes', '1. Eye Closure Episodes (EAR)', EAR_THRESH, 'Eye Ratio', 'Tracks micro-sleeps over time.')
plot_event_metric(df_starts, 'yawn', '2. Yawning Episodes (MAR)', MAR_THRESH, 'Mouth Ratio', 'Tracks yawn events over time.')
plot_event_metric(df_starts, 'nod', '3. Head Posture Episodes (Pitch Delta)', PITCH_THRESH, 'Pitch Delta (deg)', 'Tracks head drops and nods.')

# AI TRIP REPORT 
st.markdown("---")
st.subheader("AI Trip Summary")

if st.button("Generate Summary"):
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please set it in your .env file or Streamlit Secrets.")
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-3.6-flash')

            total_warnings = len(df_starts)
            event_counts = df_starts['Warning_Type'].value_counts().to_dict()
            
            prompt = f"""
            Analyze this driver alert summary casually:
            - Total warning episodes: {total_warnings}
            - Breakdown: {event_counts}
            
            Provide the summary in 3 concise bullet points:
            * Overall alertness level during the trip
            * Most frequent type of warning observed
            * A quick safety tip for the driver
            """

            with st.spinner("Analyzing trip logs..."):
                response = model.generate_content(prompt)
                st.write(response.text)

        except Exception as e:
            st.error(f"Could not generate summary: {e}")