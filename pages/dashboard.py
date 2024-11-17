import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"dashboard_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_diagnostic(message, data=None):
    """Log diagnostic information to both file and console"""
    logging.info(message)
    if data is not None:
        logging.info(f"Data: {data}")

# Database path
DB_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\experiment_data_20241116.sqlite"

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at {DB_PATH}.")
    st.stop()

# Initialize SQLite connection
conn = sqlite3.connect(DB_PATH)

def get_plot_metadata():
    """Get plot metadata including treatment and crop info"""
    query = """
    SELECT p.plot_id, p.treatment, y.trt_name, y.crop_type
    FROM plots p
    LEFT JOIN yields y ON p.plot_id = y.plot_id
    """
    return pd.read_sql_query(query, conn)

def parse_sensor_name(sensor_name):
    pattern = r'^(?P<SensorType>[A-Z]{3})(?P<FieldNumber>\d{4})(?P<Node>[A-E])(?P<Treatment>[1-4])(?P<Depth>\d{2}|xx)24$'
    match = re.match(pattern, sensor_name)
    return match.groupdict() if match else {}

@st.cache_data(ttl=600)
def fetch_data(plot_id, start_date, end_date):
    """Fetch data from database"""
    query = f"""
        SELECT timestamp, variable_name, value
        FROM data
        WHERE plot_id = '{plot_id}'
        AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert values to numeric, coercing errors to NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Drop rows where value is NaN
    df = df.dropna(subset=['value'])
    
    # Keep only the first occurrence for each timestamp-variable combination
    df = df.groupby(['timestamp', 'variable_name']).first().reset_index()
    
    return df

@st.cache_data(ttl=600)
def fetch_irrigation_events(plot_id, start_date, end_date):
    query = f"""
        SELECT date, amount_inches, amount_mm
        FROM irrigation_events
        WHERE plot_id = '{plot_id}'
        AND DATE(date) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    return df

def normalize_cwsi(value):
    """Normalize CWSI from 0-1.75 to 0-1 scale"""
    try:
        if pd.isna(value) or value is None:
            return None
        return float(value) / 1.75
    except (TypeError, ValueError):
        return None

def diagnose_data(df, df_pivot):
    """Diagnose data availability and quality"""
    required_vars = {
        'Weather': ['Solar_2m_Avg', 'WndAveSpd_3m', 'RH_2m_Avg', 'etc'],
        'Temperature': ['Ta_2m_Avg', 'cwsi'],
        'Water': ['swsi', 'Rain_1m_Tot']
    }
    
    # Get available variables
    available_vars = df['variable_name'].unique()
    log_diagnostic(f"\nAvailable variables: {sorted(available_vars)}")
    
    # Check TDR sensors
    tdr_sensors = [v for v in available_vars if v.startswith('TDR')]
    log_diagnostic(f"\nFound TDR sensors: {tdr_sensors}")
    
    # Check for missing required variables
    for category, vars in required_vars.items():
        missing = [v for v in vars if v not in available_vars]
        if missing:
            log_diagnostic(f"\nMissing {category} variables: {missing}")
        
        # Log data ranges for available variables
        for var in vars:
            if var in available_vars:
                var_data = df[df['variable_name'] == var]
                log_diagnostic(
                    f"\n{var} data:",
                    f"Count: {len(var_data)}, Range: {var_data['value'].min():.2f} to {var_data['value'].max():.2f}"
                )

    return {
        'available_vars': available_vars,
        'tdr_sensors': tdr_sensors,
        'missing_vars': {cat: [v for v in vars if v not in available_vars] 
                        for cat, vars in required_vars.items()}
    }

def main():
    st.set_page_config(
        page_title="Crop2Cloud Platform",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🌾 **CROP2CLOUD Platform** 🌾")

    # Get plot metadata
    plot_metadata = get_plot_metadata()
    
    # Create plot selection options with metadata
    plot_options = {
        row['plot_id']: f"Plot {row['plot_id']} - {row['crop_type'].title()} ({row['trt_name']})"
        for _, row in plot_metadata.iterrows()
    }

    # Left column (1/3 width) for controls and summary
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.header("Plot Selection & Summary")
        plot_selection = st.selectbox(
            "Select Plot",
            options=list(plot_options.keys()),
            format_func=lambda x: plot_options[x]
        )
        
        date_range = st.date_input(
            "Select Date Range",
            [datetime.today() - timedelta(days=7), datetime.today()]
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            st.error("Please select a valid date range.")
            st.stop()

        current_plot = plot_metadata[plot_metadata['plot_id'] == plot_selection].iloc[0]

        # Display plot summary
        st.subheader("Plot Information")
        st.write(f"**Crop Type:** {current_plot['crop_type'].title()}")
        st.write(f"**Treatment:** {current_plot['trt_name']}")
        st.write(f"**Treatment Number:** {current_plot['treatment']}")

    # Fetch data
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    with st.spinner("Loading data..."):
        df = fetch_data(plot_selection, start_date_str, end_date_str)
        irrigation_df = fetch_irrigation_events(plot_selection, start_date_str, end_date_str)
        
        # Run diagnostics
        diagnostic_info = diagnose_data(df, None)  # We'll create df_pivot after diagnostics
        
        # Create pivot table for easier plotting
        df_pivot = df.pivot(index='timestamp', columns='variable_name', values='value').reset_index()

    if df_pivot.empty:
        st.warning("No data available for the selected period.")
        st.stop()

    with right_col:
        # 1. Weather Graph
        st.subheader("Weather Conditions")
        fig1 = go.Figure()

        # Solar radiation
        if 'Solar_2m_Avg' in df_pivot.columns:
            fig1.add_trace(go.Scatter(
                x=df_pivot['timestamp'],
                y=df_pivot['Solar_2m_Avg'],
                name='Solar Radiation',
                line=dict(color='gold', width=2),
                yaxis='y'
            ))

        # Wind speed
        if 'WndAveSpd_3m' in df_pivot.columns:
            fig1.add_trace(go.Scatter(
                x=df_pivot['timestamp'],
                y=df_pivot['WndAveSpd_3m'],
                name='Wind Speed',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))

        # Relative humidity
        if 'RH_2m_Avg' in df_pivot.columns:
            fig1.add_trace(go.Scatter(
                x=df_pivot['timestamp'],
                y=df_pivot['RH_2m_Avg'],
                name='Relative Humidity',
                line=dict(color='green', width=2),
                yaxis='y3'
            ))

        # ETC
        if 'etc' in df_pivot.columns:
            fig1.add_trace(go.Scatter(
                x=df_pivot['timestamp'],
                y=df_pivot['etc'],
                name='ETC',
                line=dict(color='red', width=2),
                yaxis='y'
            ))

        fig1.update_layout(
            height=300,
            yaxis=dict(
                title='Solar Radiation (W/m²) / ETC (mm)',
                side='left'
            ),
            yaxis2=dict(
                title='Wind Speed (m/s)',
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title='RH (%)',
                overlaying='y',
                side='right',
                position=0.85,
                titlefont=dict(color='green'),
                tickfont=dict(color='green')
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=60, r=60, t=40, b=40)
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Temperature & CWSI Graph
        st.subheader("Temperature & Crop Water Stress")
        fig2 = go.Figure()

        # Canopy temperature (IRT sensors)
        irt_columns = [col for col in df['variable_name'].unique() if col.startswith('IRT')]
        if irt_columns:
            for irt in irt_columns:
                irt_data = df[df['variable_name'] == irt]
                fig2.add_trace(go.Scatter(
                    x=irt_data['timestamp'],
                    y=irt_data['value'],
                    name='Canopy Temperature',
                    line=dict(color='red', width=2)
                ))

        # Air temperature
        if 'Ta_2m_Avg' in df_pivot.columns:
            fig2.add_trace(go.Scatter(
                x=df_pivot['timestamp'],
                y=df_pivot['Ta_2m_Avg'],
                name='Air Temperature',
                line=dict(color='orange', width=2)
            ))

        # CWSI as bars
        cwsi_data = df[df['variable_name'] == 'cwsi'].copy()
        if not cwsi_data.empty:
            cwsi_data['value'] = cwsi_data['value'].apply(normalize_cwsi)
            cwsi_data = cwsi_data.dropna(subset=['value'])
            if not cwsi_data.empty:
                fig2.add_trace(go.Bar(
                    x=cwsi_data['timestamp'],
                    y=cwsi_data['value'],
                    name='CWSI',
                    marker_color='purple',
                    opacity=0.7,
                    yaxis='y2',
                    width=3600000  # 1 hour in milliseconds
                ))

        fig2.update_layout(
            height=300,
            yaxis=dict(title='Temperature (°C)', side='left'),
            yaxis2=dict(
                title='CWSI',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=60, r=60, t=40, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Soil Moisture, SWSI & Irrigation Graph
        st.subheader("Soil Moisture & Water Management")
        fig3 = go.Figure()

        # TDR sensors
        tdr_columns = diagnostic_info['tdr_sensors']
        if tdr_columns:
            for tdr in tdr_columns:
                parsed = parse_sensor_name(tdr)
                depth = parsed.get('Depth', 'xx')
                tdr_data = df[df['variable_name'] == tdr]
                fig3.add_trace(go.Scatter(
                    x=tdr_data['timestamp'],
                    y=tdr_data['value'],
                    name=f'VWC {depth}cm',
                    line=dict(width=2)
                ))

        # SWSI
        swsi_data = df[df['variable_name'] == 'swsi']
        if not swsi_data.empty:
            fig3.add_trace(go.Bar(
                x=swsi_data['timestamp'],
                y=swsi_data['value'],
                name='SWSI',
                marker_color='brown',
                opacity=0.7,
                yaxis='y2',
                width=3600000  # 1 hour in milliseconds
            ))

        # Irrigation events
        if not irrigation_df.empty:
            fig3.add_trace(go.Bar(
                x=irrigation_df['date'],
                y=irrigation_df['amount_inches'],
                name='Irrigation',
                marker_color='green',
                yaxis='y2'
            ))

        # Rainfall
        rain_data = df[df['variable_name'] == 'Rain_1m_Tot']
        if not rain_data.empty:
            fig3.add_trace(go.Bar(
                x=rain_data['timestamp'],
                y=rain_data['value'],
                name='Rainfall',
                marker_color='blue',
                opacity=0.6,
                yaxis='y2'
            ))

        fig3.update_layout(
            height=400,
            yaxis=dict(title='VWC (%)', side='left'),
            yaxis2=dict(
                title='Water Input (inches) / SWSI',
                overlaying='y',
                side='right',
                range=[0, 1]
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=60, r=60, t=40, b=40)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Crop2Cloud Platform")

if __name__ == "__main__":
    main()
