# app.py

import os
import re
import json
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# -------------------- Configuration --------------------

# Load environment variables
load_dotenv()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not GOOGLE_APPLICATION_CREDENTIALS:
    st.error("GOOGLE_APPLICATION_CREDENTIALS not set in .env file.")
    st.stop()

# Initialize BigQuery client
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_APPLICATION_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# -------------------- Helper Functions --------------------

def get_all_plots():
    """
    Retrieves all unique plot names from the BigQuery datasets.
    """
    datasets = [
        'LINEAR_CORN_trt1', 'LINEAR_CORN_trt2', 'LINEAR_CORN_trt3', 'LINEAR_CORN_trt4',
        'LINEAR_SOYBEAN_trt1', 'LINEAR_SOYBEAN_trt2', 'LINEAR_SOYBEAN_trt3', 'LINEAR_SOYBEAN_trt4'
    ]
    plot_list = []
    for dataset in datasets:
        query = f"""
            SELECT DISTINCT table_id
            FROM `{client.project}.{dataset}.__TABLES_SUMMARY__`
        """
        query_job = client.query(query)
        results = query_job.result()
        for row in results:
            plot_name = row.table_id
            plot_list.append(f"{dataset}.{plot_name}")
    return sorted(plot_list)

def parse_sensor_name(sensor_name):
    """
    Parses the sensor name based on the provided nomenclature.
    Example: TDR5006B10624 -> {
        'Sensor Type': 'TDR',
        'Field Number': '5006',
        'Node': 'B',
        'Treatment': '1',
        'Depth': '06',
        'Timestamp': '24'
    }
    """
    pattern = r'^(?P<SensorType>[A-Z]{3})(?P<FieldNumber>\d{4})(?P<Node>[A-E])(?P<Treatment>[1256])(?P<Depth>\d{2}|xx)(?P<Timestamp>\d{2})$'
    match = re.match(pattern, sensor_name)
    if match:
        return match.groupdict()
    else:
        return {}

@st.cache_data(ttl=600)
def fetch_data(dataset, table, start_date, end_date):
    """
    Fetches data from a specific BigQuery table within the date range.
    """
    query = f"""
        SELECT *
        FROM `{client.project}.{dataset}.{table}`
        WHERE TIMESTAMP BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY TIMESTAMP ASC
    """
    df = client.query(query).to_dataframe()
    return df

def get_sensor_columns(df, sensor_type):
    """
    Retrieves columns related to a specific sensor type.
    """
    pattern = rf'^{sensor_type}\d+'
    return [col for col in df.columns if re.match(pattern, col)]

def map_irrigation_recommendation(recommendation):
    """
    Maps irrigation recommendation to a float between 0 and 1.
    """
    if isinstance(recommendation, float) or isinstance(recommendation, int):
        return float(recommendation)
    elif isinstance(recommendation, str):
        if recommendation.lower() == 'irrigate':
            return 1.0
        elif recommendation.lower() in ["don't irrigate", "dont irrigate"]:
            return 0.0
    return 0.0  # Default fallback

# -------------------- Streamlit App --------------------

def main():
    # Set page configuration
    st.set_page_config(
        page_title="🌾 Irrigation Management Dashboard 🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply dark theme CSS
    st.markdown(
        """
        <style>
        .css-18e3th9 {
            background-color: #0e1117;
        }
        .css-1d391kg {
            color: #ffffff;
        }
        .css-1v0mbdj {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌾 **Irrigation Management Dashboard** 🌾")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("📊 Controls")
        plot_selection = st.selectbox("Select Plot", options=get_all_plots())
        date_range = st.date_input("Select Date Range", [
            datetime.today() - timedelta(days=7),
            datetime.today()
        ])
        if len(date_range) == 1:
            start_date = date_range[0]
            end_date = datetime.today()
        elif len(date_range) == 2:
            start_date, end_date = date_range
        else:
            st.error("Please select a valid date range.")
            st.stop()
        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()

    # Extract dataset and table from plot_selection
    try:
        dataset, table = plot_selection.split('.')
    except ValueError:
        st.error("Invalid plot selection.")
        st.stop()

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Fetch data
    with st.spinner("🔍 Fetching data from BigQuery..."):
        df = fetch_data(dataset, table, start_date_str, end_date_str)

    if df.empty:
        st.warning("⚠️ No data available for the selected plot and date range.")
        st.stop()

    # Parse timestamps
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # -------------------- Layout --------------------

    # Create two main columns: Left for grids, Right for indices and irrigation
    left_col, right_col = st.columns([3, 1], gap="small")

    with left_col:
        # Create a 2x2 grid within the left column
        grid1, grid2 = st.columns(2, gap="small")
        grid3, grid4 = st.columns(2, gap="small")

        # Top-Left Box: Combined Weather Parameters
        with grid1:
            st.subheader("☀️ **Weather Parameters**")
            weather_params = {
                'Solar_2m_Avg': {'label': 'Solar Radiation (W/m²)', 'unit': 'W/m²'},
                'WndAveSpd_3m': {'label': 'Wind Speed (m/s)', 'unit': 'm/s'},
                'RH_2m_Avg': {'label': 'Relative Humidity (%)', 'unit': '%'}
            }
            available_weather = [param for param in weather_params.keys() if param in df.columns]
            if available_weather:
                fig = go.Figure()
                for param in available_weather:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df[param],
                        mode='lines+markers',
                        name=weather_params[param]['label'],
                        line=dict(width=2)
                    ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    xaxis_title='Timestamp',
                    yaxis_title='Value',
                    legend_title='Parameters',
                    hovermode='x unified'
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No weather data available.")

        # Top-Right Box: Temperature (Air & Canopy)
        with grid2:
            st.subheader("🌡️ **Temperature**")
            temp_params = {
                'Ta_2m_Avg': {'label': 'Canopy Temperature (°C)', 'unit': '°C'},
                'TaMax_2m': {'label': 'Max Air Temperature (°C)', 'unit': '°C'},
                'TaMin_2m': {'label': 'Min Air Temperature (°C)', 'unit': '°C'}
            }
            available_temps = [param for param in temp_params.keys() if param in df.columns]
            if available_temps:
                fig = go.Figure()
                for param in available_temps:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df[param],
                        mode='lines+markers',
                        name=temp_params[param]['label'],
                        line=dict(width=2)
                    ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    xaxis_title='Timestamp',
                    yaxis_title='Temperature (°C)',
                    legend_title='Temperature Metrics',
                    hovermode='x unified'
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No temperature data available.")

        # Bottom-Left Box: Precipitation
        with grid3:
            st.subheader("🌧️ **Precipitation**")
            precip_param = 'Rain_1m_Tot'
            if precip_param in df.columns:
                # Check if there's variation in precipitation
                if df[precip_param].nunique() > 1:
                    fig = px.bar(
                        df,
                        x='TIMESTAMP',
                        y=precip_param,
                        title="Rainfall (1m Total)",
                        labels={'Rain_1m_Tot': 'Rainfall (mm)'},
                        template='plotly_dark'
                    )
                    fig.update_layout(
                        height=300,
                        xaxis_title='Timestamp',
                        yaxis_title='Rainfall (mm)',
                        hovermode='x unified'
                    )
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No rainfall occurred during the selected period.")
            else:
                st.write("No precipitation data available.")

        # Bottom-Right Box: Volumetric Water Content
        with grid4:
            st.subheader("💧 **Volumetric Water Content**")
            tdr_columns = get_sensor_columns(df, 'TDR')
            if tdr_columns:
                # Extract depth information and sort
                depth_info = {}
                for tdr in tdr_columns:
                    parsed = parse_sensor_name(tdr)
                    depth = parsed.get('Depth', 'xx')
                    if depth != 'xx':
                        depth_label = f"Depth {depth} inches"
                    else:
                        depth_label = "Depth N/A"
                    depth_info[tdr] = depth_label

                # Sort columns by depth (numeric sort)
                def sort_key(col):
                    label = depth_info[col]
                    match = re.search(r'\d+', label)
                    return int(match.group()) if match else 0

                sorted_tdr = sorted(tdr_columns, key=sort_key)

                fig = go.Figure()
                for tdr in sorted_tdr:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df[tdr],
                        mode='lines+markers',
                        name=depth_info[tdr],
                        line=dict(width=2)
                    ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300,
                    xaxis_title='Timestamp',
                    yaxis_title='Volumetric Water Content (%)',
                    legend_title='Depths',
                    hovermode='x unified'
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No TDR data available.")

    with right_col:
        # Indices Visualization
        st.subheader("📊 **Irrigation Indices**")

        indices = {
            'cwsi': {'label': 'CWSI', 'color': 'indianred'},
            'swsi': {'label': 'SWSI', 'color': 'teal'},
            'eto': {'label': 'eto', 'color': 'gold'}
        }

        indices_data = {}
        for key, meta in indices.items():
            # Handle case-insensitive matching
            matching_cols = [col for col in df.columns if col.lower() == key.lower()]
            if matching_cols:
                col = matching_cols[0]
                if not df[col].dropna().empty:
                    latest_value = df[col].dropna().iloc[-1]
                    indices_data[meta['label']] = latest_value

        if indices_data:
            indices_df = pd.DataFrame({
                'Index': list(indices_data.keys()),
                'Value': list(indices_data.values())
            })

            # Determine the maximum value for scaling
            max_value = max(indices_data.values()) if indices_data else 1

            # Create bar chart
            fig = go.Figure()
            for _, row in indices_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['Index']],
                    y=[row['Value']],
                    name=row['Index'],
                    marker_color=indices[row['Index'].lower()]['color'],
                    text=[f"{row['Value']:.2f}"],
                    textposition='auto',
                    width=0.5
                ))
            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title='Indices',
                yaxis_title='Value',
                showlegend=False,
                hovermode='x unified',
                yaxis=dict(range=[0, max_value * 1.2])  # Add some headroom
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No indices data available.")

        # Irrigation Recommendation Bar
        st.subheader("💧 **Irrigation Recommendation**")
        recommendation_col = [col for col in df.columns if col.lower() == 'recommendation']
        if recommendation_col:
            recommendation_col = recommendation_col[0]
            if not df[recommendation_col].dropna().empty:
                recommendation = df[recommendation_col].dropna().iloc[-1]
                irrigation_value = map_irrigation_recommendation(recommendation)
                
                # Enhanced Irrigation Bar using Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=irrigation_value,
                    title={'text': "Recommended Irrigation (inches)"},
                    gauge={
                        'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "lightskyblue", 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightcoral"},
                            {'range': [0.5, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': irrigation_value
                        }
                    }
                ))
                fig.update_layout(
                    template='plotly_dark',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No irrigation recommendation available.")
        else:
            st.write("No irrigation recommendation data available.")

    # -------------------- Footer --------------------
    st.markdown("---")
    st.markdown("© 2024 Crop2Cloud24. All rights reserved.")

if __name__ == "__main__":
    main()
