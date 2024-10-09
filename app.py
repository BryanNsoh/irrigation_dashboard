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

def find_nearest_data(dataset, table, selected_start, selected_end, client, required_column, max_days=30):
    """
    Finds the nearest date range with available data by shifting the selected range
    forward and backward day by day, up to a maximum number of days.
    Additionally, ensures that the required_column has at least one non-null value.
    
    Returns:
        new_start (datetime): New start date with available data.
        new_end (datetime): New end date with available data.
        df (pd.DataFrame): DataFrame containing the fetched data.
        direction (str): 'before' or 'after' indicating the direction of the shift.
    """
    for delta in range(1, max_days + 1):
        # Check before the selected range
        new_start_before = selected_start - timedelta(days=delta)
        new_end_before = selected_end - timedelta(days=delta)
        df_before = fetch_data(dataset, table, new_start_before.strftime('%Y-%m-%d'), new_end_before.strftime('%Y-%m-%d'))
        if not df_before.empty and required_column in df_before.columns and df_before[required_column].notnull().any():
            return new_start_before, new_end_before, df_before, 'before'

        # Check after the selected range
        new_start_after = selected_start + timedelta(days=delta)
        new_end_after = selected_end + timedelta(days=delta)
        df_after = fetch_data(dataset, table, new_start_after.strftime('%Y-%m-%d'), new_end_after.strftime('%Y-%m-%d'))
        if not df_after.empty and required_column in df_after.columns and df_after[required_column].notnull().any():
            return new_start_after, new_end_after, df_after, 'after'

    return None, None, None, None

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

    st.title("🌾 **CROP2CLOUD Platform** 🌾")

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

    # Handle empty data by finding the nearest available data range
    if df.empty or ('Solar_2m_Avg' not in df.columns) or df['Solar_2m_Avg'].isnull().all():
        # Define the required column for non-null data
        required_column = 'Solar_2m_Avg'
        new_start, new_end, df_new, direction = find_nearest_data(
            dataset, table, start_date, end_date, client, required_column
        )
        if df_new is not None:
            if direction == 'before':
                st.info(
                    f"No data found for the selected date range with valid **Solar Radiation**. "
                    f"Showing data from **{new_start.strftime('%Y-%m-%d')}** to **{new_end.strftime('%Y-%m-%d')}** (shifted earlier)."
                )
            else:
                st.info(
                    f"No data found for the selected date range with valid **Solar Radiation**. "
                    f"Showing data from **{new_start.strftime('%Y-%m-%d')}** to **{new_end.strftime('%Y-%m-%d')}** (shifted later)."
                )
            start_date = new_start
            end_date = new_end
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            df = df_new
        else:
            st.warning(
                "⚠️ No data available for the selected plot and date range, even after searching nearby dates "
                "with valid **Solar Radiation** data."
            )
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

                # Add Solar Radiation trace (Primary Y-Axis)
                if 'Solar_2m_Avg' in available_weather:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df['Solar_2m_Avg'],
                        mode='lines+markers',
                        name=weather_params['Solar_2m_Avg']['label'],
                        line=dict(color='gold', width=2),
                        yaxis='y'
                    ))

                # Add Wind Speed trace (Secondary Y-Axis)
                if 'WndAveSpd_3m' in available_weather:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df['WndAveSpd_3m'],
                        mode='lines+markers',
                        name=weather_params['WndAveSpd_3m']['label'],
                        line=dict(color='teal', width=2),
                        yaxis='y2'
                    ))

                # Add Relative Humidity trace (Tertiary Y-Axis)
                if 'RH_2m_Avg' in available_weather:
                    fig.add_trace(go.Scatter(
                        x=df['TIMESTAMP'],
                        y=df['RH_2m_Avg'],
                        mode='lines+markers',
                        name=weather_params['RH_2m_Avg']['label'],
                        line=dict(color='dodgerblue', width=2),
                        yaxis='y3'
                    ))

                # Update layout for multiple y-axes
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    xaxis=dict(
                        title='Timestamp',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='gray'
                    ),
                    yaxis=dict(
                        title='Solar Radiation (W/m²)',
                        titlefont=dict(color='gold'),
                        tickfont=dict(color='gold'),
                        showgrid=False,
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Wind Speed (m/s)',
                        titlefont=dict(color='teal'),
                        tickfont=dict(color='teal'),
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    yaxis3=dict(
                        title='Relative Humidity (%)',
                        titlefont=dict(color='dodgerblue'),
                        tickfont=dict(color='dodgerblue'),
                        overlaying='y',
                        side='right',
                        position=0.95,  # Adjusted to be within [0,1]
                        showgrid=False
                    ),
                    legend=dict(
                        x=1.05,
                        y=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)'
                    ),
                    margin=dict(r=200),  # Increased right margin to accommodate legend and yaxis3
                    hovermode='x unified'
                )

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
                    height=400,
                    xaxis=dict(
                        title='Timestamp',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='gray'
                    ),
                    yaxis=dict(
                        title='Temperature (°C)',
                        showgrid=False,
                        side='left'
                    ),
                    legend=dict(
                        x=1.05,
                        y=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)'
                    ),
                    margin=dict(r=200),  # Increased right margin to accommodate legend
                    hovermode='x unified'
                )
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
                        height=400,
                        xaxis=dict(
                            title='Timestamp',
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='gray'
                        ),
                        yaxis=dict(
                            title='Rainfall (mm)',
                            showgrid=False,
                            side='left'
                        ),
                        legend=dict(
                            x=1.05,
                            y=1,
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='rgba(0,0,0,0)'
                        ),
                        margin=dict(r=200),  # Increased right margin to accommodate legend
                        hovermode='x unified'
                    )
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
                    height=400,
                    xaxis=dict(
                        title='Timestamp',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='gray'
                    ),
                    yaxis=dict(
                        title='Volumetric Water Content (%)',
                        showgrid=False,
                        side='left'
                    ),
                    legend=dict(
                        x=1.05,
                        y=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)'
                    ),
                    margin=dict(r=200),  # Increased right margin to accommodate legend
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No TDR data available.")

    with right_col:
        # Indices Visualization
        st.subheader("📊 **Irrigation Indices**")

        indices = {
            'cwsi': {'label': 'CWSI', 'color': 'indianred'},
            'swsi': {'label': 'SWSI', 'color': 'teal'},
            'eto': {'label': 'ETO', 'color': 'gold'}
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

            # Create figure with secondary y-axis
            fig = go.Figure()

            for _, row in indices_df.iterrows():
                if row['Index'].upper() in ['CWSI', 'SWSI']:
                    # Assign to secondary y-axis
                    fig.add_trace(go.Bar(
                        x=[row['Index']],
                        y=[row['Value']],
                        name=row['Index'],
                        marker_color=indices[row['Index'].lower()]['color'],
                        text=[f"{row['Value']:.2f}"],
                        textposition='auto',
                        width=0.5,
                        yaxis='y2'
                    ))
                elif row['Index'].upper() == 'ETO':
                    # Assign to primary y-axis
                    fig.add_trace(go.Bar(
                        x=[row['Index']],
                        y=[row['Value']],
                        name=row['Index'],
                        marker_color=indices[row['Index'].lower()]['color'],
                        text=[f"{row['Value']:.2f}"],
                        textposition='auto',
                        width=0.5,
                        yaxis='y'
                    ))

            # Update layout with two y-axes
            fig.update_layout(
                template='plotly_dark',
                height=400,
                barmode='group',
                xaxis=dict(
                    title='Indices',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='gray'
                ),
                yaxis=dict(
                    title='ETO',
                    range=[0, 8],
                    showgrid=False,
                    side='left'
                ),
                yaxis2=dict(
                    title='CWSI & SWSI',
                    range=[0, 2],
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(
                    x=1.05,
                    y=1,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)'
                ),
                margin=dict(r=200),  # Increased right margin to accommodate legend
                hovermode='x unified'
            )

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
                    height=400
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
