import os
import re
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Database path
DB_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\experiment_data_20241024.sqlite"

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

def get_variable_names():
    query = "SELECT DISTINCT variable_name FROM data"
    df = pd.read_sql_query(query, conn)
    return df['variable_name'].tolist()

def parse_sensor_name(sensor_name):
    pattern = r'^(?P<SensorType>[A-Z]{3})(?P<FieldNumber>\d{4})(?P<Node>[A-E])(?P<Treatment>[1256])(?P<Depth>\d{2}|xx)(?P<Timestamp>\d{2})$'
    match = re.match(pattern, sensor_name)
    return match.groupdict() if match else {}

@st.cache_data(ttl=600)
def fetch_data(plot_id, start_date, end_date):
    query = f"""
        SELECT timestamp, variable_name, value
        FROM data
        WHERE plot_id = '{plot_id}'
        AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
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
    return df

def get_sensor_columns(df, sensor_type):
    pattern = rf'^{sensor_type}\d+'
    return [col for col in df['variable_name'].unique() if re.match(pattern, col)]

def map_irrigation_recommendation(recommendation):
    if isinstance(recommendation, (float, int)):
        return float(recommendation)
    elif isinstance(recommendation, str):
        if recommendation.lower() == 'irrigate':
            return 1.0
        elif recommendation.lower() in ["don't irrigate", "dont irrigate"]:
            return 0.0
    return 0.0

def main():
    st.set_page_config(
        page_title="🌾 Irrigation Management Dashboard 🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌾 **CROP2CLOUD Platform** 🌾")

    # Get plot metadata for enhanced selection
    plot_metadata = get_plot_metadata()
    
    # Create plot selection options with metadata
    plot_options = {
        row['plot_id']: f"Plot {row['plot_id']} - {row['crop_type']} ({row['trt_name']})"
        for _, row in plot_metadata.iterrows()
    }

    with st.sidebar:
        st.header("📊 Controls")
        plot_selection = st.selectbox(
            "Select Plot",
            options=list(plot_options.keys()),
            format_func=lambda x: plot_options[x]
        )
        
        current_plot = plot_metadata[plot_metadata['plot_id'] == plot_selection].iloc[0]
        
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

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    with st.spinner("🔍 Fetching data from the database..."):
        df = fetch_data(plot_selection, start_date_str, end_date_str)
        irrigation_df = fetch_irrigation_events(plot_selection, start_date_str, end_date_str)

    if df.empty:
        st.warning("⚠️ No data available for the selected plot and date range.")
        st.stop()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    irrigation_df['date'] = pd.to_datetime(irrigation_df['date'])
    
    data_pivot = df.pivot_table(index='timestamp', columns='variable_name', values='value').reset_index()

    left_col, right_col = st.columns([3, 1], gap="small")

    with left_col:
        grid1, grid2 = st.columns(2, gap="small")
        grid3, grid4 = st.columns(2, gap="small")

        # Top-Left Box: Combined Weather Parameters
        with grid1:
            st.subheader("☀️ **Weather Parameters**")
            weather_params = {
                'Solar_2m_Avg': {'label': 'Solar Radiation (W/m²)', 'unit': 'W/m²'},
                'WndAveSpd_3m': {'label': 'Wind Speed (m/s)', 'unit': 'm/s'}
            }
            available_weather = [param for param in weather_params.keys() if param in data_pivot.columns]
            if available_weather:
                fig = go.Figure()

                if 'Solar_2m_Avg' in available_weather:
                    fig.add_trace(go.Scatter(
                        x=data_pivot['timestamp'],
                        y=data_pivot['Solar_2m_Avg'],
                        mode='lines+markers',
                        name=weather_params['Solar_2m_Avg']['label'],
                        line=dict(color='gold', width=2),
                        yaxis='y'
                    ))

                if 'WndAveSpd_3m' in available_weather:
                    fig.add_trace(go.Scatter(
                        x=data_pivot['timestamp'],
                        y=data_pivot['WndAveSpd_3m'],
                        mode='lines+markers',
                        name=weather_params['WndAveSpd_3m']['label'],
                        line=dict(color='teal', width=2),
                        yaxis='y2'
                    ))

                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    title=f"{current_plot['crop_type']} - {current_plot['trt_name']}",
                    xaxis=dict(title='Timestamp', showgrid=True, gridwidth=1, gridcolor='gray', tickfont=dict(size=12)),
                    yaxis=dict(
                        title='Solar Radiation (W/m²)',
                        titlefont=dict(color='gold', size=14),
                        tickfont=dict(color='gold', size=12),
                        showgrid=False,
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Wind Speed (m/s)',
                        titlefont=dict(color='teal', size=14),
                        tickfont=dict(color='teal', size=12),
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    legend=dict(x=1.05, y=1, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(size=12)),
                    margin=dict(r=200),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No weather data available.")

        # Top-Right Box: Temperature (Air & Canopy)
        with grid2:
            st.subheader("🌡️ **Temperature**")
            
            # Get IRT data for canopy temperature
            irt_columns = get_sensor_columns(df, 'IRT')
            
            fig = go.Figure()
            
            # Add canopy temperature if available
            if irt_columns:
                irt_data = df[df['variable_name'].isin(irt_columns)]
                if not irt_data.empty:
                    fig.add_trace(go.Scatter(
                        x=irt_data['timestamp'],
                        y=irt_data['value'],
                        mode='lines+markers',
                        name='Canopy Temperature',
                        line=dict(color='red', width=2)
                    ))
            
            # Add max/min temperatures
            if 'TaMax_2m' in data_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=data_pivot['timestamp'],
                    y=data_pivot['TaMax_2m'],
                    mode='lines',
                    name='Max Air Temperature',
                    line=dict(color='orange', dash='dash')
                ))
            
            if 'TaMin_2m' in data_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=data_pivot['timestamp'],
                    y=data_pivot['TaMin_2m'],
                    mode='lines',
                    name='Min Air Temperature',
                    line=dict(color='blue', dash='dash')
                ))

            fig.update_layout(
                template='plotly_white',
                height=400,
                title=f"{current_plot['crop_type']} - {current_plot['trt_name']}",
                xaxis=dict(title='Timestamp', showgrid=True, gridwidth=1, gridcolor='gray', tickfont=dict(size=12)),
                yaxis=dict(
                    title='Temperature (°C)',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12),
                    showgrid=False,
                    side='left'
                ),
                legend=dict(x=1.05, y=1, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', font=dict(size=12)),
                margin=dict(r=200),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        # Combined VWC and Irrigation Graph (replacing both previous graphs)
        with grid3:
            st.subheader("💧 **Soil Moisture & Water Inputs**")
            
            tdr_columns = get_sensor_columns(df, 'TDR')
            if tdr_columns:
                # Process TDR data
                depth_info = {}
                for tdr in tdr_columns:
                    parsed = parse_sensor_name(tdr)
                    depth = parsed.get('Depth', 'xx')
                    if depth != 'xx':
                        depth_label = f"Depth {depth} cm"
                    else:
                        depth_label = "Depth N/A"
                    depth_info[tdr] = depth_label

                def sort_key(col):
                    label = depth_info[col]
                    match = re.search(r'\d+', label)
                    return int(match.group()) if match else 0

                sorted_tdr = sorted(tdr_columns, key=sort_key)

                # Create combined figure
                fig = go.Figure()

                # Add VWC traces
                for tdr in sorted_tdr:
                    tdr_df = df[df['variable_name'] == tdr]
                    fig.add_trace(go.Scatter(
                        x=tdr_df['timestamp'],
                        y=tdr_df['value'],
                        mode='lines',
                        name=depth_info[tdr],
                        line=dict(width=2),
                        yaxis='y'
                    ))

                # Add rainfall bars
                if 'Rain_1m_Tot' in data_pivot.columns:
                    rainfall_data = data_pivot[data_pivot['Rain_1m_Tot'] > 0]
                    if not rainfall_data.empty:
                        fig.add_trace(go.Bar(
                            x=rainfall_data['timestamp'],
                            y=rainfall_data['Rain_1m_Tot'],
                            name='Rainfall (mm)',
                            marker_color='blue',
                            opacity=0.7,
                            yaxis='y2'
                        ))

                # Add irrigation events
                if not irrigation_df.empty:
                    fig.add_trace(go.Bar(
                        x=irrigation_df['date'],
                        y=irrigation_df['amount_mm'],
                        name='Irrigation (mm)',
                        marker_color='green',
                        opacity=0.7,
                        yaxis='y2'
                    ))

                fig.update_layout(
                    template='plotly_white',
                    height=600,  # Increased height
                    title=f"{current_plot['crop_type']} - {current_plot['trt_name']}",
                    xaxis=dict(
                        title='Date',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='gray',
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title='Volumetric Water Content (%)',
                        titlefont=dict(size=14),
                        tickfont=dict(size=12),
                        showgrid=True,
                        side='left'
                    ),
                    yaxis2=dict(
                        title='Water Input (mm)',
                        titlefont=dict(size=14),
                        tickfont=dict(size=12),
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        range=[0, max(
                            data_pivot.get('Rain_1m_Tot', pd.Series([0])).max() * 1.2,
                            irrigation_df.get('amount_mm', pd.Series([0])).max() * 1.2,
                            0.1  # Minimum range
                        )]
                    ),
                    legend=dict(
                        x=1.05,
                        y=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)',
                        font=dict(size=12)
                    ),
                    margin=dict(r=200),
                    hovermode='x unified',
                    barmode='overlay'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No soil moisture data available.")

        # Remove the original grid4 content since we've combined the graphs
        with grid4:
            st.empty()

    with right_col:
        # Indices Visualization
        st.subheader("📊 **Irrigation Indices**")

        indices = {
            'cwsi': {'label': 'CWSI', 'color': 'indianred'},
            'swsi': {'label': 'SWSI', 'color': 'teal'},
            'etc': {'label': 'ETC', 'color': 'gold'}
        }

        indices_data = {}
        for key, meta in indices.items():
            # Handle case-insensitive matching
            matching_rows = df[df['variable_name'].str.lower() == key.lower()]
            if not matching_rows.empty:
                latest_value = matching_rows['value'].dropna().iloc[-1]
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
                elif row['Index'].upper() == 'ETC':
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
                template='plotly_white',
                height=400,
                title=f"{current_plot['crop_type']} - {current_plot['trt_name']}",
                barmode='group',
                xaxis=dict(
                    title='Indices',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='gray',
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title='ETC',
                    range=[0, 8],
                    showgrid=False,
                    side='left',
                    tickfont=dict(size=12)
                ),
                yaxis2=dict(
                    title='CWSI & SWSI',
                    range=[0, 2],
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    tickfont=dict(size=12)
                ),
                legend=dict(
                    x=1.05,
                    y=1,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',
                    font=dict(size=12)
                ),
                margin=dict(r=200),
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No indices data available.")

        # Irrigation Recommendation Bar
        st.subheader("💧 **Irrigation Recommendation**")
        recommendation_rows = df[df['variable_name'].str.lower() == 'recommendation']
        if not recommendation_rows.empty:
            recommendation = recommendation_rows['value'].dropna().iloc[-1]
            irrigation_value = map_irrigation_recommendation(recommendation)

            # Enhanced Irrigation Bar using Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=irrigation_value,
                title={
                    'text': f"Recommended Irrigation (inches)\n{current_plot['crop_type']} - {current_plot['trt_name']}", 
                    'font': {'size': 24}
                },
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 2, 'tickcolor': "darkblue"},
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
                },
                number={'font': {'size': 36}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No irrigation recommendation available.")

    # Footer
    st.markdown("---")
    st.markdown("<p class='footer-text'>© 2024 Crop2Cloud24. All rights reserved.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()