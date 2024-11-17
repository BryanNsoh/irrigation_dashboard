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
    """Log diagnostic information"""
    logging.info(message)
    if data is not None:
        logging.info(f"Data: {data}")

# Database path
DB_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\experiment_data_20241117.sqlite"

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at {DB_PATH}.")
    st.stop()

# Initialize SQLite connection
conn = sqlite3.connect(DB_PATH)

# Custom styling for plots
PLOT_STYLE = {
    'font_size': 16,
    'title_size': 18,
    'legend_size': 14,
    'line_width': 3,
    'marker_size': 8,
    'bar_width': 1800000,  # 30 minutes in milliseconds
    'opacity': 0.8
}

def get_plot_metadata():
    """Get plot metadata including treatment and crop info"""
    query = """
    SELECT p.plot_id, p.treatment, y.trt_name, y.crop_type,
           y.irrigation_applied_inches, y.avg_yield_bu_ac
    FROM plots p
    LEFT JOIN yields y ON p.plot_id = y.plot_id
    """
    return pd.read_sql_query(query, conn)

def get_date_range():
    """Get full date range from database"""
    query = """
    SELECT MIN(DATE(timestamp)) as start_date, 
           MAX(DATE(timestamp)) as end_date 
    FROM data
    """
    return pd.read_sql_query(query, conn)

def parse_sensor_name(sensor_name):
    pattern = r'^(?P<SensorType>[A-Z]{3})(?P<FieldNumber>\d{4})(?P<Node>[A-E])(?P<Treatment>[1-4])(?P<Depth>\d{2}|xx)24$'
    match = re.match(pattern, sensor_name)
    return match.groupdict() if match else {}

@st.cache_data(ttl=600)
def fetch_data(plot_id, start_date, end_date):
    """Fetch and clean data"""
    query = f"""
        SELECT timestamp, variable_name, value
        FROM data
        WHERE plot_id = '{plot_id}'
        AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
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

def calculate_summary_stats(df, irrigation_df):
    """Calculate summary statistics for the selected period"""
    stats = {}
    
    if not df.empty:
        for var in ['etc', 'cwsi', 'swsi']:
            var_data = df[df['variable_name'] == var]
            if not var_data.empty:
                stats[var] = {
                    'mean': var_data['value'].mean(),
                    'max': var_data['value'].max(),
                    'min': var_data['value'].min()
                }
        
        # Calculate cumulative values
        rain_data = df[df['variable_name'] == 'Rain_1m_Tot']
        if not rain_data.empty:
            stats['total_rain'] = rain_data['value'].sum()
            
    if not irrigation_df.empty:
        stats['total_irrigation'] = irrigation_df['amount_inches'].sum()
        
    return stats

def create_weather_plot(df_pivot, style=PLOT_STYLE):
    """Create weather conditions plot"""
    fig = go.Figure()

    # Update trace names to include units
    fig.add_trace(go.Scatter(
        x=df_pivot['timestamp'],
        y=df_pivot['Solar_2m_Avg'],
        name='Solar Radiation (W/m²)',
        line=dict(color='gold', width=style['line_width']),
        yaxis='y'
    ))

    if 'etc' in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot['timestamp'],
            y=df_pivot['etc'],
            name='ETC (mm)',
            line=dict(color='red', width=style['line_width']),
            yaxis='y2'
        ))

    if 'WndAveSpd_3m' in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot['timestamp'],
            y=df_pivot['WndAveSpd_3m'],
            name='Wind Speed (m/s)',
            line=dict(color='blue', width=style['line_width']),
            yaxis='y3'
        ))

    if 'RH_2m_Avg' in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot['timestamp'],
            y=df_pivot['RH_2m_Avg'],
            name='RH (%)',
            line=dict(color='green', width=style['line_width']),
            yaxis='y4'
        ))

    fig.update_layout(
        height=300,
        yaxis=dict(
            title=None,
            side='left',
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size'], color='gold'),
            showgrid=True,
            domain=[0, 0.85]
        ),
        yaxis2=dict(
            title=None,
            overlaying='y',
            side='left',
            position=0.05,
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size'], color='red'),
            anchor='free'
        ),
        yaxis3=dict(
            title=None,
            overlaying='y',
            side='right',
            position=0.95,
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size'], color='blue'),
            anchor='free'
        ),
        yaxis4=dict(
            title=None,
            overlaying='y',
            side='right',
            position=1.0,
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size'], color='green'),
            anchor='free'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            font=dict(size=style['legend_size'])
        ),
        margin=dict(l=80, r=80, t=40, b=40),  # Reduced margins
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(domain=[0.05, 0.95])  # Expanded plot width
    )
    
    return fig

def create_temperature_plot(df, df_pivot, style=PLOT_STYLE):
    """Create temperature and CWSI plot"""
    fig = go.Figure()

    # Canopy temperature (IRT sensors)
    irt_columns = [col for col in df['variable_name'].unique() if col.startswith('IRT')]
    if irt_columns:
        for irt in irt_columns:
            irt_data = df[df['variable_name'] == irt]
            fig.add_trace(go.Scatter(
                x=irt_data['timestamp'],
                y=irt_data['value'],
                name='Canopy Temperature',
                line=dict(color='red', width=style['line_width']),
                mode='lines'
            ))

    # Air temperature
    if 'Ta_2m_Avg' in df_pivot.columns:
        fig.add_trace(go.Scatter(
            x=df_pivot['timestamp'],
            y=df_pivot['Ta_2m_Avg'],
            name='Air Temperature',
            line=dict(color='orange', width=style['line_width']),
            mode='lines'
        ))

    # CWSI as thin bars
    cwsi_data = df[df['variable_name'] == 'cwsi'].copy()
    if not cwsi_data.empty:
        cwsi_data['value'] = cwsi_data['value'].apply(normalize_cwsi)
        cwsi_data = cwsi_data.dropna(subset=['value'])
        if not cwsi_data.empty:
            fig.add_trace(go.Bar(
                x=cwsi_data['timestamp'],
                y=cwsi_data['value'],
                name='CWSI',
                marker_color='purple',
                opacity=style['opacity'],
                yaxis='y2',
                width=style['bar_width']
            ))

    fig.update_layout(
        height=300,
        yaxis=dict(
            title='Temperature (°C)',
            side='left',
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size'])
        ),
        yaxis2=dict(
            title='CWSI',
            overlaying='y',
            side='right',
            range=[0, 1],
            titlefont=dict(size=style['font_size'], color='purple'),
            tickfont=dict(size=style['font_size'], color='purple')
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            font=dict(size=style['legend_size'])
        ),
        margin=dict(l=60, r=60, t=40, b=40)
    )
    
    return fig

def create_water_management_plot(df, irrigation_df, style=PLOT_STYLE):
    """Create soil moisture and water management plot"""
    fig = go.Figure()

    # TDR sensors - Reduce line width and adjust opacity
    tdr_columns = [col for col in df['variable_name'].unique() if col.startswith('TDR')]
    colors = px.colors.qualitative.Set2
    if tdr_columns:
        for i, tdr in enumerate(tdr_columns):
            parsed = parse_sensor_name(tdr)
            depth = parsed.get('Depth', 'xx')
            tdr_data = df[df['variable_name'] == tdr]
            fig.add_trace(go.Scatter(
                x=tdr_data['timestamp'],
                y=tdr_data['value'],
                name=f'VWC {depth}cm',
                line=dict(
                    color=colors[i % len(colors)], 
                    width=style['line_width'] - 1,  # Slightly thinner lines
                    dash='solid'  # You could also use 'dot' or 'dash' for some depths
                ),
                opacity=0.8,  # Slightly more transparent
                mode='lines'
            ))

    # SWSI as continuous line - Make it stand out more
    swsi_data = df[df['variable_name'] == 'swsi']
    if not swsi_data.empty:
        fig.add_trace(go.Scatter(
            x=swsi_data['timestamp'],
            y=swsi_data['value'],
            name='SWSI',
            line=dict(
                color='brown', 
                width=style['line_width'] + 1,  # Slightly thicker
                dash='solid'
            ),
            opacity=1,  # Full opacity
            mode='lines',
            yaxis='y2'
        ))

    # Constants for bar widths
    ONE_DAY_MS = 86400000  # milliseconds in a day
    RAIN_WIDTH = ONE_DAY_MS * 0.7  # Reduced from 0.98 to 0.7
    IRR_WIDTH = ONE_DAY_MS * 0.7   # Match rainfall width

    # Rainfall bars - Adjust opacity and width
    rain_data = df[df['variable_name'] == 'Rain_1m_Tot'].copy()
    if not rain_data.empty:
        rain_data['date'] = rain_data['timestamp'].dt.date
        daily_rain = rain_data.groupby('date')['value'].sum().reset_index()
        daily_rain['date'] = pd.to_datetime(daily_rain['date'])
        
        fig.add_trace(go.Bar(
            x=daily_rain['date'],
            y=daily_rain['value'],
            name='Rainfall',
            marker_color='rgba(0, 0, 255, 0.5)',  # Lighter blue with transparency
            yaxis='y2',
            width=RAIN_WIDTH
        ))

    # Irrigation bars - Adjust opacity and width
    if not irrigation_df.empty:
        fig.add_trace(go.Bar(
            x=irrigation_df['date'],
            y=irrigation_df['amount_inches'],
            name='Irrigation',
            marker_color='rgba(0, 128, 0, 0.5)',  # Lighter green with transparency
            yaxis='y2',
            width=IRR_WIDTH
        ))

    # Update layout with better spacing and grid
    fig.update_layout(
        height=400,  # Increased height
        yaxis=dict(
            title='VWC (%)',
            side='left',
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size']),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',  # Light grid
            zeroline=False
        ),
        yaxis2=dict(
            title='Water Input (inches) / SWSI',
            overlaying='y',
            side='right',
            range=[0, 1],
            titlefont=dict(size=style['font_size']),
            tickfont=dict(size=style['font_size']),
            showgrid=False
        ),
        plot_bgcolor='white',  # White background
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            font=dict(size=style['legend_size']),
            bgcolor='rgba(255, 255, 255, 0.8)'  # Semi-transparent white background
        ),
        margin=dict(l=60, r=60, t=40, b=40)
    )
    
    return fig

def create_date_slider():
    """Create date range slider"""
    date_range = get_date_range()
    start = pd.to_datetime(date_range['start_date'].iloc[0])
    end = pd.to_datetime(date_range['end_date'].iloc[0])
    
    # Create list of dates for slider
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Convert dates to ints for slider
    date_to_int = {date: i for i, date in enumerate(dates)}
    int_to_date = {i: date for date, i in date_to_int.items()}
    
    # Create slider with better formatting
    st.markdown("### Select Date Range")
    cols = st.columns([1, 3, 1])
    with cols[1]:
        selected = st.select_slider(
            "",  # Remove label since we have the markdown header
            options=list(date_to_int.values()),
            value=(0, len(dates)-1),
            format_func=lambda x: "",  # Hide the dates on slider itself
            key="date_slider"
        )
        
        # Display selected dates separately below slider
        start_date = int_to_date[selected[0]]
        end_date = int_to_date[selected[1]]
        
        # Display dates below with better spacing
        st.markdown(
            f"<div style='display: flex; justify-content: space-between; margin-top: 10px;'>"
            f"<span>Start: {start_date.strftime('%Y-%m-%d')}</span>"
            f"<span>End: {end_date.strftime('%Y-%m-%d')}</span>"
            "</div>",
            unsafe_allow_html=True
        )
    
    return start_date, end_date

def main():
    st.set_page_config(
        page_title="Crop2Cloud Platform",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🌾 **CROP2CLOUD Platform** ����")

    # Get plot metadata
    plot_metadata = get_plot_metadata()
    
    # Create plot selection options with metadata
    plot_options = {
        row['plot_id']: f"Plot {row['plot_id']} - {row['crop_type'].title()} ({row['trt_name']})"
        for _, row in plot_metadata.iterrows()
    }

    # Layout columns
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.header("Plot Selection & Summary")
        plot_selection = st.selectbox(
            "Select Plot",
            options=list(plot_options.keys()),
            format_func=lambda x: plot_options[x]
        )
        
        # Date range slider
        start_date, end_date = create_date_slider()

        current_plot = plot_metadata[plot_metadata['plot_id'] == plot_selection].iloc[0]

        # Display plot summary with enhanced information
        st.subheader("Plot Information")
        st.write(f"**Crop Type:** {current_plot['crop_type'].title()}")
        st.write(f"**Treatment:** {current_plot['trt_name']}")
        st.write(f"**Treatment Number:** {current_plot['treatment']}")
        st.write(f"**Total Irrigation Applied:** {current_plot['irrigation_applied_inches']:.2f} inches")
        st.write(f"**Yield:** {current_plot['avg_yield_bu_ac']:.1f} bu/ac")

    # Fetch and process data
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    with st.spinner("Loading data..."):
        df = fetch_data(plot_selection, start_date_str, end_date_str)
        irrigation_df = fetch_irrigation_events(plot_selection, start_date_str, end_date_str)
        
        if df.empty:
            st.warning("No data available for the selected period.")
            st.stop()
            
        # Calculate summary statistics
        stats = calculate_summary_stats(df, irrigation_df)
        
        # Create pivot table for plotting
        df_pivot = df.pivot(index='timestamp', columns='variable_name', values='value').reset_index()

        # Display summary statistics in left column
        with left_col:
            st.subheader("Period Summary")
            if 'etc' in stats:
                st.write(f"**Mean ETC:** {stats['etc']['mean']:.2f} mm")
            if 'cwsi' in stats:
                st.write(f"**Mean CWSI:** {stats['cwsi']['mean']:.2f}")
            if 'total_rain' in stats:
                st.write(f"**Total Rainfall:** {stats['total_rain']:.2f} inches")
            if 'total_irrigation' in stats:
                st.write(f"**Total Irrigation:** {stats['total_irrigation']:.2f} inches")

        # Create and display plots in right column
        with right_col:
            st.plotly_chart(create_weather_plot(df_pivot), use_container_width=True)
            st.plotly_chart(create_temperature_plot(df, df_pivot), use_container_width=True)
            st.plotly_chart(create_water_management_plot(df, irrigation_df), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Crop2Cloud Platform")

if __name__ == "__main__":
    main()
