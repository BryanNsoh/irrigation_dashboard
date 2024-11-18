import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import logging

# Database connection
DB_PATH = r"C:\Users\bnsoh2\OneDrive - University of Nebraska-Lincoln\Documents\Projects\masters-project\cwsi-swsi-et\experiment_data_20241117.sqlite"
conn = sqlite3.connect(DB_PATH)

# Set page config
st.set_page_config(page_title="Water Use Efficiency Analysis", layout="wide")
st.title("Agricultural Water Use Efficiency Analysis")

# Date range for middle growth phase
START_DATE = '2024-07-30'
END_DATE = '2024-08-20'

def calculate_water_efficiency_metrics():
    # Get yields and irrigation data
    yields_query = """
    SELECT y.*, p.field, p.treatment
    FROM yields y
    JOIN plots p ON y.plot_id = p.plot_id
    """
    yields_df = pd.read_sql_query(yields_query, conn)
    
    # Get ETc data for the period
    etc_query = f"""
    SELECT plot_id, SUM(value) as total_etc
    FROM data
    WHERE variable_name = 'etc'
    AND date(timestamp) BETWEEN '{START_DATE}' AND '{END_DATE}'
    GROUP BY plot_id
    """
    etc_df = pd.read_sql_query(etc_query, conn)
    
    # Get precipitation data
    precip_query = f"""
    SELECT plot_id, SUM(value) * 25.4 as total_precip_mm
    FROM data
    WHERE variable_name = 'Rain_1m_Tot'
    AND date(timestamp) BETWEEN '{START_DATE}' AND '{END_DATE}'
    GROUP BY plot_id
    """
    precip_df = pd.read_sql_query(precip_query, conn)
    
    # Merge all data
    results_df = yields_df.merge(etc_df, on='plot_id').merge(precip_df, on='plot_id')
    
    # Calculate effective precipitation (85% of total)
    results_df['effective_precip_mm'] = results_df['total_precip_mm'] * 0.85
    
    # Calculate efficiency metrics
    results_df['EWUE'] = results_df['yield_kg_ha'] / (results_df['irrigation_applied_mm'] + results_df['effective_precip_mm'])
    results_df['IAE'] = (results_df['total_etc'] - results_df['effective_precip_mm']) / results_df['irrigation_applied_mm'] * 100
    results_df['CWP'] = results_df['yield_kg_ha'] / results_df['total_etc']
    
    return results_df

def create_efficiency_plots(df):
    # Create figure for EWUE
    fig_ewue = px.box(df, x='field', y='EWUE', color='trt_name',
                      title='Effective Water Use Efficiency by Crop and Treatment',
                      labels={'field': 'Crop Type', 'EWUE': 'EWUE (kg/ha/mm)',
                             'trt_name': 'Treatment'})
    
    # Create figure for IAE
    fig_iae = px.box(df, x='field', y='IAE', color='trt_name',
                     title='Irrigation Application Efficiency by Crop and Treatment',
                     labels={'field': 'Crop Type', 'IAE': 'IAE (%)',
                            'trt_name': 'Treatment'})
    
    # Create figure for CWP
    fig_cwp = px.box(df, x='field', y='CWP', color='trt_name',
                     title='Crop Water Productivity by Crop and Treatment',
                     labels={'field': 'Crop Type', 'CWP': 'CWP (kg/ha/mm)',
                            'trt_name': 'Treatment'})
    
    return fig_ewue, fig_iae, fig_cwp

def main():
    try:
        # Calculate metrics
        results_df = calculate_water_efficiency_metrics()
        
        # Display summary statistics
        st.subheader("Water Use Efficiency Metrics Summary")
        summary_stats = results_df.groupby(['field', 'trt_name'])[['EWUE', 'IAE', 'CWP']].agg(['mean', 'std']).round(3)
        st.dataframe(summary_stats)
        
        # Create and display plots
        fig_ewue, fig_iae, fig_cwp = create_efficiency_plots(results_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig_ewue, use_container_width=True)
        with col2:
            st.plotly_chart(fig_iae, use_container_width=True)
        with col3:
            st.plotly_chart(fig_cwp, use_container_width=True)
        
        # Display raw data in expandable section
        with st.expander("View Raw Data"):
            st.dataframe(results_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
