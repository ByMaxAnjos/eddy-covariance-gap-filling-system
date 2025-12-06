import streamlit as st
import base64
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from xgboost import XGBRegressor
import os
import sys
import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Tuple, Dict, Union, Any
import requests
from streamlit_option_menu import option_menu
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
import re
import io
from eddy_functions import upload_zip_and_extract_csv,load_example_data, create_time_features, create_lag_features, create_rolling_features, calculate_vpd, create_met_features, encode_categorical_features
from eddy_functions import train_model, introduce_nan, plot_flux_partitioning
from eddy_functions import detect_and_preprocess_dataset


# Set page configuration
st.set_page_config(
    page_title="Eddy Covariance Gap-Filling System",
    page_icon="static/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('static/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define app state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'filled_data' not in st.session_state:
    st.session_state.filled_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

# Define color palette
colors = {
    "primary": "#1E5631",
    "secondary": "#4A8B41",
    "accent": "#88B04B",
    "neutral": "#F5F5F5",
    "text": "#333333",
    "highlight": "#3498DB"
}

# Helper functions for UI components
def create_metric_card(title, value, delta=None, delta_color="normal"):
    with stylable_container(
        key=f"metric_{title}",
        css_styles="""
            {
                background: linear-gradient(135deg, #4A8B41 0%, #1E5631 100%);
                border-radius: 10px;
                padding: 1rem;
                color: white;
            }
            p {
                color: rgba(255, 255, 255, 0.8) !important;
            }
        """
    ):
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )

def create_section_header(title, description=None):
    colored_header(
        label=title,
        description=description,
        color_name="green-70"
    )

def show_success_banner(message):
    st.success(message)

def show_info_banner(message):
    st.info(message)

def show_warning_banner(message):
    st.warning(message)

def show_error_banner(message):
    st.error(message)

def create_card_container(title, content_function, key=None):
    with stylable_container(
        key=key or f"card_{title}",
        css_styles="""
            {
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                background-color: white;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            :hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            }
        """
    ):
        st.markdown(f"### {title}")
        content_function()

# Navigation
with st.container():
    selected_tab = option_menu(
        menu_title=None,
        options=[
            "Home", 
            "Upload & Explore", 
            "Data Preprocessing", 
            "Model Training", 
            "Gap-Filling", 
            "Gap-Fill Evaluation", 
            "Advanced Flux Visualization", 
            "About"
        ],
        icons=[
            "house", 
            "cloud-upload", 
            "gear", 
            "cpu", 
            "puzzle", 
            "graph-up", 
            "bar-chart", 
            "info-circle"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0px", "background-color": "#f8f9fa", "border-radius": "10px"},
            "icon": {"color": "#4A8B41", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px", 
                "text-align": "center", 
                "margin": "0px", 
                "padding": "10px", 
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": "#1E5631", "color": "white"},
        }
    )
    
    st.session_state.active_tab = selected_tab

# Home
if st.session_state.active_tab == "Home":
    # Header with logo and title
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Load and display logo
        logo_path = "static/logo.png"  # Adjust to your path
        with open(logo_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <div style='text-align: center; padding: 20px;'>
                <img src="data:image/png;base64,{encoded}" width="500"/>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <h1 style='color: #1E5631;'>Universal Time Series & Eddy Covariance Gap-Filling System</h1>
            <h4 style='color: #4A8B41; font-weight: normal;'>
            An interactive platform for processing, filling, and evaluating gaps in any time series - including flux tower datasets, using Machine Learning models.
            </h4>
            """, 
            unsafe_allow_html=True
        )
   
   # Horizontal rule
    st.markdown("<div class='section-header'><h2>🌍 Supported Datasets</h2></div>", unsafe_allow_html=True)
       # Custom CSS for consistent styling and improved padding
    st.markdown(
        """
        <style>
        .dataset-column {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 15px;
            margin: 5px; /* Added margin for spacing between columns */
            height: 100%; /* Ensures columns have consistent height */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .dataset-column img {
            max-width: 100%;
            height: auto;
            margin-bottom: 10px;
        }
        .dataset-column h4 {
            color: #1E5631; /* Changed to match app title color */
            margin-top: 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #e0e0e0; /* Subtle separator */
        }
        .dataset-column ul {
            list-style-type: none;
            padding-left: 0;
            margin-top: 10px;
        }
        .dataset-column ul li {
            margin-bottom: 8px;
            color: #4A8B41;
            display: flex;
            align-items: center;
        }
        .dataset-column ul li::before {
            content: '✅'; /* Checkmark icon */
            margin-right: 8px;
            color: #1E5631; /* Green checkmark */
            font-size: 1.1em;
        }
        .dataset-column p {
            font-size: 0.9em;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = st.columns(4)

    with dataset_col1:
        st.markdown(
            """
            <div class='dataset-column'>
                <h4>Diverse Environmental & Energy Data</h4>
                <ul>
                    <li>Meteorological station data</li>
                    <li>Climate reanalysis time series</li>
                    <li>Remote sensing time series</li>
                    <li>Hydrological series (e.g., river discharge, rainfall)</li>
                    <li>Energy generation series (e.g., solar, wind)</li>
                    <li>Any environmental or sensor data with gaps</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with dataset_col2:
        st.markdown(
            """
            <div class='dataset-column' style='text-align: center;'>
                <a href="https://fluxnet.org/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/a/ac/Fluxnet_Logo.jpg" width="100"/></a>
                <p><a href="https://fluxnet.org/" target="_blank"><b>FLUXNET</b></a><br/>
                Global network of micrometeorological tower sites measuring ecosystem fluxes.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with dataset_col3:
        st.markdown(
            """
            <div class='dataset-column' style='text-align: center;'>
                <a href="https://ameriflux.lbl.gov/" target="_blank"><img src="https://ameriflux.lbl.gov/wp-content/uploads/2014/06/Logo-AmerifluxNet-Horiz1.png" width="180"/></a>
                <p><a href="https://ameriflux.lbl.gov/" target="_blank"><b>AmeriFlux</b></a><br/>
                North and South American flux data on ecosystem–atmosphere exchanges.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with dataset_col4:
        st.markdown(
            """
            <div class='dataset-column' style='text-align: center;'>
                <a href="https://icos-ri.eu/" target="_blank"><img src="https://www.icos-cp.eu/media/253" width="180"/></a>
                <p><a href="https://icos-ri.eu/" target="_blank"><b>ICOS</b></a><br/>
                Integrated Carbon Observation System: harmonized GHG flux data across Europe.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Main features section
    st.markdown("<div class='section-header'><h2>Key Features</h2></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Add this CSS to prevent flickering
    st.markdown("""
    <style>
        .feature-card {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: white;
            height: 100%;
            transition: box-shadow 0.3s ease !important;
        }
        .feature-card:hover {
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1) !important;
        }
        /* Prevent content disappearing */
        html, body {
            -webkit-user-select: none;
            user-select: none;
        }
        .stApp {
            overflow-x: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Gap Detection & Filling</h3>
            <p>Pinpoints missing observations and reconstructs them with ML-driven estimates—preserving the rhythm and seasonality of your time-series..</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Advanced Analytics</h3>
            <p>Explore patterns, trends, and anomalies in flux and environmental data using interactive plots and robust statistical summaries.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Machine Learning Models</h3>
            <p>Leverage state-of-the-art models like Random Forest and XGBoost to predict missing values, with detailed performance metrics and visual diagnostics.</p>
        </div>
        """, unsafe_allow_html=True)
        
   # Workflow section
    st.markdown("<div class='section-header'><h2>Workflow</h2></div>", unsafe_allow_html=True)
    
    # Create a timeline-like workflow visualization
    workflow_steps = [
        {"icon": "cloud-upload", "title": "Upload Data", "description": "Import your eddy covariance data"},
        {"icon": "search", "title": "Explore", "description": "Analyze patterns and identify gaps"},
        {"icon": "gear", "title": "Preprocess", "description": "Clean data and engineer features"},
        {"icon": "cpu", "title": "Train Models", "description": "Build and optimize ML models"},
        {"icon": "puzzle", "title": "Fill Gaps", "description": "Apply models to fill missing data"},
        {"icon": "graph-up", "title": "Evaluate", "description": "Validate results with metrics"}
    ]
    
    # Display workflow as a horizontal timeline
    cols = st.columns(len(workflow_steps))
    for i, (col, step) in enumerate(zip(cols, workflow_steps)):
        with col:
            with stylable_container(
                key=f"workflow_step_{i}",
                css_styles="""
                    {
                        border-radius: 10px;
                        padding: 1rem;
                        text-align: center;
                        background-color: white;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                        height: 100%;
                        position: relative;
                    }
                    :hover {
                        background-color: #f8f9fa;
                    }
                """
            ):
                st.markdown(f"<i class='bi bi-{step['icon']}' style='font-size: 2rem; color: #1E5631;'></i>", unsafe_allow_html=True)
                st.markdown(f"**{step['title']}**")
                st.markdown(f"<small>{step['description']}</small>", unsafe_allow_html=True)
                
                # Add connecting line except for the last item
                if i < len(workflow_steps) - 1:
                    st.markdown("""
                        <div style='position: absolute; top: 50%; right: -15px; width: 30px; height: 2px; background-color: #ddd; z-index: 1;'></div>
                    """, unsafe_allow_html=True)
    # Author info section
    st.markdown("<div class='section-header'><h2>Developer and Contributor</h2></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with stylable_container(
            key="author_card_1",
            css_styles="""
                {
                    border-radius: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                    padding: 2.0rem;
                    margin-bottom: 1.5rem;
                    background-color: white;
                }
            """
        ):
            # Load author photo
            with open("static/max_photo.jpg", "rb") as image_file:
                photo_encoded = base64.b64encode(image_file.read()).decode()
            
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; gap: 20px;'>
                    <img src="data:image/png;base64,{photo_encoded}" width="100" style="border-radius: 10px;"/>
                    <div>
                        <h3 style='margin-top: 0;'>Max Anjos</h3>
                        <p>Researcher in urban climate, environmental modeling, and geospatial analysis.</p>
                        <div style='display: flex; gap: 10px;'>
                            <a href="https://github.com/maxanjos" target="_blank" style='color: #1E5631;'><i class='bi bi-github'></i> GitHub</a>
                            <a href="https://www.linkedin.com/in/maxanjos/" target="_blank" style='color: #1E5631;'><i class='bi bi-linkedin'></i> LinkedIn</a>
                            <a href="mailto:maxanjos@campus.ul.pt" style='color: #1E5631;'><i class='bi bi-envelope'></i> Email</a>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        with stylable_container(
            key="author_card_2",
            css_styles="""
                {
                    border-radius: 10px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                    padding: 2.0rem;
                    margin-bottom: 1.5rem;
                    background-color: white;
                }
            """
        ):
            # Load contributor photo
            with open("static/fred_photo.jpg", "rb") as image_file:
                photo_encoded = base64.b64encode(image_file.read()).decode()
            
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; gap: 20px;'>
                    <img src="data:image/png;base64,{photo_encoded}" width="120" style="border-radius: 10px;"/>
                    <div>
                        <h3 style='margin-top: 0;'>Fred Meier</h3>
                        <p>Senior Scientist Urban Climatology and Urban Climate Observatory (UCO) Berlin.</p>
                        <div style='display: flex; gap: 10px;'>
                            <a href="https://www.tu.berlin/en/klima/about-us/meier-fred" target="_blank" style='color: #1E5631;'><i class='bi bi-building'></i> TU-Berlin</a>
                            <a href="mailto:fred.meier@tu-berlin.de" style='color: #1E5631;'><i class='bi bi-envelope'></i> Email</a>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    
    def show_donation_message():
       st.markdown(
        """
        <div style="
            background: #FFFFFF;
            color: #333333;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
        ">
            <h2 style="color:#1E5631; margin-bottom: 0.5rem; font-weight:700;">
                💚 Support This Platform
            </h2>
            <p style="font-size: 1.1rem; margin-bottom: 1.2rem;">
                Your contribution helps us maintain and improve this Platform 🌍
            </p>
            <form action="https://www.paypal.com/donate" method="post" target="_top">
                <input type="hidden" name="hosted_button_id" value="RHT74PFXMT3V6" />
                <button type="submit" 
                    style="
                        background-color: #1E5631;
                        color: white;
                        padding: 0.75rem 2rem;
                        border: none;
                        border-radius: 8px;
                        font-size: 1rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    "
                    onmouseover="this.style.backgroundColor='#154823';" 
                    onmouseout="this.style.backgroundColor='#1E5631';">
                    ☕ Donate with PayPal
                </button>
            </form>
        </div>
        """,
        unsafe_allow_html=True
    )
    show_donation_message()
    # Footer
    st.markdown(
        """
        <div class='footer'>
            <p>© 2025 Max Anjos • Eddy Covariance Gap-Filling System | Version 1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# 1. Upload & Explore Page
if st.session_state.active_tab == "Upload & Explore":
    
    # Page header
    colored_header(
        label="Data Upload & Exploration",
        description="Import and understand your data",
        color_name="green-70"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Data")
        #uploaded_file = st.file_uploader("Choose a CSV or txt file", type=["csv", "txt"])
        #use_example = st.checkbox("Use example dataset", value=False)
        upload_method = st.radio("Choose data upload format:", ["Upload CSV", "Upload ZIP", "Use example dataset"])
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("📄 Upload CSV or TXT", type=["csv", "txt"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    df, source = detect_and_preprocess_dataset(df)
                    st.session_state.data = df
                    st.success(f"✅ File loaded. Format detected: **{source}**")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        elif upload_method == "Upload ZIP":
            df, source = upload_zip_and_extract_csv()
            if df is not None:
                st.session_state.data = df
        elif upload_method == "Use example dataset":
            df = load_example_data()
            if df is not None:
                st.session_state.data = df
                st.success("✅ Example dataset loaded successfully!")

    with col2:
        if st.session_state.data is not None:
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: {st.session_state.data.shape[0]}")
            st.write(f"Number of columns: {st.session_state.data.shape[1]}")
            st.write("First 5 rows:")
            st.dataframe(st.session_state.data.head())

    # Data Analysis Section
    if st.session_state.data is not None:
        st.header("Data Analysis")

        # Missing values analysis
        st.subheader("Top Missing Values Overview")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Missing values summary
            missing_values = st.session_state.data.isna().sum().sort_values(ascending=False)
            missing_values_pct = (missing_values / len(st.session_state.data) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Values': missing_values,
                'Percentage (%)': missing_values_pct
            })
            #st.dataframe(missing_df.head(10))
              # Apply conditional formatting
            styled_df = missing_df.style\
                .background_gradient(subset='Missing Values', cmap='Reds')\
                .format({'Percentage (%)': "{:.2f}%"})

            st.dataframe(styled_df, use_container_width=True)

        with col2:
            # Visualize missing values
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(st.session_state.data.isna().transpose(),
                        cmap='viridis',
                        cbar_kws={'label': 'Missing'},
                         )
            plt.title('🔬 Heatmap of Missing Data (by Column and Record)', fontsize=14)
            plt.xlabel("Date")
            plt.ylabel("Variables")
            plt.tight_layout()
            st.pyplot(fig)
            

        # Column selection for exploration
        st.subheader("Variable Exploration")

        flux_vars = ["date", "time", "datetime", "co2_flux_qc"]
        actual_flux_cols = [col for col in st.session_state.data.columns if col not in flux_vars]

        selected_variable = st.selectbox(
            "Select Variable to Analyze:",
            actual_flux_cols,
            index=0 if actual_flux_cols else None
        )

        if selected_variable:
            col1, col2 = st.columns([1, 1])

            with col1:
                # Time series plot
                fig = px.line(
                    st.session_state.data.reset_index(),
                    x='datetime' if 'datetime' in st.session_state.data.reset_index().columns else st.session_state.data.reset_index().index,
                    y=selected_variable,
                    title=f'Time Series: {selected_variable}'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribution plot
                fig = px.histogram(
                    st.session_state.data,
                    x=selected_variable,
                    title=f'Distribution: {selected_variable}',
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Daily/monthly patterns
            if isinstance(st.session_state.data.index, pd.DatetimeIndex):
                st.subheader("Temporal Patterns")

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Diurnal pattern
                    hourly_data = st.session_state.data[selected_variable].groupby(st.session_state.data.index.hour).mean()
                    fig = px.line(
                        x=hourly_data.index,
                        y=hourly_data.values,
                        title=f'Diurnal Pattern: {selected_variable}',
                        labels={'x': 'Hour of Day', 'y': selected_variable}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Monthly pattern
                    if len(st.session_state.data.index) > 28:
                        st.session_state.data['month'] = st.session_state.data.index.month
                        fig = px.box(
                            st.session_state.data,
                            x='month',
                            y=selected_variable,
                            points="all",
                            title=f'Monthly Distribution: {selected_variable}',
                            labels={'month': 'Month', selected_variable: selected_variable}
                        )
                        st.plotly_chart(fig, use_container_width=True)

# 2. Data Preprocessing Page
elif st.session_state.active_tab == "Data Preprocessing":
    # Page header
    colored_header(
        label="Data Preprocessing",
        description="Prepare your data",
        color_name="green-70"
    )

    if st.session_state.data is None:
        st.warning("Please upload data or load the example dataset first.")
    else:
        st.subheader("Quality Assurance & Control")

        col1, col2 = st.columns([1, 1])

        with col1:
            # QA/QC parameters
            st.write("Set QA/QC Parameters:")
            remove_outliers = st.checkbox("Remove outliers", value=False)

            if remove_outliers:
                # Select only numerical columns for outlier detection
                numerical_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["IQR", "Z-score", "Modified Z-score"],
                    index=0,
                    help = "Proceed only applied to numerical variables"
                )
                if outlier_method == "IQR":
                    iqr_factor = st.slider("IQR factor", 1.5, 3.0, 1.5, 0.1)
                elif outlier_method == "Z-score":
                    z_threshold = st.slider("Z-score threshold", 2.0, 8.0, 3.0, 0.1)
                elif outlier_method == "Modified Z-score":
                    mod_z_threshold = st.slider("Modified Z-score threshold", 3.0, 10.0, 3.5, 0.1)
                    
            apply_qc_flags = st.checkbox("Apply quality-control filter", value=False)

            if apply_qc_flags:

                # 1️⃣  Pick one or more columns that *contain* QC / flag values
                qc_columns = st.multiselect(
                    "Select QC / flag column(s):",
                    options=st.session_state.data.columns.tolist()
                )
                if qc_columns:
                    # 3️⃣  For each chosen QC column, let user set a threshold
                    qc_thresholds = {}
                    for qc_col in qc_columns:
                        qc_min = int(st.session_state.data[qc_col].min(skipna=True))
                        qc_max = int(st.session_state.data[qc_col].max(skipna=True))
                        qc_thresholds[qc_col] = st.slider(
                            f"Maximum acceptable value for **{qc_col}**:",
                            min_value=qc_min,
                            max_value=qc_max,
                            value=min(qc_max, 1),
                            step=1,
                            help = "0 = Highest quality, 1 = Medium quality, 2 = Low quality (or custom scale).Please, ensure the correct QC value. Check metadata out!"
                        )   
                else:
                    st.info("Select at least one QC column to proceed.")
       
        #For Outliers
        flux_vars = ["date", "time", "datetime", "co2_flux_qc", "month"]
        actual_flux_cols_out = [col for col in st.session_state.data.columns if col not in flux_vars]
        
        #For flag
        flux_vars = ["co2_flux", "latent_heat_flux", "sensible_heat_flux"]
        actual_flux_cols_flag = [col for col in st.session_state.data.columns if any(flux_var in col for flux_var in flux_vars)]
        
        # Buttons for preprocessing actions
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("⚙️ Preprocess Data", key="preprocess_button"):
                try:
                    with st.spinner("Preprocessing data..."):
                        # Apply preprocessing based on selected options
                        preprocessed_data = st.session_state.data.copy()
                        
                        # Apply QC flags if selected
                        if apply_qc_flags and qc_columns:
                            for flux_col in actual_flux_cols_flag:
                                qc_col = f"qc_{flux_col}"
                                if qc_col in preprocessed_data.columns:
                                    # Mark as NaN where QC flag exceeds threshold
                                    mask = preprocessed_data[qc_col] > qc_threshold
                                    preprocessed_data.loc[mask, flux_col] = np.nan
                        
                        # Handle outliers if selected
                        if remove_outliers:
                            for flux_col in actual_flux_cols_out:
                                if outlier_method == "IQR":
                                    Q1 = preprocessed_data[flux_col].quantile(0.25)
                                    Q3 = preprocessed_data[flux_col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    mask = (preprocessed_data[flux_col] < (Q1 - iqr_factor * IQR)) | (
                                            preprocessed_data[flux_col] > (Q3 + iqr_factor * IQR))
                                    preprocessed_data.loc[mask, flux_col] = np.nan
                                elif outlier_method == "Z-score":
                                    mean = preprocessed_data[flux_col].mean()
                                    std = preprocessed_data[flux_col].std()
                                    mask = abs((preprocessed_data[flux_col] - mean) / std) > z_threshold
                                    preprocessed_data.loc[mask, flux_col] = np.nan
                                elif outlier_method == "Modified Z-score":
                                    median = preprocessed_data[flux_col].median()
                                    mad = np.median(abs(preprocessed_data[flux_col] - median)) * 1.4826
                                    mask = abs((preprocessed_data[flux_col] - median) / mad) > mod_z_threshold
                                    preprocessed_data.loc[mask, flux_col] = np.nan
                         
                        # Update app state
                        st.session_state.data = preprocessed_data
                        st.success("Data preprocessing completed!")

                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")

        with col2:
            if st.button("Reset to Original Data", key="reset_button"):
                try:
                    # Reload original data or example data
                    if 'original_data' in st.session_state:
                        st.session_state.data = st.session_state.original_data.copy()
                    else:
                        st.session_state.data = load_example_data()
                    st.success("Data reset to original!")
                except Exception as e:
                    st.error(f"Error resetting data: {str(e)}")

        # Show preprocessing results
        if st.session_state.data is not None:
            st.subheader("Gap Analysis")

            # Identify and analyze gaps
            st.write("Gap Distribution:")

            # Calculate gap statistics
            missing_counts = st.session_state.data.isna().sum()
            missing_pct = (missing_counts / len(st.session_state.data)) * 100

            # Create tabs for gap analysis views
            gap_tabs = st.tabs([ "Gap Length Distribution"])

            with gap_tabs[0]:
                # Calculate gap lengths
                if actual_flux_cols_out:
                    selected_var = st.selectbox("Select variable for gap length analysis:", actual_flux_cols_out)

                    # Calculate gap lengths
                    is_na = st.session_state.data[selected_var].isna()
                    gap_starts = []
                    gap_lengths = []
                    in_gap = False
                    gap_start = None
                    gap_len = 0

                    for i, val in enumerate(is_na):
                        if val and not in_gap:
                            in_gap = True
                            gap_start = i
                            gap_len = 1
                        elif val and in_gap:
                            gap_len += 1
                        elif not val and in_gap:
                            gap_starts.append(gap_start)
                            gap_lengths.append(gap_len)
                            in_gap = False

                    if in_gap:
                        gap_starts.append(gap_start)
                        gap_lengths.append(gap_len)

                    # Create a plot of gap length distribution
                    if gap_lengths:
                        gap_df = pd.DataFrame({'start_idx': gap_starts, 'length': gap_lengths})
                        fig = px.histogram(
                            gap_df,
                            x='length',
                            title=f"Gap Length Distribution for {selected_var}",
                            labels={'length': 'Gap Length (timesteps)', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary statistics
                        st.write("Gap Length Statistics:")
                        st.write(f"Total gaps: {len(gap_lengths)}")
                        st.write(f"Mean gap length: {np.mean(gap_lengths):.2f} timesteps")
                        st.write(f"Median gap length: {np.median(gap_lengths):.2f} timesteps")
                        st.write(f"Max gap length: {np.max(gap_lengths)} timesteps")

# 3. Model Training Page
elif st.session_state.active_tab == "Model Training":
    # Page header
    colored_header(
        label="Model Training",
        description="Train your machine learning models for gap filling",
        color_name="green-70"
    )

    if st.session_state.data is None:
        st.warning("Please upload data or load the example dataset first.")
    else:
        st.subheader("Select Target Variable")
        
        # Define potential targets
        exclude_cols = ['date', 'time', 'datetime']
        actual_flux_cols = [col for col in st.session_state.data.columns if col not in exclude_cols]
        target_vars = st.multiselect(
            "🎯 Select the target variable to predict:",
            actual_flux_cols,
            default=actual_flux_cols[:min(0, len(actual_flux_cols))],
            help="Please select one variable you want to predict."
        )

        # Select features automatically (exclude targets, QC, datetime-related)
        if target_vars:
            exclude_patterns = ['date', 'time', '_qc', 'flag', 'datetime', 'month'] + target_vars
            potential_features = [col for col in st.session_state.data.columns if not any(p in col for p in exclude_patterns)]
            
            # Final feature list (excluding targets)
            selected_features = [col for col in potential_features if col not in target_vars]

            # Time-based features (used later if enabled)
            time_based_features = [
                'hour', 'day', 'month', 'year', 'dayofyear', 'weekday', 'is_weekend',
                'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 'month_sin',
                'month_cos', 'hour_decimal', 'season', 'season_0', 'season_1',
                'season_2', 'season_3', 'is_morning', 'is_afternoon', 'is_evening',
                'is_night'
            ]
            # Informative display
            st.markdown(f"🔎 **Features automatically selected:** `{len(selected_features)}` variables used for modeling.")
            with st.expander("📋 See selected features"):
                st.write(selected_features)
        
       # Model selection and configuration
        st.subheader("Model Configuration")

        col1, col2 = st.columns([1, 1])

        with col1:
            model_type = st.selectbox(
                "Choose a machine learning algorithm:",
                ["Random Forest", "XGBoost"],
                index=0,
                help="Select the model type to train on your data."
            )

        with col2:
            st.markdown("🔧 **Set Model Hyperparameters**")

            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees (n_estimators):", 50, 500, 100, step=10)
                max_depth = st.slider("Maximum Tree Depth (max_depth):", 5, 10, value=10)
                min_samples_split = st.slider("Minimum Samples to Split (min_samples_split):", 2, 10, 4)
                min_samples_leaf = st.slider("Minimum Samples per Leaf (min_samples_leaf):", 1, 10, 2)

                model_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'random_state': 42,
                    'max_features': 'sqrt',  # Good balance for regression
                    'bootstrap': True,
                    'n_jobs': -1,
                }

            elif model_type == "XGBoost":
                n_estimators = st.slider("Number of Trees (n_estimators):", 50, 500, 100, step=10)
                learning_rate = st.select_slider("Learning Rate (learning_rate):", options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3],value=0.01)
                max_depth = st.slider("Maximum Tree Depth (max_depth):", 3, 10, 5)

                model_params = {
                    'objective': 'reg:squarederror',
                    'booster': 'gbtree',
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'min_child_weight': 3,  # Minimum sum of instance weight needed in a child
                    'subsample': 0.7,  # Prevent overfitting by training on subsample
                    'colsample_bytree': 0.7,  # Use a subset of features per tree
                    'gamma': 0.1,  # Minimum loss reduction to make a split
                    'reg_alpha': 0.1,  # L1 regularization (sparse models)
                    'reg_lambda': 1.0,  # L2 regularization
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0,
                    'missing': np.nan
                }

            else:
                model_params = {}
     # Train model button
        if st.button("Train Model"):
            try:
                with st.spinner("🔄 Preparing data and training models..."):
                    processed_data = st.session_state.data.copy()
                    # Ensure 'datetime' is a column
                    if isinstance(processed_data.index, pd.DatetimeIndex):
                        processed_data['datetime'] = processed_data.index
                    elif 'date' in processed_data.columns and 'time (UTC)' in processed_data.columns:
                        processed_data['datetime'] = pd.to_datetime(processed_data['date'] + ' ' + processed_data['time (UTC)'], format='%d.%m.%Y %H:%M:%S')
                    else:
                        st.error("❌ Could not infer or create 'datetime' column. Please ensure your data includes datetime information.")
                        st.stop()
                    added_features = []

                    # Time features
                    processed_data = create_time_features(processed_data)
                    added_features += [col for col in processed_data.columns if col not in st.session_state.data.columns]

                    # Lag features
                    processed_data = create_lag_features(processed_data, target_vars)
                    added_features += [col for col in processed_data.columns if "_lag_" in col]

                    # Rolling features
                    processed_data = create_rolling_features(processed_data, target_vars, windows=[3, 6, 12], stats=["mean", "std"])
                    added_features += [col for col in processed_data.columns if "_rolling_" in col]

                    # VPD
                    if 'air_temperature' in processed_data.columns and 'relative_humidity' in processed_data.columns:
                        processed_data = calculate_vpd(processed_data)
                        added_features.append("vpd")

                    # Meteorological features
                    processed_data = create_met_features(processed_data)
                    added_features += [col for col in processed_data.columns if col not in st.session_state.data.columns]

                    # Categorical encoding
                    # Handle categorical encoding only if the columns were created
                    possible_categorical_cols = ['wind_dir_cat', 'stability_class']
                    existing_categorical_cols = [col for col in possible_categorical_cols if col in processed_data.columns]

                    if existing_categorical_cols:
                        processed_data = encode_categorical_features(processed_data, existing_categorical_cols)
                        added_features += existing_categorical_cols

                    # Extend selected_features with new ones (and ensure uniqueness)
                    selected_features = list(set(selected_features + added_features))
                    selected_features = [col for col in selected_features if pd.api.types.is_numeric_dtype(processed_data[col])]
                    time_based_features = [col for col in time_based_features if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col])]
                    
                    # Progress bar
                    progress = st.progress(0)
                    total = len(target_vars)

                    for idx, target_col in enumerate(target_vars):
                        st.info(f"📈 Training models for target: `{target_col}`")

                        model_all, model_time = train_model(
                            data=processed_data,
                            target_col=target_col,
                            selected_features=selected_features,
                            time_based_features=[f for f in time_based_features if f in processed_data.columns],
                            model_type=model_type,
                            base_params=model_params
                        )

                        # Save models
                        model_key_all = f"{model_type}_ALL_{target_col}"
                        model_key_time = f"{model_type}_TIME_{target_col}"

                        st.session_state.models[model_key_all] = {
                            'model': model_all,
                            'features': selected_features,
                            'feature_set': 'all'
                        }

                        st.session_state.models[model_key_time] = {
                            'model': model_time,
                            'features': [f for f in time_based_features if f in processed_data.columns],
                            'feature_set': 'time_based'
                        }

                        st.success(f"✅ Models for `{target_col}` trained and stored!")
                        progress.progress((idx + 1) / total)

                    st.success("🎉 All models trained successfully!")

            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")

# 4. Gap-Filling Page
elif st.session_state.active_tab == "Gap-Filling":
    # Page header
    colored_header(
        label="Gap-Filling with Trained Models",
        description="Gap-filling your eddy covariance data",
        color_name="green-70"
    )

    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
    elif not st.session_state.models:
        st.warning("Please train at least one model before applying gap-filling.")
    else:
        st.subheader("🔧 Select Target Variable for Gap-Filling")

        # Get target variable names (e.g. 'co2_flux') from model keys
        all_target_vars = list({k.split('_', 2)[-1] for k in st.session_state.models.keys()})
        selected_target = st.selectbox("Choose target variable:", all_target_vars)

        # Retrieve model keys for the selected target
        model_all_key = [k for k in st.session_state.models.keys() if k.endswith(selected_target) and "_ALL_" in k]
        model_time_key = [k for k in st.session_state.models.keys() if k.endswith(selected_target) and "_TIME_" in k]

        if not model_all_key or not model_time_key:
            st.warning(f"No trained models found for target variable `{selected_target}`. Please train models first.")
            st.stop()

        model_all = st.session_state.models[model_all_key[0]]['model']
        model_time = st.session_state.models[model_time_key[0]]['model']
        all_features = st.session_state.models[model_all_key[0]]['features']
        time_features = st.session_state.models[model_time_key[0]]['features']

        if st.button("🚀 Start Gap-Filling"):
            try:
                with st.spinner("Applying trained models to fill gaps..."):
                    df = st.session_state.data.copy()
                    df_gapfilled = df.copy()
                    df_gapfilled['filled'] = 0

                    # Ensure datetime column exists
                    if 'datetime' not in df_gapfilled.columns:
                        if 'date' in df_gapfilled.columns and 'time (UTC)' in df_gapfilled.columns:
                            df_gapfilled['datetime'] = pd.to_datetime(
                                df_gapfilled['date'] + ' ' + df_gapfilled['time (UTC)'],
                                format='%d.%m.%Y %H:%M:%S'
                            )
                        elif isinstance(df_gapfilled.index, pd.DatetimeIndex):
                            df_gapfilled['datetime'] = df_gapfilled.index

                    # === Apply same feature engineering used during training ===
                    df_gapfilled = create_time_features(df_gapfilled)
                    df_gapfilled = create_lag_features(df_gapfilled, [selected_target], lag_periods=[1, 3, 4, 5, 6, 24, 168])
                    df_gapfilled = create_rolling_features(df_gapfilled, [selected_target], windows=[3, 6, 12], stats=["mean", "std"])

                    if 'air_temperature' in df_gapfilled.columns and 'relative_humidity' in df_gapfilled.columns:
                        df_gapfilled = calculate_vpd(df_gapfilled)

                    df_gapfilled = create_met_features(df_gapfilled)

                    cat_cols = ['wind_dir_cat', 'stability_class']
                    cat_cols = [col for col in cat_cols if col in df_gapfilled.columns]
                    df_gapfilled = encode_categorical_features(df_gapfilled, cat_cols)

                    # === Gap-filling ===
                    missing_mask = df_gapfilled[selected_target].isnull()
                    X_missing_time_based = df_gapfilled.loc[missing_mask, time_features] 
                    X_missing_all_features = df_gapfilled.loc[missing_mask, all_features]
                    
                    # Predict using both models
                    predicted_value_all_features = model_all.predict(X_missing_all_features)
                    predicted_value_time_based = model_time.predict(X_missing_time_based)
                    
                    # Create a mask for rows where all features are present
                    all_features_present_mask = df_gapfilled.loc[missing_mask, all_features].notna().all(axis=1)
                    
                    # Assign predicted values based on feature availability
                    df_gapfilled.loc[missing_mask, selected_target] = np.where(all_features_present_mask, 
                                                                  predicted_value_all_features, 
                                                                  predicted_value_time_based)
                    df_gapfilled.loc[missing_mask, 'filled'] = 1  # Mark filled rows as 1
                    st.session_state.filled_data = df_gapfilled
                    st.success("✅ Gap-filling completed!")
                    
            except Exception as e:
                st.error(f"❌ Error during gap-filling: {e}")

        # Show preview and visualization
        if st.session_state.filled_data is not None:
            df_full = st.session_state.filled_data
            total_filled_in_data = df_full['filled'].sum()
            total_rows = len(df_full)
            percent_filled = (total_filled_in_data / total_rows) * 100
            
            st.subheader("📊 Gap-Filling Summary")
            st.metric(label = "Total gaps filled", value = f"{int(total_filled_in_data)}")
            st.metric(label="Percentage Filled", value=f"{percent_filled:.1f}%")

            #st.write(st.session_state.filled_data.head())
            st.subheader("🔍 Filled Data Preview (First 50 rows)")
            st.caption("Rows highlighted in **yellow** were estimated by the model ('filled' = 1)")
            df_preview = st.session_state.filled_data.head(50)
            def highlight_filled(row):
                style = ['background-color: yellow' if row['filled'] == 1 else '' for _ in row.index]
                return style
            st.dataframe(
                df_preview.style.apply(highlight_filled, axis=1),
                use_container_width=True,
                hide_index=False
            )

            st.subheader("📈 Visualize Original vs Filled")
            plot_col = st.selectbox("Select a variable to visualize:", [selected_target] + st.session_state.data.columns.tolist())

            df_original = st.session_state.data.copy()
            df_filled = st.session_state.filled_data.copy()
            
            # Identify gap-filled points
            filled_only = df_filled.copy()
            filled_only[plot_col] = df_filled[plot_col].where(df_filled['filled'] == 1, np.nan)
            
            fig = go.Figure()
            # Original (with NaNs)
            fig.add_trace(go.Scatter(
                x=df_original.index,
                y=df_original[plot_col],
                mode='lines',
                name='Original (with gaps)',
                line=dict(color='blue', width=2),
                connectgaps=False
            ))

            # Filled values
            fig.add_trace(go.Scatter(
                x=filled_only.index,
                y=filled_only[plot_col],
                mode='markers+lines',
                name='Gap-filled',
                line=dict(color='red', dash='dot'),
                marker=dict(size=6, color='red', symbol='circle')
            ))

            fig.update_layout(
                title=f"Gap-Filling Visualization for: {plot_col}",
                xaxis_title="Time",
                yaxis_title=plot_col,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Download option
            st.subheader("💾 Download Gap-Filled Data")
            df_export= st.session_state.filled_data.copy()
            cols_to_keep=["datetime", selected_target, 'filled']
            final_cols = [c for c in cols_to_keep if c in df_export.columns]
            df_export = df_export[final_cols]
            st.caption("The downloaded file will contain only: **datetime**, **target variable**, and **filled** status (0=original, 1=gap-filled).")
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"gapfilled_{selected_target}_clean.csv",
                    mime="text/csv",
                    type="primary" 
                )
# 5 Gap-Filling Evaluation
elif st.session_state.active_tab == "Gap-Fill Evaluation":
     # Page header
    colored_header(
        label="Gap-Filling Evaluation",
        description="Evaluate your gap filled eddy covariance data",
        color_name="green-70"
    )

    # User selects NaN injection percentage
    nan_percentage = st.slider(
        "Select percentage of values to remove for testing gap-filling:",
        min_value=1,
        max_value=50,
        value=20
    ) / 100.0
    # ... (código anterior do slider nan_percentage) ...

    # --- Configuração Avançada de Mecanismo de Erro ---
    col_mech1, col_mech2 = st.columns(2)
    
    with col_mech1:
        missing_mechanism = st.selectbox(
            "Missing Data Mechanism:",
            options=["MCAR", "MAR", "MNAR"],
            help="""
            **MCAR (Random):** Gaps occur randomly (e.g., power failure).
            **MAR (Dependent):** Gaps depend on environmental conditions (e.g., rain causes sensor error).
            **MNAR (Not Random):** Gaps depend on the value itself (e.g., extreme fluxes saturate sensor).
            """
        )

    dependency_feature = None
    if missing_mechanism == "MAR":
        with col_mech2:
            # Pega colunas numéricas disponíveis, exceto a target
            available_cols = [c for c in st.session_state.data.columns if st.session_state.data[c].dtype in ['float64', 'float32']]
            dependency_feature = st.selectbox(
                "Select variable causing missingness:",
                options=available_cols,
                index=0 if 'precipitation' not in available_cols else available_cols.index('precipitation'),
                help="Higher values of this variable will increase the probability of gaps in the target."
            )

    # ... (Resto do código de validação) ...
    # Validation checks
    if st.session_state.data is None:
        st.warning("Please upload and preprocess data first.")
    elif not st.session_state.models:
        st.warning("Please train at least one model before applying gap-filling.")
    else:        
        # Get target variable names (e.g. 'co2_flux') from model keys
        all_target_vars = list({k.split('_', 2)[-1] for k in st.session_state.models.keys()})
        selected_target = st.selectbox("**The selected target variable**:", all_target_vars)
         # Retrieve model keys for the selected target
        model_all_key = [k for k in st.session_state.models.keys() if k.endswith(selected_target) and "_ALL_" in k]
        model_time_key = [k for k in st.session_state.models.keys() if k.endswith(selected_target) and "_TIME_" in k]

        if not model_all_key or not model_time_key:
            st.warning(f"No trained models found for target variable `{selected_target}`. Please train models first.")
            st.stop()

        model_all = st.session_state.models[model_all_key[0]]['model']
        model_time = st.session_state.models[model_time_key[0]]['model']
        all_features = st.session_state.models[model_all_key[0]]['features']
        time_features = st.session_state.models[model_time_key[0]]['features']
        
        if st.button("🚀 Run Evaluation Test"):
            try:
                with st.spinner("Generating artifical gaps and applying trained models to fill them..."):
                    df = st.session_state.data.copy()
                    df_gapfilled = df.copy()
                    df_gapfilled['filled'] = 0
                    df_gapfilled.dropna(inplace=True)
                    # Ensure datetime column exists
                    if 'datetime' not in df_gapfilled.columns:
                        if 'date' in df_gapfilled.columns and 'time (UTC)' in df_gapfilled.columns:
                            df_gapfilled['datetime'] = pd.to_datetime(
                                df_gapfilled['date'] + ' ' + df_gapfilled['time (UTC)'],
                                format='%d.%m.%Y %H:%M:%S'
                            )
                        elif isinstance(df_gapfilled.index, pd.DatetimeIndex):
                            df_gapfilled['datetime'] = df_gapfilled.index

                    # === Apply same feature engineering used during training ===
                    df_gapfilled = create_time_features(df_gapfilled)
                    df_gapfilled = create_lag_features(df_gapfilled, [selected_target], lag_periods=[1, 3, 4, 5, 6, 24, 168])
                    df_gapfilled = create_rolling_features(df_gapfilled, [selected_target], windows=[3, 6, 12], stats=["mean", "std"])

                    if 'air_temperature' in df_gapfilled.columns and 'relative_humidity' in df_gapfilled.columns:
                        df_gapfilled = calculate_vpd(df_gapfilled)

                    df_gapfilled = create_met_features(df_gapfilled)

                    cat_cols = ['wind_dir_cat', 'stability_class']
                    cat_cols = [col for col in cat_cols if col in df_gapfilled.columns]
                    df_gapfilled = encode_categorical_features(df_gapfilled, cat_cols)
                    
                    #df_gapfilled_with_na = introduce_nan(df_gapfilled.copy(), [selected_target], nan_percentage, seed=42)
                    df_gapfilled_with_na = introduce_nan(
                        data=df_gapfilled.copy(), 
                        target_cols=[selected_target], 
                        nan_percentage=nan_percentage, 
                        mechanism=missing_mechanism,
                        dependency_col=dependency_feature,
                        seed=42
                        )
                    # === Gap-filling ===
                    missing_mask = df_gapfilled_with_na[selected_target].isnull()
                    original_data = df_gapfilled.loc[missing_mask, selected_target]
                    
                    X_missing_time_based = df_gapfilled_with_na.loc[missing_mask, time_features] 
                    X_missing_all_features = df_gapfilled_with_na.loc[missing_mask, all_features]
                    # Predict using both models
                    predicted_value_all_features = model_all.predict(X_missing_all_features)
                    predicted_value_time_based = model_time.predict(X_missing_time_based)
                    # Create a mask for rows where all features are present
                    all_features_present_mask = df_gapfilled_with_na.loc[missing_mask, all_features].notna().all(axis=1)
                    # Assign predicted values based on feature availability
                    df_gapfilled_with_na.loc[missing_mask, selected_target] = np.where(all_features_present_mask, 
                                                                  predicted_value_all_features, 
                                                                  predicted_value_time_based)
                    df_gapfilled_with_na.loc[missing_mask, 'filled'] = 1  # Mark filled rows as 1
                
                    # Get the gap-filled values from 'df_gapfilled_eval' for the same rows
                    filled_values = df_gapfilled_with_na.loc[missing_mask, selected_target]
                            
                    st.session_state.original_values = original_data
                    st.session_state.filled_values = filled_values
                    st.success("✅ Evaluation completed !")

            except Exception as e:
                st.error(f"❌ Error during gap-filling: {e}")
        
        # Show preview and evaluation
            # === Evaluation Metrics ===
            y_true = st.session_state.original_values
            y_pred = st.session_state.filled_values
            residuals = y_true - y_pred

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)

            st.divider()
            st.subheader("📊 Performance Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", f"{r2:.3f}", help="Closer to 1.0 is better fit")
            c2.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error - unit follows target variable")
            c3.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error - unit follows target variable")
            c4.metric("Slope (Bias)", f"{slope:.3f}", help="Should be close to 1.0 for unbiased predictions")

            # Optional residual histogram
            st.subheader("Residuals Histogram")
            fig_resid = px.histogram(
                residuals,
                nbins=30,
                title="Distribution of Residuals",
                labels={"value": "Residual"},
                marginal="rug",
                opacity=0.75
            )
            fig_resid.update_layout(bargap=0.1)
            st.plotly_chart(fig_resid, use_container_width=True)
            
            st.subheader("Predicted vs Actual Scatter Plot")
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scattergl(
                x=y_true, y=y_pred, mode='markers', 
                marker=dict(color='blue', opacity=0.5, size=5),
                name='Data Points'
                ))
            fig_scat.add_shape(type="line",
                    x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                    line=dict(color="red", width=2, dash="dash"),
                )
            fig_scat.update_layout(
                title=f"Observed vs Predicted (R²={r2:.2f})",
                xaxis_title="Observed (Ground Truth)",
                yaxis_title="Predicted (Gap-Filled)",
                height=500,
                template="plotly_white"
                )
            st.plotly_chart(fig_scat, use_container_width=True)

            # Downloadable report
            st.subheader("💾 Download Evaluation Report")
            df_report = pd.DataFrame({
                "datetime": y_true.index,
                "original": y_true.values,
                "filled": y_pred.values,
                "residual": residuals.values
            })
            df_report['mechanism'] = missing_mechanism
            df_report['nan_percentage'] = nan_percentage
            df_report['target_variable'] = selected_target
            if missing_mechanism == 'MAR':
                df_report['dependency_col'] = dependency_feature
            else:
                df_report['dependency_col'] = "N/A"
            csv_data = df_report.to_csv(index=False).encode('utf-8')
            #csv_data = df_report.to_csv(index=False)
            st.download_button(
                label="📥 Download (CSV)",
                data=csv_data,
                file_name=f"eval_report_{selected_target}_{missing_mechanism}.csv",
                mime="text/csv",
                type = "primary"
            )

# 6. Advanced Flux Visualization
elif st.session_state.active_tab == "Advanced Flux Visualization":
      # Page header
    colored_header(
        label="Advanced Flux Visualization",
        description="Visualize and explore your gap filled eddy covariance data",
        color_name="green-70"
    )

    # Validation check
    if st.session_state.filled_data is None:
        st.warning("⚠️ Please perform gap-filling before proceeding to visualization.")
    else:
        df_flux = st.session_state.filled_data.copy()
        plot_flux_partitioning(df_flux)

# About
elif st.session_state.active_tab == "About":
    # Page header
    colored_header(
        label="How it works",
        description="Understanding the logic behind the Gap-Filling Machine",
        color_name="green-70"
    )

    # Intro
    st.markdown("""
    ### 🎯 Objective
    The goal of this platform is to transform your raw, gapped times series and Eddy Covariance data into a **continuous, high-quality time series**. 
    To achieve this, we train machine earning models specifically on **your uploaded data**.
    """)
    st.divider()

    # Section 1: The Power of Time
    st.subheader("1. 🕰️ The Temporal Features (feature engineering)")
    st.markdown("""
    One of the most critical steps in our pipeline is extracting information from the timestamp. 
    The model doesn't just see a date; we break it down into potential predictors that capture natural cycles:
    
    * **Diurnal Cycle (Hour of Day):** Helps the model understand that photosynthesis peaks at noon and respiration dominates at night.
    * **Seasonality (Month/Week):** Captures phenology, such as leaf-out in spring or senescence in autumn.
    
    > **Why is this important?** Even if all your meteorological sensors fail, **time never stops**. By learning these temporal patterns, the model can make a reasonable prediction based solely on "what usually happens at this hour in this month."
    """)

    add_vertical_space(1)

    # Section 2: The Smart Fallback Strategy
    st.subheader("2. 🧠 The 'Smart Fallback' Strategy")
    st.markdown("""
    Real-world data is messy. Sometimes you have full meteorological data, sometimes you don't. 
    To handle this, we train **two parallel models** and switch between them dynamically for every single gap:
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🦄 Model A: The Precision Expert**
        
        * **Inputs:** Solar Radiation, Temperature, **AND** Time Features.
        * **When it's used:** Whenever meteorological data is available.
        * **Accuracy:** ⭐⭐⭐⭐⭐ (Highest)
        """)

    with col2:
        st.warning("""
        **⏰ Model B: The Reliable Backup**
        
        * **Inputs:** **ONLY** Time Features (Hour, Month, etc.).
        * **When it's used:** When meteorological sensors failed (NaNs).
        * **Accuracy:** ⭐⭐⭐ (Good baseline)
        """)

    st.markdown("""
    ### 🔄 The Workflow
    1.  **You upload** your data (with gaps).
    2.  The system **trains both models** on the clean parts of your data.
    3.  It scans for gaps. For each gap, it asks: *"Do I have meteo data?"*
        * **Yes?** -> Use Model A (Precision).
        * **No?** -> Use Model B (Reliability).
    4.  The result is a gap-filled dataset that maximizes accuracy without leaving any holes.
    """)

    st.divider()

    # Section 3: Technical Details (collapsed for clarity)
    with st.expander("🛠️ Under the Hood (Technical Details)"):
        st.markdown("""
        * **Algorithms:** We currently support **XGBoost** and **Random Forest**. These are ensemble tree-based methods excellent at capturing non-linear relationships in ecological data.
        * **Categorical Encoding:** Features like `wind_direction` or `stability_class` are automatically one-hot encoded (converted to binary numbers) so the math works.
        * **Validation:** We use internal testing (RMSE, R²) to check model health, but we highly recommend using the **Evaluation Tab** to simulate artificial gaps and see how the model performs on *your* specific dataset structure.
        """)


#Footer
if st.session_state.active_tab != "Home":
    st.markdown(
        """
        <div class='footer'>
            <p>© 2025 Max Anjos • Eddy System | Version 1.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    