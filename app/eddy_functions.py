
import streamlit as st
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Tuple, Dict, Union, Any
import io
import zipfile
# Eddy functions for ML model and streamlit app


def detect_and_preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = df.copy()

    # FLUXNET with hourly or half-hourly resolution
    if 'TIMESTAMP_START' in df.columns and 'TIMESTAMP_END' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
            df.set_index('datetime', inplace=True)
            df.replace(-9999, np.nan, inplace=True)
            df.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'], inplace=True)
            return df, "FLUXNET"
        except Exception as e:
            st.warning(f"⚠ Failed to parse FLUXNET hourly timestamp: {e}")

    # FLUXNET with daily resolution
    elif 'TIMESTAMP' in df.columns and 'NEE_VUT_REF' in df.columns:
        # Detect length of TIMESTAMP to infer granularity
        timestamp_sample = str(df['TIMESTAMP'].dropna().iloc[0])
        
        try:
            if len(timestamp_sample) == 6:  # Monthly (e.g., 200201)
                df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m')
            elif len(timestamp_sample) == 8:  # Daily (e.g., 20020101)
                df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d')
            elif len(timestamp_sample) == 10:  # Half-hourly or hourly (e.g., 2002010100)
                df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d%H')
            else:
                df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')  # Fallback

            df.set_index('datetime', inplace=True)
            df.replace(-9999, np.nan, inplace=True)
            df.drop(columns=['TIMESTAMP'], inplace=True)

            return df, "FLUXNET"
        except Exception as e:
            st.warning(f"⚠ Failed to parse FLUXNET timestamp: {e}")
            return df, "FLUXNET (timestamp error)"

    # ICOS
    elif 'TIMESTAMP' in df.columns and 'TIMESTAMP_END' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        return df, "ICOS"

    # AmeriFlux
    elif 'TIMESTAMP_START' in df.columns and 'FC' in df.columns:
        df['datetime'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['TIMESTAMP_START'], inplace=True)
        return df, "AMERIFLUX"

    # EC Tower / Custom
    elif 'date' in df.columns and 'time (UTC)' in df.columns and 'upward mole flux of carbon dioxide in air (1e-6 mol s-1 m-2) (co2_flux)' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time (UTC)'], format='%d.%m.%Y %H:%M:%S')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['date', 'time (UTC)'], inplace=True)
        return df, "EC Tower"



def upload_zip_and_extract_csv() -> Tuple[pd.DataFrame, str]:
    uploaded_zip = st.file_uploader("📦 Upload ZIP file with FLUXNET/ICOS/AmeriFlux CSV", type=["zip"])
    
    if uploaded_zip is not None:
        if zipfile.is_zipfile(uploaded_zip):
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                csv_files = [f for f in file_list if f.endswith('.csv') or f.endswith('.txt')]

                if not csv_files:
                    st.error("❌ No CSV or TXT files found in the ZIP.")
                    return None, "No CSV"

                selected_file = st.selectbox("📂 Select a file from ZIP:", csv_files)

                with zip_ref.open(selected_file) as file:
                    try:
                        # Read the file content into memory first
                        file_content = file.read()
                        
                        try:
                            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                        except pd.errors.ParserError:
                            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), skiprows=2)
                            
                        df, source = detect_and_preprocess_dataset(df)
                        st.success(f"✅ File loaded successfully. Dataset detected: **{source}**")
                        st.dataframe(df.head())
                        return df, source
                    except Exception as e:
                        st.error(f"❌ Failed to read CSV: {e}")
                        return None, "Error reading CSV"
        else:
            st.error("❌ Uploaded file is not a valid ZIP archive.")
            return None, "Not a ZIP"
    
    return None, "No file"

def load_example_data():
    """Load example eddy covariance dataset."""
    example_path = Path("data/eddy_covariance_data.csv")
    if example_path.exists():
        try:
            data = pd.read_csv(example_path)
            if 'date' in data.columns and 'time (UTC)' in data.columns:
                    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time (UTC)'], format='%d.%m.%Y %H:%M:%S')
                    data.set_index('datetime', inplace=True)
                    data = data.rename(columns = {
                        'date': 'date',
                        'time (UTC)': 'time',
                        'upward sensible heat flux in air (W m-2) (H)': 'sensible_heat_flux',
                        'quality flag upward sensible heat flux in air (cat) (qc_H)': 'sensible_heat_flux_qc',
                        'upward latent heat flux in air (W m-2) (LE)': 'latent_heat_flux',
                        'quality flag upward latent heat flux in air (cat) (qc_LE)': 'latent_heat_flux_qc',
                        'H2O signal strength infrared gasanalyser  (1) (H2O_sig_strgth_mean)': 'H2O_signal_strength',
                        'upward mole flux of carbon dioxide in air (1e-6 mol s-1 m-2) (co2_flux)': 'co2_flux',
                        'quality flag upward mole flux of carbon dioxide in air (cat) (qc_co2_flux)': 'co2_flux_qc',
                        'CO2 signal strength infrared gasanalyser (1) (CO2_sig_strgth_mean)': 'CO2_signal_strength',
                        'wind speed (m s-1) (wind_speed)': 'wind_speed',
                        'wind from direction (degree) (wind_dir)': 'wind_direction',
                        'friction velocity (m s-1) (u*)': 'friction_velocity',
                        'obukhov length (m) (L)': 'obukhov_length',
                        'monin-obukhov stability parameter (1) ((z-d)/L)': 'stability_parameter',
                        'bowen ratio (1) (bowen_ratio)': 'bowen_ratio',
                        'northward wind (m2 s-2) (v_var)': 'northward_wind',
                        'w wind component (m s-1) (w_unrot)': 'w_wind_component',
                        'air temperature (degree_C) (T_40m_Avg)': 'air_temperature',
                        'air temperature (degree_C) (T_1000cm_Avg)': 'air_temperature',
                        'relative humidity (%) (RH_40m_Avg)': 'relative_humidity',
                        'relative humidity (%) (RH_1000cm_Avg)': 'relative_humidity',
                        'downwelling shortwave flux in air (W m-2) (SPN_TOTAL_Avg)': 'downwelling_shortwave_flux_total',
                        'diffuse downwelling shortwave flux in air (W m-2) (SPN_DIFFUSE_Avg)': 'downwelling_shortwave_flux_diffuse',
                        'downwelling shortwave flux in air (W m-2) (Rs_downwell_Avg)': 'downwelling_shortwave_flux',
                        'downwelling longwave flux in air (W m-2) (Rl_downwell_Avg)': 'downwelling_longwave_flux',
                        'upwelling shortwave flux in air (W m-2) (Rs_upwell_Avg)': 'upwelling_shortwave_flux',
                        'upwelling longwave flux in air (W m-2) (Rl_upwell_Avg)': 'upwelling_longwave_flux',
                        'surface net downward radiative flux (W m-2)': 'net_radiative_flux'
                    })
            return data
        except Exception as e:
            st.error(f"Error loading example data: {str(e)}")
            return None
    else:
        st.error("Example data file not found.")
        return None

def create_time_features(data: pd.DataFrame, datetime_col='datetime') -> pd.DataFrame:
    """
    Adds comprehensive time-related features including cyclical transformations and categorical indicators.

    Args:
        data (pd.DataFrame): DataFrame with a datetime column.
        datetime_col (str): Name of the datetime column.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    if data[datetime_col].dtype != 'datetime64[ns]':
        data[datetime_col] = pd.to_datetime(data[datetime_col])

    df = data.copy()
    df['hour'] = df[datetime_col].dt.hour
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    df['year'] = df[datetime_col].dt.year
    df['dayofyear'] = df[datetime_col].dt.dayofyear
    df['weekday'] = df[datetime_col].dt.weekday
    df['quarter'] = df[datetime_col].dt.quarter
    df['weekofyear'] = df[datetime_col].dt.isocalendar().week.astype(int)

    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

    # Additional features
    df['hour_decimal'] = df['hour'] + df[datetime_col].dt.minute / 60.0

    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['season'] = df['month'].map(season_map)
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    return df

def create_lag_features(data: pd.DataFrame, target_cols: List[str], lag_periods: List[int] = [3, 4, 5, 6, 24, 168], shift_direction: str = 'backward') -> pd.DataFrame:
    """
    Adds lagged features to the DataFrame for specified columns and lag periods.

    Args:
        data (pd.DataFrame): Input DataFrame.
        target_cols (List[str]): List of column names to create lag features for.
        lag_periods (List[int], optional): List of lag periods. Defaults to [3, 4, 5, 6, 24, 168].
        shift_direction (str, optional): Direction of shift, 'backward' or 'forward'. Defaults to 'backward'.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    df_lag = data.copy()
    shift_sign = 1 if shift_direction.lower() == 'forward' else -1

    for col in target_cols:
        if col not in df_lag.columns:
            continue
        for lag in lag_periods:
            lag_name = f"{col}_lag_{lag}"
            df_lag[lag_name] = df_lag[col].shift(shift_sign * lag)

    return df_lag

def create_rolling_features(data: pd.DataFrame, target_cols: List[str], windows: List[int], stats: List[str] = [ 'mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Adds rolling window statistics of specified columns.

    Args:
        data (pd.DataFrame): DataFrame to add rolling features to.
        target_cols (List[str]): List of columns to create rolling statistics for.
        windows (List[int]): List of window sizes.
        stats (List[str]): List of statistics to calculate (e.g., 'mean', 'std', 'min', 'max').

    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    for target_col in target_cols:
        for window in windows:
            for stat in stats:
                if stat == 'mean':
                    data[f'{target_col}_rolling_{window}_{stat}'] = data[target_col].rolling(window=window, min_periods=1).mean()
                elif stat == 'std':
                    data[f'{target_col}_rolling_{window}_{stat}'] = data[target_col].rolling(window=window, min_periods=1).std()
                elif stat == 'min':
                    data[f'{target_col}_rolling_{window}_{stat}'] = data[target_col].rolling(window=window, min_periods=1).min()
                elif stat == 'max':
                    data[f'{target_col}_rolling_{window}_{stat}'] = data[target_col].rolling(window=window, min_periods=1).max()
                else:
                    raise ValueError(f"Invalid statistic: {stat}")
    return data

def calculate_vpd(data: pd.DataFrame, temp_col: str = 'air_temperature', rh_col: str = 'relative_humidity') -> pd.DataFrame:
    """
    Adds vapor pressure deficit (VPD) calculation to the DataFrame based on air temperature and relative humidity.

    Args:
        data (pd.DataFrame): Input DataFrame with air temperature and relative humidity.
        temp_col (str, optional): Column name for air temperature. Defaults to 'air_temperature'.
        rh_col (str, optional): Column name for relative humidity. Defaults to 'relative_humidity'.

    Returns:
        pd.DataFrame: DataFrame with added VPD feature.
    """
    df = data.copy()

    # Constants
    a = 0.611  # kPa
    b = 17.502  # dimensionless
    c = 240.97  # °C

    # Calculate saturation vapor pressure (SVP)
    svp = a * np.exp((b * df[temp_col]) / (c + df[temp_col]))

    # Calculate actual vapor pressure
    avp = svp * df[rh_col] / 100.0

    # Calculate vapor pressure deficit
    df['vpd'] = svp - avp

    return df

def calculate_potential_et(data: pd.DataFrame, temp_col: str = 'air_temperature', net_rad_col: str = 'net_radiative_flux', rh_col: str = 'relative_humidity', wind_col: str = 'wind_speed', height: float = 2.0) -> pd.DataFrame:
    """
    Adds potential evapotranspiration (PET) calculation to the DataFrame using the Penman-Monteith equation.

    Args:
        data (pd.DataFrame): Input DataFrame containing temperature, net radiation, relative humidity, and wind speed.
        temp_col (str, optional): Column name for air temperature. Defaults to 'air_temperature'.
        net_rad_col (str, optional): Column name for net radiation flux (W/m²). Defaults to 'net_radiative_flux'.
        rh_col (str, optional): Column name for relative humidity. Defaults to 'relative_humidity'.
        wind_col (str, optional): Column name for wind speed. Defaults to 'wind_speed'.
        height (float, optional): Measurement height in meters. Defaults to 2.0.

    Returns:
        pd.DataFrame: DataFrame with added PET feature.
    """
    df = data.copy()

    # Constants
    lambda_v = 2.45e6  # Latent heat of vaporization (J/kg)
    cp = 1013  # Specific heat at constant pressure (J/kg/K)
    rho_a = 1.225  # Air density (kg/m³)
    gamma = 0.067  # Psychrometric constant (kPa/°C)

    # Convert temperature to Kelvin
    temp_k = df[temp_col] + 273.15

    # Slope of saturation vapor pressure curve (kPa/°C)
    delta = 4098 * (0.6108 * np.exp((17.27 * df[temp_col]) / (df[temp_col] + 237.3))) / ((df[temp_col] + 237.3) ** 2)

    # Calculate vapor pressure deficit (kPa)
    vpd = calculate_vpd(df, temp_col, rh_col)['vpd']

    # Aerodynamic resistance (s/m)
    ra = (np.log(height / 0.01)) ** 2 / (0.41 ** 2 * df[wind_col])

    # Surface resistance (s/m) - reference crop
    rs = 70  # s/m

    # Net radiation in MJ/m²/day
    rn = df[net_rad_col] * 0.0864  # Convert W/m² to MJ/m²/day

    # Calculate potential ET (mm/day)
    df['potential_et'] = (delta * rn + rho_a * cp * vpd / ra) / (delta + gamma * (1 + rs / ra)) / lambda_v * 1000 * 86400

    return df

def create_met_features(data: pd.DataFrame) -> pd.DataFrame:
    
    df_met = data.copy()

    # Check available columns
    has_temp = 'air_temperature' in df_met.columns
    has_rh = 'relative_humidity' in df_met.columns
    has_wind = 'wind_speed' in df_met.columns
    has_rad_total = 'downwelling_shortwave_flux_total' in df_met.columns
    has_rad_diffuse = 'downwelling_shortwave_flux_diffuse' in df_met.columns

    # Calculate VPD
    if has_temp and has_rh:
        df_met = calculate_vpd(df_met)

    # Calculate potential ET
    if has_temp and has_rh and has_wind and has_rad_total:
        df_met = calculate_potential_et(df_met, net_rad_col='downwelling_shortwave_flux_total')

    # Diffuse radiation ratio
    if has_rad_total and has_rad_diffuse:
        valid_mask = (df_met['downwelling_shortwave_flux_total'] > 0)
        df_met.loc[valid_mask, 'diff_ratio'] = (
            df_met.loc[valid_mask, 'downwelling_shortwave_flux_diffuse'] /
            df_met.loc[valid_mask, 'downwelling_shortwave_flux_total']
        )

    # Stability class based on obukhov_length
    if 'obukhov_length' in df_met.columns:
        df_met['stability_class'] = pd.cut(
            df_met['obukhov_length'],
            bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
            labels=['very unstable', 'unstable', 'slightly unstable', 'neutral unstable',
                    'neutral stable', 'slightly stable', 'stable', 'very stable']
        )
        df_met['stability_index'] = pd.cut(
            df_met['obukhov_length'],
            bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
            labels=[-3, -2, -1, -0.5, 0.5, 1, 2, 3]
        ).astype(float)

    # Wind direction categories
    if 'wind_dir' in df_met.columns:
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        df_met['wind_dir_cat'] = pd.cut(
            df_met['wind_dir'] % 360,
            bins=np.arange(0, 361, 22.5),
            labels=dirs,
            include_lowest=True
        )
        wind_dir_dummies = pd.get_dummies(df_met['wind_dir_cat'], prefix='wind_dir')
        df_met = pd.concat([df_met, wind_dir_dummies], axis=1)

    return df_met

def encode_categorical_features(data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    
    df_encoded = data.copy()

    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded

@st.cache_resource(show_spinner=False)
def train_model(
    data: pd.DataFrame,
    target_col: str,
    selected_features: List[str],
    time_based_features: List[str],
    model_type: str = 'Random Forest',
    base_params: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Union[XGBRegressor, RandomForestRegressor], Union[XGBRegressor, RandomForestRegressor]]:
    """
    Trains two models: one with selected features and one with time-based features.
    Returns both trained models.
    """
    df_train_base = data.copy().dropna(subset=[target_col])

    X_train_all = df_train_base[selected_features]
    y_train_all = df_train_base[target_col]

    X_train_time = df_train_base[time_based_features]
    y_train_time = df_train_base[target_col]

    if base_params is None:
        base_params = {}

    if model_type == 'XGBoost':
        model_params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'n_estimators': 100,
            'learning_rate': 0.05,  # Slower but safer
            'max_depth': 4,  # Shallower trees generalize better
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
        model_all_features = XGBRegressor(**model_params)
        model_time_based = XGBRegressor(**model_params)
    else:
        model_params = {
        'n_estimators': 100,
        'max_depth': 10,  # Limit depth to avoid overfitting on small data
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',  # Good balance for regression
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
        }
        model_all_features = RandomForestRegressor(**model_params)
        model_time_based = RandomForestRegressor(**model_params)

    # Fit both models
    model_all_features.fit(X_train_all, y_train_all)
    model_time_based.fit(X_train_time, y_train_time)

    return model_all_features, model_time_based

# ✅ Improved introduce_nan with logging and reliability
def introduce_nan(data: pd.DataFrame, target_cols: list, nan_percentage: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """
    Introduce artificial NaNs into specified target columns for evaluation.

    Parameters:
    - data (pd.DataFrame): Dataset to inject NaNs into.
    - target_cols (list): Columns to target.
    - nan_percentage (float): Fraction of rows to nullify.
    - seed (int): Random seed.

    Returns:
    - pd.DataFrame with NaNs introduced.
    """
    df_nan = data.copy()
    rng = np.random.default_rng(seed)
    n_rows = len(df_nan)

    for col in target_cols:
        if col in df_nan.columns:
            available = df_nan[df_nan[col].notna()].index
            n_nan = int(len(available) * nan_percentage)
            if n_nan == 0:
                print(f"⚠️ No NaNs introduced in '{col}': not enough clean values.")
                continue
            selected = rng.choice(available, size=n_nan, replace=False)
            df_nan.loc[selected, col] = np.nan
            print(f"✅ Introduced {n_nan} NaNs in column '{col}'.")

    return df_nan

## Advanced Flux Visualization
def plot_flux_partitioning(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Flux Partitioning Analysis")
    
    # Create tabs for different analyses
    flux_tabs = st.tabs(["Energy Balance", "Carbon Flux", "Bowen Ratio"])
    
    with flux_tabs[0]:
        st.write("### Surface Energy Balance")
        
        # Calculate net radiation
        if all(col in df.columns for col in ['downwelling_shortwave_flux', 'upwelling_shortwave_flux', 'downwelling_longwave_flux', 'upwelling_longwave_flux']):
            df['Rnet'] = (df['downwelling_shortwave_flux'] - df['upwelling_shortwave_flux']) + (df['downwelling_longwave_flux'] - df['upwelling_longwave_flux'])
            
            # Create datetime for plotting if not exists
            if 'datetime' not in df.columns:
                 df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d.%m.%Y %H:%M:%S')
                 df.set_index('datetime', inplace=True)
                
            # Create the energy balance plot
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['Rnet'], name='Net Radiation (Rn)'))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['sensible_heat_flux'], name='Sensible Heat (H)'))
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['latent_heat_flux'], name='Latent Heat (LE)'))
            
            # Calculate energy balance closure
            df['energy_balance'] = df['sensible_heat_flux'] + df['latent_heat_flux']
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['energy_balance'], 
                                    name='sensible_heat_flux + LE', line=dict(dash='dash')))
            
            # Update layout
            fig.update_layout(
                title="Surface Energy Balance",
                xaxis_title="Date",
                yaxis_title="Energy Flux (W m⁻²)",
                legend_title="Components",
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display energy balance closure
            with st.expander("Energy Balance Closure Analysis"):
                # Linear regression of H+LE vs Rnet
                mask = (~df['Rnet'].isna()) & (~df['energy_balance'].isna())
                
                if mask.sum() > 10:  # Check if enough data points
                    x = df.loc[mask, 'Rnet']
                    y = df.loc[mask, 'energy_balance']
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Create scatter plot with regression line
                    fig = px.scatter(
                        x=x, y=y, 
                        labels={'x': 'Net Radiation (W m⁻²)', 'y': 'H + LE (W m⁻²)'},
                        title=f"Energy Balance Closure: y = {slope:.2f}x + {intercept:.2f}, R² = {r_value**2:.2f}"
                    )
                    
                    # Add regression line
                    fig.add_trace(
                        go.Scatter(
                            x=[x.min(), x.max()],
                            y=[slope * x.min() + intercept, slope * x.max() + intercept],
                            mode='lines',
                            line=dict(color='red'),
                            name=f'y = {slope:.2f}x + {intercept:.2f}'
                        )
                    )
                    
                    # Add 1:1 line
                    fig.add_trace(
                        go.Scatter(
                            x=[min(x.min(), y.min()), max(x.max(), y.max())],
                            y=[min(x.min(), y.min()), max(x.max(), y.max())],
                            mode='lines',
                            line=dict(color='black', dash='dash'),
                            name='1:1 line'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate energy balance closure ratio and residual
                    ebr = (df['sensible_heat_flux'] + df['latent_heat_flux']).sum() / df['Rnet'].sum()
                    residual = df['Rnet'] - (df['sensible_heat_flux'] + df['latent_heat_flux'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Energy Balance Ratio", f"{ebr:.2f}")
                    with col2:
                        st.metric("Mean Residual", f"{residual.mean():.1f} W m⁻²")
                    
                    st.write(f"""
                    **Interpretation:**
                    - Slope: {slope:.2f} (ideal = 1.0)
                    - R²: {r_value**2:.2f}
                    - Energy Balance Ratio: {ebr:.2f}
                    
                    A perfect energy balance closure would have a slope of 1, R² of 1, and energy balance ratio of 1.
                    """)
                else:
                    st.warning("Not enough valid data points for energy balance closure analysis.")
                    
    with flux_tabs[1]:
        st.write("### Carbon Flux Analysis")
        
        if 'co2_flux' in df.columns:
            # Create datetime for plotting if not exists
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d.%m.%Y %H:%M:%S')
                df.set_index('datetime', inplace=True)
                
            # Create carbon flux plot
            fig = go.Figure()
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['co2_flux'], 
                name='CO₂ Flux',
                mode='markers+lines',
                marker=dict(
                    color=df['co2_flux'],
                    colorscale='RdBu_r',
                    cmin=-10,
                    cmax=10,
                    colorbar=dict(title="μmol m⁻² s⁻¹")
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Carbon Dioxide Flux Time Series",
                xaxis_title="Date",
                yaxis_title="CO₂ Flux (μmol m⁻² s⁻¹)",
                height=600,
                template="plotly_white"
            )
            
            # Add horizontal line at zero
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Daily and Monthly Carbon Flux Patterns"):
                # Add hour and month columns
                df['hour'] = df['datetime'].dt.hour
                df['month'] = df['datetime'].dt.month
                df['day'] = df['datetime'].dt.day_name()
                
                tab1, tab2 = st.tabs(["Diurnal Pattern", "Monthly Pattern"])
                
                with tab1:
                    # Diurnal pattern
                    diurnal = df.groupby('hour')['co2_flux'].agg(['mean', 'std']).reset_index()
                    
                    fig = go.Figure()
                    
                    # Add mean line
                    fig.add_trace(go.Scatter(
                        x=diurnal['hour'],
                        y=diurnal['mean'],
                        mode='lines+markers',
                        name='Mean',
                        line=dict(color='blue')
                    ))
                    
                    # Add error bands
                    fig.add_trace(go.Scatter(
                        x=diurnal['hour'].tolist() + diurnal['hour'].tolist()[::-1],
                        y=(diurnal['mean'] + diurnal['std']).tolist() + (diurnal['mean'] - diurnal['std']).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,0,255,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Standard Deviation'
                    ))
                    
                    fig.update_layout(
                        title="Diurnal Pattern of CO₂ Flux",
                        xaxis_title="Hour of Day",
                        yaxis_title="CO₂ Flux (μmol m⁻² s⁻¹)",
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                        template="plotly_white"
                    )
                    
                    # Add horizontal line at zero
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    # Monthly pattern
                    monthly = df.groupby('month')['co2_flux'].mean().reset_index()
                    
                    # Create month names
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])
                    
                    fig = px.bar(
                        monthly,
                        x='month_name',
                        y='co2_flux',
                        title="Monthly Average CO₂ Flux",
                        labels={'co2_flux': 'CO₂ Flux (μmol m⁻² s⁻¹)', 'month_name': 'Month'},
                        color='co2_flux',
                        color_continuous_scale='RdBu_r',
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        xaxis=dict(categoryorder='array', categoryarray=month_names)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate carbon budget
                    if st.checkbox("Calculate Carbon Budget"):
                        # Convert from μmol m⁻² s⁻¹ to gC m⁻² day⁻¹ 
                        # Conversion: μmol CO₂ m⁻² s⁻¹ * 12.01 * 10⁻⁶ * 86400 / 1000
                        conversion_factor = 12.01 * 1e-6 * 86400 / 1000
                        
                        # Group by month and calculate daily sums
                        carbon_budget = df.groupby('month')['co2_flux'].agg(['mean', 'count']).reset_index()
                        carbon_budget['gC_m2_day'] = carbon_budget['mean'] * conversion_factor
                        
                        # Calculate days per month for total
                        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                        carbon_budget['days'] = carbon_budget['month'].apply(lambda x: days_in_month[x-1])
                        
                        # Calculate monthly total
                        carbon_budget['gC_m2_month'] = carbon_budget['gC_m2_day'] * carbon_budget['days']
                        
                        # Add month names
                        carbon_budget['month_name'] = carbon_budget['month'].apply(lambda x: month_names[x-1])
                        
                        # Plot monthly carbon budget
                        fig = px.bar(
                            carbon_budget,
                            x='month_name',
                            y='gC_m2_month',
                            title="Monthly Carbon Budget",
                            labels={'gC_m2_month': 'Carbon Budget (gC m⁻² month⁻¹)', 'month_name': 'Month'},
                            color='gC_m2_month',
                            color_continuous_scale='RdBu_r',
                            template="plotly_white"
                        )
                        
                        fig.update_layout(
                            xaxis=dict(categoryorder='array', categoryarray=month_names)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate annual carbon budget
                        annual_budget = carbon_budget['gC_m2_month'].sum()
                        
                        # Display annual budget
                        st.metric(
                            "Annual Carbon Budget", 
                            f"{annual_budget:.1f} gC m⁻² year⁻¹",
                            delta=None,
                            delta_color="normal"
                        )
                        
                        # Interpret result
                        if annual_budget < 0:
                            st.success(f"This ecosystem is a carbon sink, sequestering approximately {-annual_budget:.1f} gC m⁻² year⁻¹")
                        else:
                            st.warning(f"This ecosystem is a carbon source, emitting approximately {annual_budget:.1f} gC m⁻² year⁻¹")
                            
    with flux_tabs[2]:
        st.write("### Bowen Ratio Analysis")
        
        if all(col in df.columns for col in ['sensible_heat_flux', 'latent_heat_flux']):
            # Check for existing Bowen ratio or calculate
            if 'bowen_ratio' not in df.columns:
                df['bowen_ratio'] = df['sensible_heat_flux'] / df['latent_heat_flux']
                
            # Create datetime for plotting if not exists
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d.%m.%Y %H:%M:%S')
                df.set_index('datetime', inplace=True)
                
            # Add hour and month columns
            df['hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month
            
            # Create the Bowen ratio plot
            fig = go.Figure()
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['bowen_ratio'], 
                name='Bowen Ratio',
                mode='markers',
                marker=dict(
                    color=df['bowen_ratio'],
                    colorscale='Viridis',
                    cmin=0,
                    cmax=5,
                    colorbar=dict(title="Bowen Ratio")
                )
            ))
            
            # Update layout
            fig.update_layout(
                title="Bowen Ratio Time Series",
                xaxis_title="Date",
                yaxis_title="Bowen Ratio (H/LE)",
                height=600,
                template="plotly_white"
            )
            
            # Add horizontal line at one
            fig.add_hline(y=1, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bowen ratio patterns
            with st.expander("Bowen Ratio Patterns"):
                tab1, tab2 = st.tabs(["Diurnal Pattern", "Monthly Pattern"])
                
                with tab1:
                    # Filter out extreme values for better visualization
                    filtered_df = df[(df['bowen_ratio'] > -10) & (df['bowen_ratio'] < 10)]
                    
                    # Diurnal pattern
                    diurnal = filtered_df.groupby('hour')['bowen_ratio'].agg(['mean', 'std']).reset_index()
                    
                    fig = go.Figure()
                    
                    # Add mean line
                    fig.add_trace(go.Scatter(
                        x=diurnal['hour'],
                        y=diurnal['mean'],
                        mode='lines+markers',
                        name='Mean',
                        line=dict(color='green')
                    ))
                    
                    # Add error bands
                    fig.add_trace(go.Scatter(
                        x=diurnal['hour'].tolist() + diurnal['hour'].tolist()[::-1],
                        y=(diurnal['mean'] + diurnal['std']).tolist() + (diurnal['mean'] - diurnal['std']).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,128,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Standard Deviation'
                    ))
                    
                    fig.update_layout(
                        title="Diurnal Pattern of Bowen Ratio",
                        xaxis_title="Hour of Day",
                        yaxis_title="Bowen Ratio (H/LE)",
                        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
                        template="plotly_white"
                    )
                    
                    # Add horizontal line at one
                    fig.add_hline(y=1, line_dash="dash", line_color="black")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    # Filter out extreme values for better visualization
                    filtered_df = df[(df['bowen_ratio'] > -10) & (df['bowen_ratio'] < 10)]
                    
                    # Monthly pattern
                    monthly = filtered_df.groupby('month')['bowen_ratio'].mean().reset_index()
                    
                    # Create month names
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    monthly['month_name'] = monthly['month'].apply(lambda x: month_names[x-1])
                    
                    fig = px.bar(
                        monthly,
                        x='month_name',
                        y='bowen_ratio',
                        title="Monthly Average Bowen Ratio",
                        labels={'bowen_ratio': 'Bowen Ratio (H/LE)', 'month_name': 'Month'},
                        color='bowen_ratio',
                        color_continuous_scale='Viridis',
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        xaxis=dict(categoryorder='array', categoryarray=month_names)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
