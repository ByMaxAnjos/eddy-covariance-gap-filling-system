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
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Tuple, Dict, Union, Any
import requests
from streamlit_option_menu import option_menu
# from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
import re
import io
import textwrap
from eddy_functions import upload_zip_and_extract_csv,load_example_data, create_time_features, create_lag_features, create_rolling_features, calculate_vpd, create_met_features, encode_categorical_features
from eddy_functions import train_model, introduce_nan, plot_flux_partitioning
from eddy_functions import detect_and_preprocess_dataset
from eddy_functions import is_qc_like_column, match_qc_to_flux_column

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

LANGUAGE_OPTIONS = {
    "English": "en",
    "Português (Brasil)": "pt-BR",
    "Español": "es",
    "Français": "fr",
    "Deutsch": "de",
    "Italiano": "it",
    "Polski": "pl",
    "简体中文": "zh-CN",
    "日本語": "ja",
}

@st.cache_data(show_spinner=False)
def load_translations(language_code: str) -> Dict[str, str]:
    locale_path = Path("locales") / f"{language_code}.json"
    fallback_path = Path("locales") / "en.json"

    translations = {}
    if fallback_path.exists():
        with open(fallback_path, encoding="utf-8") as f:
            translations.update(json.load(f))
    if locale_path.exists() and locale_path != fallback_path:
        with open(locale_path, encoding="utf-8") as f:
            translations.update(json.load(f))
    return translations


@st.cache_data(show_spinner="Parsing uploaded file...")
def _load_and_detect_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return detect_and_preprocess_dataset(df)

def tr(key: str, default: Optional[str] = None, **kwargs) -> str:
    translations = st.session_state.get("translations", {})
    text = translations.get(key, default if default is not None else key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text

def translate_text(value):
    if isinstance(value, str):
        return tr(value, value)
    return value

def translate_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    translated = dict(kwargs)
    for key in ("label", "help", "placeholder", "text", "body"):
        if key in translated:
            translated[key] = translate_text(translated[key])
    return translated

def patch_streamlit_translation():
    method_names = [
        "title", "header", "subheader", "caption", "info", "success", "warning",
        "error", "button", "download_button", "file_uploader", "selectbox",
        "multiselect", "radio", "slider", "checkbox", "metric", "select_slider"
    ]

    # `st._main` is the stable root DeltaGenerator singleton -- unlike
    # `st.<name>`, it is never reassigned, so it's a safe anchor for finding
    # the owning Mixin class on every rerun (Streamlit re-executes this whole
    # script top to bottom on every interaction, so nothing at module scope
    # persists between reruns; only objects reachable from the `streamlit`
    # package itself, which stays imported, do).
    main_dg = st._main

    for method_name in method_names:
        if not hasattr(main_dg, method_name):
            continue

        owner_cls = None
        for klass in type(main_dg).__mro__:
            if method_name in klass.__dict__:
                owner_cls = klass
                break
        if owner_cls is None:
            continue

        original = owner_cls.__dict__[method_name]
        if not getattr(original, "_translation_patched", False):
            def make_wrapper(func):
                def wrapper(self, *args, **kwargs):
                    translated_args = list(args)
                    if translated_args:
                        translated_args[0] = translate_text(translated_args[0])
                    return func(self, *translated_args, **translate_kwargs(kwargs))
                wrapper._translation_patched = True
                return wrapper

            # Patch the Mixin class itself so every DeltaGenerator instance
            # created at runtime -- st.columns()/st.tabs()/st.container()
            # children, e.g. `col.metric(...)` -- resolves to the translated
            # version too; those instances look up the method live through
            # the class on each access, so this covers that call style.
            setattr(owner_cls, method_name, make_wrapper(original))

        # `st.<name>` is a bound-method snapshot Streamlit creates once at
        # its own import time, bound to `main_dg`. Patching the Mixin class
        # above does not retroactively change this already-created bound
        # method object, so direct `st.subheader(...)`-style calls (most of
        # this app's calls) would keep silently using the untranslated
        # original unless this snapshot is refreshed to point at the
        # (now patched) class method too.
        setattr(st, method_name, getattr(owner_cls, method_name).__get__(main_dg, owner_cls))

_original_colored_header = colored_header

def colored_header(label, description=None, color_name="green-70"):
    return _original_colored_header(
        label=translate_text(label),
        description=translate_text(description),
        color_name=color_name
    )

# Define app state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'filled_data' not in st.session_state:
    st.session_state.filled_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"
if 'language_code' not in st.session_state:
    st.session_state.language_code = "en"

st.session_state.translations = load_translations(st.session_state.language_code)
patch_streamlit_translation()

# Define color palette
colors = {
    "primary": "#1E5631",
    "secondary": "#4A8B41",
    "accent": "#88B04B",
    "neutral": "#F5F5F5",
    "text": "#333333",
    "highlight": "#3498DB"
}

def style_fig(fig, rangeslider=False, unified_hover=True, height=380):
    """Apply the app's brand theme to a Plotly figure (consistent look across tabs)."""
    fig.update_layout(
        template="plotly_white",
        colorway=[colors["primary"], colors["highlight"], colors["accent"], colors["secondary"]],
        font=dict(family="sans-serif", color=colors["text"]),
        title_font=dict(size=16, color=colors["primary"]),
        hovermode="x unified" if unified_hover else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#DDDDDD")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    if rangeslider:
        fig.update_xaxes(rangeslider_visible=True)
    return fig

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
language_labels = list(LANGUAGE_OPTIONS.keys())
current_language_label = next(
    (label for label, code in LANGUAGE_OPTIONS.items() if code == st.session_state.language_code),
    "English"
)

language_spacer, language_control = st.columns([0.72, 0.28])
with language_control:
    selected_language_label = st.selectbox(
        tr("Language", "Language"),
        language_labels,
        index=language_labels.index(current_language_label),
        key="language_selector",
        help=tr("Change the interface language", "Change the interface language"),
    )

selected_language_code = LANGUAGE_OPTIONS[selected_language_label]
if selected_language_code != st.session_state.language_code:
    st.session_state.language_code = selected_language_code
    st.session_state.translations = load_translations(selected_language_code)
    st.rerun()

TAB_KEYS = [
    "Home",
    "Upload & Explore",
    "Data Preprocessing",
    "Model Training",
    "Gap-Filling",
    "Gap-Fill Evaluation",
    "Advanced Flux Visualization",
    "About",
]
TAB_LABELS = {
    "Home": tr("nav.home", "Home"),
    "Upload & Explore": tr("nav.upload_explore", "Upload & Explore"),
    "Data Preprocessing": tr("nav.preprocessing", "Data Preprocessing"),
    "Model Training": tr("nav.model_training", "Model Training"),
    "Gap-Filling": tr("nav.gap_filling", "Gap-Filling"),
    "Gap-Fill Evaluation": tr("nav.evaluation", "Gap-Fill Evaluation"),
    "Advanced Flux Visualization": tr("nav.flux_visualization", "Advanced Flux Visualization"),
    "About": tr("nav.about", "About"),
}
TAB_LABEL_TO_KEY = {label: key for key, label in TAB_LABELS.items()}

with st.container():
    selected_tab = option_menu(
        menu_title=None,
        options=[TAB_LABELS[tab] for tab in TAB_KEYS],
        icons=[
            "house", 
            "cloud-upload", 
            "gear", 
            "cpu", 
            "puzzle", 
            "graph-up",
            "bar-chart",
            "book"
        ],
        menu_icon="cast",
        default_index=TAB_KEYS.index(st.session_state.active_tab)
        if st.session_state.get("active_tab") in TAB_KEYS
        else 0,
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
    st.session_state.active_tab = TAB_LABEL_TO_KEY.get(selected_tab, "Home")

# Home
if st.session_state.active_tab == "Home":
    def encode_asset(path: str) -> str:
        with open(path, "rb") as asset_file:
            return base64.b64encode(asset_file.read()).decode()

    logo_encoded = encode_asset("static/favicon.png")
    animation_encoded = encode_asset("static/home_gapfill_animation.gif")
    max_photo_encoded = encode_asset("static/max_photo.jpg")
    fred_photo_encoded = encode_asset("static/fred_photo.jpg")
    github_issues_url = "https://github.com/ByMaxAnjos/eddy-covariance-gap-filling-system/issues"

    st.markdown(
        textwrap.dedent(f"""
        <style>
            .home-shell {{
                margin-top: 1.25rem;
            }}
            .home-hero {{
                display: grid;
                grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
                gap: 2rem;
                align-items: center;
                padding: 2.4rem 0 2.2rem;
                border-bottom: 1px solid #dfe7dc;
            }}
            .home-brand {{
                display: flex;
                align-items: center;
                gap: 0.85rem;
                margin-bottom: 1.2rem;
            }}
            .home-brand img {{
                width: 70px;
                height: auto;
            }}
            .home-kicker {{
                color: #4A8B41;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }}
            .home-title {{
                color: #173f25;
                font-size: 2.62rem;
                line-height: 1.08;
                font-weight: 760;
                margin: 0 0 1rem;
                letter-spacing: 0;
            }}
            .home-subtitle {{
                color: #4c5c50;
                font-size: 1.08rem;
                line-height: 1.65;
                max-width: 760px;
                margin: 0 0 1.45rem;
            }}
            .home-actions {{
                display: flex;
                gap: 0.75rem;
                flex-wrap: wrap;
                margin-bottom: 1.4rem;
            }}
            .home-action-primary,
            .home-action-secondary {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                border-radius: 7px;
                padding: 0.68rem 0.95rem;
                font-weight: 700;
            }}
            .home-action-primary {{
                background: #1E5631;
                color: white;
            }}
            .home-action-secondary {{
                border: 1px solid #bfd0be;
                color: #1E5631;
                background: #fbfdfb;
            }}
            .home-meta {{
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
            }}
            .home-meta span {{
                border: 1px solid #d8e4d6;
                border-radius: 999px;
                color: #45604b;
                background: #fbfdfb;
                padding: 0.38rem 0.7rem;
                font-size: 0.82rem;
                font-weight: 650;
            }}
            .home-visual {{
                border: 1px solid #dce6da;
                border-radius: 8px;
                background: #f8faf7;
                padding: 0.85rem;
                box-shadow: 0 18px 44px rgba(32, 62, 38, 0.09);
            }}
            .home-visual img {{
                width: 100%;
                border-radius: 6px;
                display: block;
            }}
            .home-section {{
                margin-top: 2.3rem;
            }}
            .home-section-head {{
                display: flex;
                justify-content: space-between;
                align-items: end;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .home-section-head h2 {{
                color: #173f25;
                font-size: 1.35rem;
                margin: 0;
            }}
            .home-section-head p {{
                color: #657268;
                margin: 0;
                font-size: 0.92rem;
            }}
            .home-feature-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
            }}
            .home-feature {{
                border: 1px solid #dfe7dc;
                border-radius: 8px;
                background: white;
                padding: 1.15rem;
                min-height: 162px;
            }}
            .home-svg {{
                width: 1.35rem;
                height: 1.35rem;
                stroke: currentColor;
                stroke-width: 2;
                stroke-linecap: round;
                stroke-linejoin: round;
                fill: none;
                display: block;
            }}
            .home-feature .home-svg {{
                color: #1E5631;
            }}
            .home-feature h3 {{
                color: #223628;
                font-size: 1rem;
                margin: 0.75rem 0 0.42rem;
            }}
            .home-feature p {{
                color: #5d6a60;
                font-size: 0.9rem;
                line-height: 1.55;
                margin: 0;
            }}
            .home-workflow {{
                display: grid;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                border: 1px solid #dfe7dc;
                border-radius: 8px;
                overflow: hidden;
                background: white;
            }}
            .home-step {{
                padding: 1rem 0.9rem;
                border-right: 1px solid #edf2eb;
                min-height: 120px;
            }}
            .home-step:last-child {{
                border-right: none;
            }}
            .home-step span {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: #eef5ec;
                color: #1E5631;
                margin-bottom: 0.65rem;
            }}
            .home-step .home-svg {{
                width: 1.05rem;
                height: 1.05rem;
            }}
            .home-step strong {{
                display: block;
                color: #213728;
                font-size: 0.92rem;
            }}
            .home-step small {{
                color: #6b776d;
                line-height: 1.35;
            }}
            .home-datasets {{
                display: flex;
                gap: 0.65rem;
                flex-wrap: wrap;
            }}
            .home-datasets a,
            .home-datasets span {{
                border: 1px solid #d9e5d7;
                border-radius: 999px;
                padding: 0.55rem 0.8rem;
                color: #1E5631 !important;
                background: #fbfdfb;
                text-decoration: none !important;
                font-size: 0.9rem;
                font-weight: 700;
            }}
            .home-logo-row {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 1rem;
            }}
            .home-logo-card {{
                border: 1px solid #d9e5d7;
                border-radius: 8px;
                background: white;
                min-height: 96px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0.9rem;
            }}
            .home-logo-card img {{
                max-width: 175px;
                max-height: 58px;
                object-fit: contain;
            }}
            .home-logo-card span {{
                color: #1E5631;
                font-weight: 750;
            }}
            .home-footer-panel {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                gap: 1rem;
                align-items: center;
                margin-top: 2.4rem;
                padding: 1rem 0;
                border-top: 1px solid #dfe7dc;
            }}
            .home-people {{
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }}
            .home-person {{
                display: flex;
                align-items: center;
                gap: 0.7rem;
                color: #4d5d52;
            }}
            .home-person img {{
                width: 44px;
                height: 44px;
                object-fit: cover;
                border-radius: 50%;
            }}
            .home-person strong {{
                display: block;
                color: #213728;
                font-size: 0.92rem;
            }}
            .home-person a {{
                color: #1E5631 !important;
                font-size: 0.82rem;
                margin-right: 0.45rem;
                text-decoration: none !important;
            }}
            .home-feedback {{
                border: 1px solid #d9e5d7;
                border-radius: 8px;
                padding: 0.8rem 0.95rem;
                background: #fbfdfb;
                color: #4d5d52;
                min-width: 250px;
            }}
            .home-feedback strong {{
                display: block;
                color: #213728;
                margin-bottom: 0.2rem;
                font-size: 0.92rem;
            }}
            .home-feedback a {{
                color: #1E5631 !important;
                font-weight: 700;
                font-size: 0.86rem;
                text-decoration: none !important;
            }}
            .home-citation {{
                margin-top: 0.75rem;
                color: #5e6d62;
                font-size: 0.78rem;
                line-height: 1.45;
            }}
            .home-citation a {{
                color: #1E5631 !important;
                font-weight: 650;
                text-decoration: none !important;
            }}
            @media (max-width: 900px) {{
                .home-hero,
                .home-feature-grid,
                .home-footer-panel {{
                    grid-template-columns: 1fr;
                }}
                .home-title {{
                    font-size: 2.25rem;
                }}
                .home-workflow {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
                .home-logo-row {{
                    grid-template-columns: 1fr;
                }}
                .home-step {{
                    border-bottom: 1px solid #edf2eb;
                }}
            }}
        </style>
        <div class="home-hero">
            <div>
                <div class="home-brand">
                    <img src="data:image/png;base64,{logo_encoded}" alt="Eddy System favicon"/>
                    <div class="home-kicker">{tr("home.kicker", "Machine learning for environmental time series")}</div>
                </div>
                <h1 class="home-title">{tr("home.title", "Universal Time Series & Eddy Covariance Gap-Filling System")}</h1>
                <p class="home-subtitle">
                    {tr("home.subtitle", "Gap filling with evidence, not guesswork. Upload time-series data, inspect missingness, train validated Random Forest or XGBoost models, and export gap-filled results with diagnostics built for flux and environmental datasets.")}
                </p>
                <div class="home-actions">
                    <span class="home-action-primary">{tr("home.action.upload", "Upload data")}</span>
                    <span class="home-action-secondary">{tr("home.action.train", "Train validated models")}</span>
                </div>
                <div class="home-meta">
                    <span>{tr("home.meta.validation", "Gap-aware validation")}</span>
                    <span>{tr("home.meta.predictors", "Manual predictors")}</span>
                    <span>{tr("home.meta.fallback", "Fallback models")}</span>
                    <span>{tr("home.meta.workflow", "Flux-ready workflow")}</span>
                </div>
            </div>
            <div class="home-visual">
                <img src="data:image/gif;base64,{animation_encoded}" alt="Animated time-series gap filling preview"/>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    icons = {
        "activity": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M3 12h4l3-8 4 16 3-8h4'/></svg>",
        "sliders": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M4 6h7'/><path d='M15 6h5'/><path d='M4 12h3'/><path d='M11 12h9'/><path d='M4 18h10'/><path d='M18 18h2'/><circle cx='13' cy='6' r='2'/><circle cx='9' cy='12' r='2'/><circle cx='16' cy='18' r='2'/></svg>",
        "clipboard": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M9 4h6l1 2h3v14H5V6h3z'/><path d='M9 10h6'/><path d='M9 14h4'/></svg>",
        "upload": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M12 16V4'/><path d='m7 9 5-5 5 5'/><path d='M5 20h14'/></svg>",
        "search": "<svg class='home-svg' viewBox='0 0 24 24'><circle cx='11' cy='11' r='6'/><path d='m16 16 4 4'/></svg>",
        "funnel": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M4 5h16l-6 7v5l-4 2v-7z'/></svg>",
        "cpu": "<svg class='home-svg' viewBox='0 0 24 24'><rect x='7' y='7' width='10' height='10' rx='2'/><path d='M9 1v3'/><path d='M15 1v3'/><path d='M9 20v3'/><path d='M15 20v3'/><path d='M1 9h3'/><path d='M1 15h3'/><path d='M20 9h3'/><path d='M20 15h3'/></svg>",
        "fill": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M4 18c4-8 8-8 16-12'/><path d='M7 18h10'/><path d='M17 6h3v3'/></svg>",
        "download": "<svg class='home-svg' viewBox='0 0 24 24'><path d='M12 4v12'/><path d='m7 11 5 5 5-5'/><path d='M5 20h14'/></svg>",
    }

    st.markdown(
        f"<div class='home-section-head'><h2>{tr('home.features.heading', 'What The App Does')}</h2><p>{tr('home.features.description', 'Focused tools for inspecting gaps, training models, and evaluating fills.')}</p></div>",
        unsafe_allow_html=True
    )
    feature_cols = st.columns(3)
    feature_cards = [
        ("activity", tr("home.feature.detect.title", "Detect missing structure"), tr("home.feature.detect.body", "Summarize gaps, inspect time-series behavior, and identify patterns before modeling.")),
        ("sliders", tr("home.feature.predictors.title", "Control predictors"), tr("home.feature.predictors.body", "Select measured variables manually and optionally add engineered temporal and meteorological drivers.")),
        ("clipboard", tr("home.feature.validate.title", "Validate model behavior"), tr("home.feature.validate.body", "Use chronological or blocked time-series validation with bias, slope, residual spread, and feature importance.")),
    ]
    for col, (icon, title, text) in zip(feature_cols, feature_cards):
        with col:
            st.markdown(
                f"<div class='home-feature'>{icons[icon]}<h3>{title}</h3><p>{text}</p></div>",
                unsafe_allow_html=True
            )

    st.markdown(
        f"<div class='home-section-head home-section'><h2>{tr('home.workflow.heading', 'Workflow')}</h2><p>{tr('home.workflow.description', 'From raw observations to exportable filled series.')}</p></div>",
        unsafe_allow_html=True
    )
    workflow_cols = st.columns(6)
    workflow_steps = [
        ("upload", tr("home.workflow.upload.title", "Upload"), tr("home.workflow.upload.body", "CSV, ZIP, or example data")),
        ("search", tr("home.workflow.explore.title", "Explore"), tr("home.workflow.explore.body", "Missingness and time-series overview")),
        ("funnel", tr("home.workflow.prepare.title", "Prepare"), tr("home.workflow.prepare.body", "Quality control and gap structure")),
        ("cpu", tr("home.workflow.train.title", "Train"), tr("home.workflow.train.body", "RF or XGBoost with validation")),
        ("fill", tr("home.workflow.fill.title", "Fill"), tr("home.workflow.fill.body", "Apply full or fallback model")),
        ("download", tr("home.workflow.export.title", "Export"), tr("home.workflow.export.body", "Download clean results")),
    ]
    for col, (icon, title, text) in zip(workflow_cols, workflow_steps):
        with col:
            st.markdown(
                f"<div class='home-step'><span>{icons[icon]}</span><strong>{title}</strong><small>{text}</small></div>",
                unsafe_allow_html=True
            )

    st.markdown(
        textwrap.dedent(f"""
        <div class="home-section-head home-section">
            <h2>{tr("home.inputs.heading", "Supported Inputs")}</h2>
            <p>{tr("home.inputs.description", "Designed for flux towers, environmental monitoring, and sensor time series.")}</p>
        </div>
        <div class="home-datasets">
            <span>{tr("home.inputs.custom", "Custom environmental data")}</span>
            <a href="https://fluxnet.org/" target="_blank">FLUXNET</a>
            <a href="https://ameriflux.lbl.gov/" target="_blank">AmeriFlux</a>
            <a href="https://icos-ri.eu/" target="_blank">ICOS</a>
            <span>{tr("home.inputs.met", "Meteorological stations")}</span>
            <span>{tr("home.inputs.hydro", "Hydrology and energy series")}</span>
        </div>
        <div class="home-logo-row">
            <a class="home-logo-card" href="https://fluxnet.org/" target="_blank">
                <img src="https://upload.wikimedia.org/wikipedia/commons/a/ac/Fluxnet_Logo.jpg" alt="FLUXNET logo"/>
            </a>
            <a class="home-logo-card" href="https://ameriflux.lbl.gov/" target="_blank">
                <img src="https://ameriflux.lbl.gov/wp-content/uploads/2014/06/Logo-AmerifluxNet-Horiz1.png" alt="AmeriFlux logo"/>
            </a>
            <a class="home-logo-card" href="https://icos-ri.eu/" target="_blank">
                <img src="https://www.icos-cp.eu/media/253" alt="ICOS logo"/>
            </a>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="home-footer-panel">
            <div class="home-people">
                <div class="home-person">
                    <img src="data:image/jpeg;base64,{max_photo_encoded}" alt="Max Anjos"/>
                    <div>
                        <strong>Max Anjos</strong>
                        <span>{tr("home.people.max.role", "Professor at UFJF")}</span><br/>
                        <a href="https://github.com/maxanjos" target="_blank"><i class="bi bi-github"></i> GitHub</a>
                        <a href="https://www.linkedin.com/in/maxanjos/" target="_blank"><i class="bi bi-linkedin"></i> LinkedIn</a>
                    </div>
                </div>
                <div class="home-person">
                    <img src="data:image/jpeg;base64,{fred_photo_encoded}" alt="Fred Meier"/>
                    <div>
                        <strong>Fred Meier</strong>
                        <a href="https://www.tu.berlin/en/klima/about-us/meier-fred" target="_blank"><i class="bi bi-building"></i> TU-Berlin</a>
                        <a href="mailto:fred.meier@tu-berlin.de"><i class="bi bi-envelope"></i> Email</a>
                    </div>
                </div>
            </div>
            <div class="home-feedback">
                <strong>{tr("home.feedback.title", "Contribute and give feedback")}</strong>
                <span>{tr("home.feedback.body", "Ideas, issues, and bug reports are welcome.")}</span><br/>
                <a href="{github_issues_url}" target="_blank"><i class="bi bi-github"></i> {tr("home.feedback.issue", "Open an issue")}</a>
                <div class="home-citation">
                    <strong>{tr("home.citation.label", "Citation:")}</strong> Anjos, M., &amp; Meier, F. (2026).
                    Universal Time Series &amp; Eddy Covariance Gap-Filling System
                    (<a href="https://github.com/ByMaxAnjos/eddy-covariance-gap-filling-system" target="_blank">v1.0.0</a>).
                    Zenodo. <a href="https://doi.org/10.5281/zenodo.20412105" target="_blank">https://doi.org/10.5281/zenodo.20412105</a>
                </div>
            </div>
        </div>
        <div class='footer'>
            <p>© 2025 Max Anjos • Eddy Covariance Gap-Filling System | Version 1.0</p>
        </div>
        """),
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
        upload_method = st.radio("Choose data upload format:", ["Upload CSV", "Upload ZIP", "Use example dataset"], format_func=tr)
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("📄 Upload CSV or TXT", type=["csv", "txt"])
            if uploaded_file is not None:
                try:
                    # Streamlit reruns this whole script on every widget interaction
                    # (e.g. the variable selectors further down this tab). Without
                    # caching, a large CSV gets re-parsed from scratch on every one
                    # of those reruns even though the uploaded file hasn't changed.
                    df, source = _load_and_detect_csv(uploaded_file.getvalue())
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    st.success(f"✅ File loaded. Format detected: **{source}**")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        elif upload_method == "Upload ZIP":
            df, source = upload_zip_and_extract_csv()
            if df is not None:
                st.session_state.data = df
                st.session_state.original_data = df.copy()
        elif upload_method == "Use example dataset":
            df = load_example_data()
            if df is not None:
                st.session_state.data = df
                st.session_state.original_data = df.copy()
                st.success("✅ Example dataset loaded successfully!")

    with col2:
        if st.session_state.data is not None:
            st.subheader("Dataset Overview")

            data = st.session_state.data
            total_cells = data.shape[0] * data.shape[1]
            missing_pct_overall = (data.isna().sum().sum() / total_cells * 100) if total_cells else 0

            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) > 1:
                span_days = (data.index[-1] - data.index[0]).days
                date_range_label = f"{span_days} days"
                resolution = data.index.to_series().diff().median()
                resolution_label = f"{resolution.total_seconds() / 60:.0f} min" if pd.notna(resolution) else "n/a"
            else:
                date_range_label = "n/a"
                resolution_label = "n/a"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{data.shape[0]:,}")
            m2.metric("Columns", f"{data.shape[1]:,}")
            m3.metric("Time span", date_range_label)
            m4.metric("Resolution", resolution_label)
            style_metric_cards()

            st.caption(tr("Overall missing data: **{pct}%** of all cells").format(pct=f"{missing_pct_overall:.1f}"))
            st.write(tr("First 5 rows:"))
            st.dataframe(data.head(), use_container_width=True)

    # Data Analysis Section
    if st.session_state.data is not None:
        st.header("Data Analysis")

        # Missing values analysis
        st.subheader("Top Missing Values Overview")

        flux_vars = ["date", "time", "datetime", "co2_flux_qc"]
        actual_flux_cols = [col for col in st.session_state.data.columns if col not in flux_vars]
        numeric_overview_cols = [
            col for col in actual_flux_cols
            if pd.api.types.is_numeric_dtype(st.session_state.data[col])
        ]

        col1, col2 = st.columns([1, 2])

        with col1:
            # Missing values summary
            missing_values = st.session_state.data.isna().sum().sort_values(ascending=False)
            missing_values_pct = (missing_values / len(st.session_state.data) * 100).round(2)
            missing_values_col = tr('Missing Values')
            percentage_col = tr('Percentage (%)')
            missing_df = pd.DataFrame({
                missing_values_col: missing_values,
                percentage_col: missing_values_pct
            })
            #st.dataframe(missing_df.head(10))
              # Apply conditional formatting
            styled_df = missing_df.style\
                .background_gradient(subset=missing_values_col, cmap='Reds')\
                .format({percentage_col: "{:.2f}%"})

            st.dataframe(styled_df, use_container_width=True)

        with col2:
            overview_mode = st.radio(
                "Initial data overview:",
                ["Time series", "Missing-data heatmap", "Both"],
                horizontal=True,
                help=tr("Choose the first visual inspection plot for the uploaded data."),
                format_func=tr
            )
            if overview_mode in ["Time series", "Both"] and numeric_overview_cols:
                overview_var = st.selectbox(
                    tr("Variable for initial time-series overview:"),
                    numeric_overview_cols,
                    index=0
                )
                data_reset = st.session_state.data.reset_index()
                fig = px.line(
                    data_reset,
                    x='datetime' if 'datetime' in data_reset.columns else data_reset.index,
                    y=overview_var,
                    title=tr("Time Series Overview: {var}").format(var=overview_var)
                )
                fig.update_traces(line=dict(color=colors["primary"], width=1.5))
                style_fig(fig, rangeslider=True, height=350)
                st.plotly_chart(fig, use_container_width=True)

            if overview_mode in ["Missing-data heatmap", "Both"]:
                missing_matrix = st.session_state.data.isna()
                fig = go.Figure(data=go.Heatmap(
                    z=missing_matrix.T.values.astype(int),
                    x=missing_matrix.index,
                    y=missing_matrix.columns,
                    colorscale=[[0, "#F5F5F5"], [1, colors["primary"]]],
                    showscale=False,
                    hovertemplate=tr("Variable: %{y}<br>Date: %{x}<br>Missing: %{z}<extra></extra>"),
                ))
                fig.update_layout(title=tr("Missing Data Map (green = missing)"))
                style_fig(fig, height=350, unified_hover=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)


        # Column selection for exploration
        st.subheader("Variable Exploration")

        selected_variable = st.selectbox(
            tr("Select Variable to Analyze:"),
            actual_flux_cols,
            index=0 if actual_flux_cols else None
        )

        if selected_variable:
            col1, col2 = st.columns([1, 1])

            with col1:
                # Time series plot
                data_reset = st.session_state.data.reset_index()
                fig = px.line(
                    data_reset,
                    x='datetime' if 'datetime' in data_reset.columns else data_reset.index,
                    y=selected_variable,
                    title=tr("Time Series: {var}").format(var=selected_variable)
                )
                fig.update_traces(line=dict(color=colors["primary"], width=1.5))
                style_fig(fig, rangeslider=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribution plot
                series = st.session_state.data[selected_variable].dropna()
                fig = px.histogram(
                    st.session_state.data,
                    x=selected_variable,
                    title=tr("Distribution: {var}").format(var=selected_variable),
                    marginal="box",
                    color_discrete_sequence=[colors["primary"]],
                )
                if not series.empty:
                    fig.add_vline(
                        x=series.mean(), line_dash="dash", line_color=colors["highlight"],
                        annotation_text=tr("mean = {value}").format(value=f"{series.mean():.2f}"), annotation_position="top",
                        row=2, col=1,
                    )
                style_fig(fig, unified_hover=False)
                st.plotly_chart(fig, use_container_width=True)

            # Daily/monthly patterns
            if isinstance(st.session_state.data.index, pd.DatetimeIndex):
                st.subheader("Temporal Patterns")

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Diurnal pattern: mean +/- 1 std band, more informative than the mean line alone
                    grouped = st.session_state.data[selected_variable].groupby(st.session_state.data.index.hour)
                    hourly_mean = grouped.mean()
                    hourly_std = grouped.std().fillna(0)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(hourly_mean.index) + list(hourly_mean.index[::-1]),
                        y=list(hourly_mean + hourly_std) + list((hourly_mean - hourly_std)[::-1]),
                        fill='toself', fillcolor='rgba(30, 86, 49, 0.15)',
                        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
                        name=tr('±1 std'), showlegend=True,
                    ))
                    fig.add_trace(go.Scatter(
                        x=hourly_mean.index, y=hourly_mean.values,
                        mode='lines+markers', name=tr('Mean'),
                        line=dict(color=colors["primary"], width=2), marker=dict(size=5),
                    ))
                    fig.update_layout(
                        title=tr("Diurnal Pattern: {var}").format(var=selected_variable),
                        xaxis_title=tr('Hour of Day'), yaxis_title=selected_variable,
                    )
                    style_fig(fig)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Monthly pattern
                    if len(st.session_state.data.index) > 28:
                        st.session_state.data['month'] = st.session_state.data.index.month
                        fig = px.box(
                            st.session_state.data,
                            x='month',
                            y=selected_variable,
                            points="outliers",
                            title=tr("Monthly Distribution: {var}").format(var=selected_variable),
                            labels={'month': tr('Month'), selected_variable: selected_variable},
                            color_discrete_sequence=[colors["primary"]],
                        )
                        style_fig(fig, unified_hover=False)
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

        # Candidate columns for outlier detection: numeric, non-temporal, and NOT
        # QC/flag columns (those are categorical codes like 0/1/2, not continuous
        # measurements -- running IQR/Z-score on them is meaningless).
        non_numeric_like = ["date", "time", "datetime", "month"]
        numeric_cols_all = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        actual_flux_cols_out = [
            col for col in numeric_cols_all
            if col not in non_numeric_like and not is_qc_like_column(col)
        ]

        # Candidate columns for QC filtering: anything that looks like a flag column
        qc_like_cols = [col for col in st.session_state.data.columns if is_qc_like_column(col)]

        # Vocabulary used as a fallback to pair a QC column with its flux column
        # when the name doesn't reduce to it exactly (verbose custom headers).
        flux_vars = ["co2_flux", "latent_heat_flux", "sensible_heat_flux"]

        col1, col2 = st.columns([1, 1])

        with col1:
            # QA/QC parameters
            st.write(tr("Set QA/QC Parameters:"))
            remove_outliers = st.checkbox("Remove outliers", value=False)

            if remove_outliers:
                outlier_method = st.selectbox(
                    "Outlier detection method:",
                    ["IQR", "Z-score", "Modified Z-score"],
                    index=0,
                    help="Applied only to continuous numeric variables (QC/flag columns are excluded automatically)."
                )
                if outlier_method == "IQR":
                    iqr_factor = st.slider("IQR factor", 1.5, 3.0, 1.5, 0.1)
                elif outlier_method == "Z-score":
                    z_threshold = st.slider("Z-score threshold", 2.0, 8.0, 3.0, 0.1)
                elif outlier_method == "Modified Z-score":
                    mod_z_threshold = st.slider("Modified Z-score threshold", 3.0, 10.0, 3.5, 0.1)

                # Live preview so users can tune the threshold before committing
                preview_flagged = 0
                preview_total = 0
                for flux_col in actual_flux_cols_out:
                    series = st.session_state.data[flux_col]
                    if outlier_method == "IQR":
                        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (series < Q1 - iqr_factor * IQR) | (series > Q3 + iqr_factor * IQR)
                    elif outlier_method == "Z-score":
                        mask = ((series - series.mean()) / series.std()).abs() > z_threshold
                    else:
                        median = series.median()
                        mad = np.median(np.abs(series - median)) * 1.4826
                        mask = ((series - median) / mad).abs() > mod_z_threshold if mad else pd.Series(False, index=series.index)
                    preview_flagged += int(mask.sum())
                    preview_total += int(series.notna().sum())
                preview_pct = (preview_flagged / preview_total * 100) if preview_total else 0
                st.caption(tr("🔍 Preview: **{count}** values (**{pct}%** of valid data) across {n_vars} variables would be flagged as outliers.").format(
                    count=f"{preview_flagged:,}", pct=f"{preview_pct:.2f}", n_vars=len(actual_flux_cols_out)
                ))

            apply_qc_flags = st.checkbox("Apply quality-control filter", value=False)

            if apply_qc_flags:
                qc_columns = st.multiselect(
                    "Select QC / flag column(s):",
                    options=st.session_state.data.columns.tolist(),
                    default=qc_like_cols,
                    help="Pre-selected: columns whose name contains 'qc'."
                )
                if qc_columns:
                    qc_thresholds = {}
                    for qc_col in qc_columns:
                        qc_min = int(st.session_state.data[qc_col].min(skipna=True))
                        qc_max = int(st.session_state.data[qc_col].max(skipna=True))
                        qc_thresholds[qc_col] = st.slider(
                            tr("Maximum acceptable value for **{qc_col}**:").format(qc_col=qc_col),
                            min_value=qc_min,
                            max_value=qc_max,
                            value=min(qc_max, 1),
                            step=1,
                            help = "0 = Highest quality, 1 = Medium quality, 2 = Low quality (or custom scale).Please, ensure the correct QC value. Check metadata out!"
                        )
                else:
                    st.info("Select at least one QC column to proceed.")

        with col2:
            if remove_outliers:
                st.write(tr("Outlier impact by variable:"))
                per_col_flagged = {}
                for flux_col in actual_flux_cols_out:
                    series = st.session_state.data[flux_col]
                    if outlier_method == "IQR":
                        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (series < Q1 - iqr_factor * IQR) | (series > Q3 + iqr_factor * IQR)
                    elif outlier_method == "Z-score":
                        mask = ((series - series.mean()) / series.std()).abs() > z_threshold
                    else:
                        median = series.median()
                        mad = np.median(np.abs(series - median)) * 1.4826
                        mask = ((series - median) / mad).abs() > mod_z_threshold if mad else pd.Series(False, index=series.index)
                    valid = int(series.notna().sum())
                    per_col_flagged[flux_col] = (int(mask.sum()), (mask.sum() / valid * 100) if valid else 0)

                variable_col, flagged_col, pct_col = tr("Variable"), tr("Flagged"), tr("Percentage (%)")
                impact_df = pd.DataFrame(
                    [(k, v[0], v[1]) for k, v in per_col_flagged.items()],
                    columns=[variable_col, flagged_col, pct_col]
                ).sort_values(flagged_col, ascending=False).set_index(variable_col)
                styled_impact = impact_df.style\
                    .background_gradient(subset=flagged_col, cmap='Reds')\
                    .format({pct_col: "{:.2f}%"})
                st.dataframe(styled_impact, use_container_width=True, height=280)

        # Buttons for preprocessing actions
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("⚙️ Preprocess Data", key="preprocess_button", type="primary"):
                try:
                    with st.spinner(tr("Preprocessing data...")):
                        # Apply preprocessing based on selected options
                        preprocessed_data = st.session_state.data.copy()
                        missing_before = preprocessed_data.isna().sum().sum()
                        qc_removed = 0
                        outlier_removed = 0

                        # Apply QC flags if selected. QC column naming isn't consistent across
                        # sources (FLUXNET suffixes "_QC", AmeriFlux/EddyPro inserts
                        # "_SSITC_TEST", ICOS raw exports prefix "qc_", the bundled example
                        # suffixes "_qc") -- match_qc_to_flux_column() handles all of them.
                        if apply_qc_flags and qc_columns:
                            for qc_col, threshold in qc_thresholds.items():
                                matched_flux_col = match_qc_to_flux_column(qc_col, actual_flux_cols_out, flux_vars)
                                if matched_flux_col:
                                    # Mark as NaN where QC flag exceeds the per-column threshold
                                    mask = preprocessed_data[qc_col] > threshold
                                    qc_removed += int((mask & preprocessed_data[matched_flux_col].notna()).sum())
                                    flux_col = matched_flux_col
                                    preprocessed_data.loc[mask, flux_col] = np.nan

                        # Handle outliers if selected
                        if remove_outliers:
                            for flux_col in actual_flux_cols_out:
                                series = preprocessed_data[flux_col]
                                if outlier_method == "IQR":
                                    Q1 = series.quantile(0.25)
                                    Q3 = series.quantile(0.75)
                                    IQR = Q3 - Q1
                                    mask = (series < (Q1 - iqr_factor * IQR)) | (series > (Q3 + iqr_factor * IQR))
                                elif outlier_method == "Z-score":
                                    mean = series.mean()
                                    std = series.std()
                                    mask = ((series - mean) / std).abs() > z_threshold
                                elif outlier_method == "Modified Z-score":
                                    median = series.median()
                                    mad = np.median(np.abs(series - median)) * 1.4826
                                    mask = ((series - median) / mad).abs() > mod_z_threshold if mad else pd.Series(False, index=series.index)
                                outlier_removed += int((mask & series.notna()).sum())
                                preprocessed_data.loc[mask, flux_col] = np.nan

                        missing_after = preprocessed_data.isna().sum().sum()

                        # Update app state
                        st.session_state.data = preprocessed_data
                        st.session_state.preprocess_diag = {
                            "missing_before": int(missing_before),
                            "missing_after": int(missing_after),
                            "qc_removed": qc_removed,
                            "outlier_removed": outlier_removed,
                        }
                        st.success("Data preprocessing completed!")

                except Exception as e:
                    st.error(tr("Error during preprocessing: {error}").format(error=str(e)))

        with col2:
            if st.button("Reset to Original Data", key="reset_button"):
                try:
                    if st.session_state.original_data is not None:
                        st.session_state.data = st.session_state.original_data.copy()
                        st.session_state.pop("preprocess_diag", None)
                        st.success("Data reset to original!")
                    else:
                        st.warning("No original dataset found. Please upload or load data first.")
                except Exception as e:
                    st.error(tr("Error resetting data: {error}").format(error=str(e)))

        if st.session_state.get("preprocess_diag"):
            diag = st.session_state.preprocess_diag
            total_cells = st.session_state.data.shape[0] * st.session_state.data.shape[1]
            pct_before = diag["missing_before"] / total_cells * 100 if total_cells else 0
            pct_after = diag["missing_after"] / total_cells * 100 if total_cells else 0
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Values removed by QC filter", f"{diag['qc_removed']:,}")
            d2.metric("Values removed by outlier detection", f"{diag['outlier_removed']:,}")
            d3.metric("Missing cells before", f"{diag['missing_before']:,}")
            d4.metric("Missing cells after", f"{diag['missing_after']:,}", delta=f"{pct_after - pct_before:+.1f} pp", delta_color="inverse")
            style_metric_cards()

        # Show preprocessing results
        if st.session_state.data is not None:
            st.subheader("Gap Analysis")

            # Data resolution, so gap lengths can be interpreted in real time (not just timesteps)
            if isinstance(st.session_state.data.index, pd.DatetimeIndex) and len(st.session_state.data.index) > 1:
                resolution_min = st.session_state.data.index.to_series().diff().median().total_seconds() / 60
            else:
                resolution_min = None
            steps_per_hour = 60 / resolution_min if resolution_min else None

            # Create tabs for gap analysis views
            gap_tabs = st.tabs([tr("Missing by Variable"), tr("Gap Length Distribution")])

            with gap_tabs[0]:
                missing_counts = st.session_state.data[actual_flux_cols_out].isna().sum()
                missing_pct = (missing_counts / len(st.session_state.data) * 100).sort_values(ascending=False)
                top_missing = missing_pct[missing_pct > 0].head(15)
                if not top_missing.empty:
                    fig = px.bar(
                        x=top_missing.values, y=top_missing.index, orientation='h',
                        labels={'x': tr('Missing (%)'), 'y': ''},
                        title=tr("Variables with the Most Missing Data"),
                        text=[f"{v:.1f}%" for v in top_missing.values],
                        color_discrete_sequence=[colors["primary"]],
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_yaxes(autorange="reversed")
                    style_fig(fig, unified_hover=False, height=max(300, 28 * len(top_missing)))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values remain in the analyzed variables.")

            with gap_tabs[1]:
                # Calculate gap lengths
                if actual_flux_cols_out:
                    selected_var = st.selectbox(tr("Select variable for gap length analysis:"), actual_flux_cols_out)

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

                        # Categorize gaps -- short gaps are good candidates for simple
                        # interpolation/MDV, long gaps need the ML models this app trains.
                        if steps_per_hour:
                            short_max = 2 * steps_per_hour       # <= 2h
                            medium_max = 24 * steps_per_hour     # <= 1 day
                            n_short = int((gap_df['length'] <= short_max).sum())
                            n_medium = int(((gap_df['length'] > short_max) & (gap_df['length'] <= medium_max)).sum())
                            n_long = int((gap_df['length'] > medium_max).sum())

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Short gaps (≤2h)", n_short, help="Well suited to interpolation or MDV methods.")
                            c2.metric("Medium gaps (2h–1 day)", n_medium, help="Benefit from ML models using meteorological drivers.")
                            c3.metric("Long gaps (>1 day)", n_long, help="Hardest to fill reliably; rely on the time-only fallback model or external data.")
                            style_metric_cards()

                        fig = px.histogram(
                            gap_df,
                            x='length',
                            title=tr("Gap Length Distribution for {var}").format(var=selected_var),
                            labels={'length': tr('Gap Length (timesteps)'), 'count': tr('Frequency')},
                            color_discrete_sequence=[colors["primary"]],
                        )
                        fig.add_vline(
                            x=np.mean(gap_lengths), line_dash="dash", line_color=colors["highlight"],
                            annotation_text=tr("mean = {value}").format(value=f"{np.mean(gap_lengths):.1f}"), annotation_position="top",
                        )
                        # Log scale only pays off with enough gaps to show a heavy tail;
                        # with a handful of gaps it just produces confusing fractional ticks.
                        if len(gap_lengths) >= 20:
                            fig.update_yaxes(type="log", title=tr("Frequency (log scale)"))
                        style_fig(fig, unified_hover=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Summary statistics
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total gaps", len(gap_lengths))
                        s2.metric("Mean length", f"{np.mean(gap_lengths):.1f} steps")
                        s3.metric("Median length", f"{np.median(gap_lengths):.1f} steps")
                        s4.metric("Max length", f"{np.max(gap_lengths)} steps")
                        style_metric_cards()

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

        selected_features = []
        include_engineered_features = True
        time_based_features = [
            'hour', 'day', 'month', 'year', 'dayofyear', 'weekday', 'is_weekend',
            'hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos', 'month_sin',
            'month_cos', 'hour_decimal', 'season', 'season_0', 'season_1',
            'season_2', 'season_3', 'is_morning', 'is_afternoon', 'is_evening',
            'is_night'
        ]

        # Select features automatically (exclude targets, QC, datetime-related)
        if target_vars:
            exclude_patterns = ['date', 'time', '_qc', 'flag', 'datetime', 'month'] + target_vars
            potential_features = [
                col for col in st.session_state.data.columns
                if not any(p in col for p in exclude_patterns)
                and pd.api.types.is_numeric_dtype(st.session_state.data[col])
            ]
            
            # Final feature list (excluding targets)
            auto_selected_features = [col for col in potential_features if col not in target_vars]
            # Informative display
            st.markdown(tr("🔎 **Default predictors:** `{n}` variables available for modeling.").format(n=len(auto_selected_features)))
            selected_features = st.multiselect(
                "Select predictor variables:",
                options=potential_features,
                default=auto_selected_features,
                help="Choose the measured variables used by the full-feature model. Time-only fallback predictors are generated automatically."
            )
            include_engineered_features = st.checkbox(
                "Include engineered time, lag, rolling, and meteorological predictors",
                value=True,
                help="Adds derived predictors after upload. Rolling target statistics use only previous values to avoid target leakage."
            )
            with st.expander(tr("📋 See selected measured predictors")):
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
            st.markdown(tr("🔧 **Set Model Hyperparameters**"))

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
                learning_rate = st.select_slider(tr("Learning Rate (learning_rate):"), options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3],value=0.01)
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

        st.subheader("Validation Settings")
        val_col1, val_col2 = st.columns([1, 1])
        with val_col1:
            validation_strategy = st.selectbox(
                "Train/test validation:",
                ["Blocked time-series CV", "Chronological holdout", "Random holdout", "None"],
                index=0,
                help="Blocked time-series CV is recommended for time series because it tests several future blocks.",
                format_func=tr
            )
        with val_col2:
            validation_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                disabled=validation_strategy == "None"
            ) / 100.0
        n_splits = 5
        if validation_strategy == "Blocked time-series CV":
            n_splits = st.slider(
                "Number of validation blocks:",
                min_value=2,
                max_value=8,
                value=5,
                help="Each block trains on earlier data and tests on the next chronological block."
            )
     # Train model button
        if st.button("Train Model", disabled=not target_vars):
            try:
                with st.spinner(tr("🔄 Preparing data and training models...")):
                    processed_data = st.session_state.data.copy()
                    # Ensure 'datetime' is a column
                    if isinstance(processed_data.index, pd.DatetimeIndex):
                        processed_data['datetime'] = processed_data.index
                    elif 'date' in processed_data.columns and 'time (UTC)' in processed_data.columns:
                        processed_data['datetime'] = pd.to_datetime(processed_data['date'] + ' ' + processed_data['time (UTC)'], format='%d.%m.%Y %H:%M:%S')
                    else:
                        st.error(tr("❌ Could not infer or create 'datetime' column. Please ensure your data includes datetime information."))
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

                    # Extend selected_features with engineered predictors when enabled.
                    if include_engineered_features:
                        selected_features = list(dict.fromkeys(selected_features + added_features))
                    selected_features = [col for col in selected_features if pd.api.types.is_numeric_dtype(processed_data[col])]
                    time_based_features = [col for col in time_based_features if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col])]
                    if not selected_features:
                        st.error("❌ Please select at least one numeric predictor for the full-feature model.")
                        st.stop()
                    
                    # Progress bar
                    progress = st.progress(0)
                    total = len(target_vars)

                    for idx, target_col in enumerate(target_vars):
                        st.info(tr("📈 Training models for target: `{target_col}`").format(target_col=target_col))

                        model_all, model_time, validation_metrics = train_model(
                            data=processed_data,
                            target_col=target_col,
                            selected_features=selected_features,
                            time_based_features=[f for f in time_based_features if f in processed_data.columns],
                            model_type=model_type,
                            base_params=model_params,
                            validation_strategy=validation_strategy,
                            test_size=validation_size,
                            n_splits=n_splits
                        )

                        # Save models
                        model_key_all = f"{model_type}_ALL_{target_col}"
                        model_key_time = f"{model_type}_TIME_{target_col}"

                        st.session_state.models[model_key_all] = {
                            'model': model_all,
                            # Features actually used to fit the model, not the raw
                            # selection -- predictors that were entirely empty (see
                            # dropped_empty_features) were excluded during training,
                            # and gap-filling must check availability against the
                            # same list the model expects.
                            'features': validation_metrics['all_features'].get('used_features', selected_features),
                            'feature_set': 'all',
                            'validation': validation_metrics['all_features'],
                            'metadata': {
                                'target': target_col,
                                'model_type': model_type,
                                'validation_strategy': validation_strategy,
                                'test_size': validation_size,
                                'n_splits': n_splits,
                                'parameters': model_params,
                                'include_engineered_features': include_engineered_features,
                                'n_features': len(selected_features),
                                'n_rows': len(processed_data)
                            }
                        }

                        st.session_state.models[model_key_time] = {
                            'model': model_time,
                            'features': validation_metrics['time_based'].get(
                                'used_features', [f for f in time_based_features if f in processed_data.columns]
                            ),
                            'feature_set': 'time_based',
                            'validation': validation_metrics['time_based'],
                            'metadata': {
                                'target': target_col,
                                'model_type': model_type,
                                'validation_strategy': validation_strategy,
                                'test_size': validation_size,
                                'n_splits': n_splits,
                                'parameters': model_params,
                                'include_engineered_features': True,
                                'n_features': len([f for f in time_based_features if f in processed_data.columns]),
                                'n_rows': len(processed_data)
                            }
                        }

                        st.success(tr("✅ Models for `{target_col}` trained and stored!").format(target_col=target_col))
                        dropped_empty = validation_metrics['all_features'].get('dropped_empty_features', [])
                        if dropped_empty:
                            st.info(tr("ℹ️ {n} predictor(s) had no data at all and were excluded automatically: {cols}").format(
                                n=len(dropped_empty), cols=", ".join(f"`{c}`" for c in dropped_empty)
                            ))
                        if validation_strategy != "None":
                            metrics_df = pd.DataFrame(validation_metrics).T
                            display_cols = [
                                'strategy', 'n_train', 'n_test', 'n_folds',
                                'r2', 'rmse', 'mae', 'bias', 'slope', 'residual_std'
                            ]
                            st.dataframe(
                                metrics_df[[col for col in display_cols if col in metrics_df.columns]],
                                use_container_width=True
                            )
                        importance_rows = validation_metrics['all_features'].get('feature_importance', [])
                        if importance_rows:
                            with st.expander(tr("Top predictors for `{target_col}`").format(target_col=target_col)):
                                importance_df = pd.DataFrame(importance_rows)
                                fig_importance = px.bar(
                                    importance_df.sort_values('importance', ascending=True),
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title=tr("Feature Importance: {target_col}").format(target_col=target_col)
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                        progress.progress((idx + 1) / total)

                    st.success("🎉 All models trained successfully!")

            except Exception as e:
                st.error(tr("❌ Error during model training: {error}").format(error=str(e)))

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
            st.warning(tr("No trained models found for target variable `{target}`. Please train models first.").format(target=selected_target))
            st.stop()

        model_all = st.session_state.models[model_all_key[0]]['model']
        model_time = st.session_state.models[model_time_key[0]]['model']
        all_features = st.session_state.models[model_all_key[0]]['features']
        time_features = st.session_state.models[model_time_key[0]]['features']

        if st.button("🚀 Start Gap-Filling"):
            try:
                with st.spinner(tr("Applying trained models to fill gaps...")):
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
                    all_features_present_mask = df_gapfilled.loc[missing_mask, all_features].notna().all(axis=1)

                    predicted_values = pd.Series(
                        model_time.predict(X_missing_time_based),
                        index=df_gapfilled.index[missing_mask]
                    )
                    # Track which model produced each estimate -- the full-feature model is
                    # trained on meteorological drivers and is more reliable; the time-only
                    # fallback only sees cyclical hour/month encodings, so users should be
                    # able to tell the two apart when judging trust in a filled value.
                    df_gapfilled['fill_method'] = ''
                    df_gapfilled.loc[missing_mask, 'fill_method'] = 'time_only_fallback'
                    if all_features_present_mask.any():
                        all_feature_rows = all_features_present_mask[all_features_present_mask].index
                        predicted_values.loc[all_feature_rows] = model_all.predict(
                            df_gapfilled.loc[all_feature_rows, all_features]
                        )
                        df_gapfilled.loc[all_feature_rows, 'fill_method'] = 'full_model'

                    # Assign predicted values based on feature availability
                    df_gapfilled.loc[missing_mask, selected_target] = predicted_values
                    df_gapfilled.loc[missing_mask, 'filled'] = 1  # Mark filled rows as 1
                    st.session_state.filled_data = df_gapfilled
                    st.session_state.filled_target = selected_target
                    st.success("✅ Gap-filling completed!")
                    
            except Exception as e:
                st.error(tr("❌ Error during gap-filling: {error}").format(error=str(e)))

        # Show visualization and summary
        if st.session_state.filled_data is not None:
            df_full = st.session_state.filled_data
            total_filled_in_data = df_full['filled'].sum()
            total_rows = len(df_full)
            percent_filled = (total_filled_in_data / total_rows) * 100

            st.subheader("📊 Gap-Filling Summary")
            fill_counts = df_full.loc[df_full['filled'] == 1, 'fill_method'].value_counts()
            n_full = int(fill_counts.get('full_model', 0))
            n_fallback = int(fill_counts.get('time_only_fallback', 0))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total gaps filled", f"{int(total_filled_in_data):,}")
            m2.metric("Percentage filled", f"{percent_filled:.1f}%")
            m3.metric("Filled by full model", f"{n_full:,}", help="Used all meteorological features -- higher-confidence estimate.")
            m4.metric("Filled by time-only fallback", f"{n_fallback:,}", help="Meteorological drivers were missing too; estimate relies on cyclical time features only -- lower confidence.")
            style_metric_cards()

            st.subheader("📈 Visualize Original vs Filled")
            plot_col = st.selectbox("Select a variable to visualize:", [selected_target] + st.session_state.data.columns.tolist())

            df_original = st.session_state.data.copy()
            df_filled = st.session_state.filled_data.copy()

            # Split filled points by which model produced them, so users can see at a
            # glance which estimates are higher- vs lower-confidence.
            full_model_only = df_filled[plot_col].where(
                (df_filled['filled'] == 1) & (df_filled['fill_method'] == 'full_model'), np.nan
            )
            fallback_only = df_filled[plot_col].where(
                (df_filled['filled'] == 1) & (df_filled['fill_method'] == 'time_only_fallback'), np.nan
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_original.index,
                y=df_original[plot_col],
                mode='lines',
                name=tr('Original (with gaps)'),
                line=dict(color=colors["primary"], width=1.5),
                connectgaps=False
            ))
            fig.add_trace(go.Scatter(
                x=df_filled.index,
                y=full_model_only,
                mode='markers',
                name=tr('Gap-filled (full model)'),
                marker=dict(size=6, color=colors["highlight"], symbol='circle')
            ))
            fig.add_trace(go.Scatter(
                x=df_filled.index,
                y=fallback_only,
                mode='markers',
                name=tr('Gap-filled (time-only fallback)'),
                marker=dict(size=6, color="#E67E22", symbol='diamond')
            ))

            fig.update_layout(
                title=tr("Gap-Filling Visualization for: {var}").format(var=plot_col),
                yaxis_title=plot_col,
            )
            style_fig(fig, rangeslider=True, unified_hover=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(tr("🔍 View filled data table (first 50 rows)")):
                st.caption("Rows highlighted in **yellow** were estimated by the model ('filled' = 1).")
                df_preview = st.session_state.filled_data.head(50)
                def highlight_filled(row):
                    return ['background-color: yellow' if row['filled'] == 1 else '' for _ in row.index]
                st.dataframe(
                    df_preview.style.apply(highlight_filled, axis=1),
                    use_container_width=True,
                    hide_index=False
                )

            # Download option
            st.subheader("💾 Download Gap-Filled Data")
            df_export= st.session_state.filled_data.copy()
            cols_to_keep=["datetime", selected_target, 'filled', 'fill_method']
            final_cols = [c for c in cols_to_keep if c in df_export.columns]
            df_export = df_export[final_cols]
            st.caption("The downloaded file will contain only: **datetime**, **target variable**, **filled** status (0=original, 1=gap-filled), and **fill_method** (which model produced each estimate).")
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
    if missing_mechanism == "MAR" and st.session_state.data is not None:
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
            st.warning(tr("No trained models found for target variable `{target}`. Please train models first.").format(target=selected_target))
            st.stop()

        model_all = st.session_state.models[model_all_key[0]]['model']
        model_time = st.session_state.models[model_time_key[0]]['model']
        all_features = st.session_state.models[model_all_key[0]]['features']
        time_features = st.session_state.models[model_time_key[0]]['features']
        
        if st.button("🚀 Run Evaluation Test"):
            try:
                with st.spinner(tr("Generating artifical gaps and applying trained models to fill them...")):
                    df = st.session_state.data.copy()
                    df_gapfilled = df.copy()
                    df_gapfilled['filled'] = 0
                    # Only the target needs to be complete here -- introduce_nan()
                    # injects artificial gaps into it and needs genuine ground truth
                    # to compare against. Requiring every other column to be
                    # non-null too (as before) drops the entire dataset on any
                    # real file with typical per-sensor missingness spread across
                    # dozens of columns.
                    df_gapfilled.dropna(subset=[selected_target], inplace=True)
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
                    all_features_present_mask = df_gapfilled_with_na.loc[missing_mask, all_features].notna().all(axis=1)

                    predicted_values = pd.Series(
                        model_time.predict(X_missing_time_based),
                        index=df_gapfilled_with_na.index[missing_mask]
                    )
                    if all_features_present_mask.any():
                        all_feature_rows = all_features_present_mask[all_features_present_mask].index
                        predicted_values.loc[all_feature_rows] = model_all.predict(
                            df_gapfilled_with_na.loc[all_feature_rows, all_features]
                        )

                    # Assign predicted values based on feature availability
                    df_gapfilled_with_na.loc[missing_mask, selected_target] = predicted_values
                    df_gapfilled_with_na.loc[missing_mask, 'filled'] = 1  # Mark filled rows as 1
                
                    # Get the gap-filled values from 'df_gapfilled_eval' for the same rows
                    filled_values = df_gapfilled_with_na.loc[missing_mask, selected_target]

                    # Store the run's own context alongside its results -- widgets
                    # (target, mechanism, %) can change on a later rerun before the user
                    # looks at these numbers again, and they must not silently relabel.
                    st.session_state.eval_results = {
                        "original": original_data,
                        "filled": filled_values,
                        "target": selected_target,
                        "mechanism": missing_mechanism,
                        "nan_percentage": nan_percentage,
                        "dependency_feature": dependency_feature,
                    }
                    st.success("✅ Evaluation completed !")

            except Exception as e:
                st.error(tr("❌ Error during gap-filling: {error}").format(error=str(e)))

        # Results persist across reruns (e.g. tweaking another widget) instead of
        # vanishing once st.button's one-shot True reverts to False.
        if st.session_state.get("eval_results"):
            eval_results = st.session_state.eval_results
            y_true = eval_results["original"]
            y_pred = eval_results["filled"]
            eval_target = eval_results["target"]
            eval_mechanism = eval_results["mechanism"]
            eval_nan_percentage = eval_results["nan_percentage"]
            eval_dependency_feature = eval_results["dependency_feature"]
            st.caption(
                tr("Showing results for target **{target}**, **{mechanism}** mechanism, **{pct}** artificial gaps.").format(
                    target=eval_target, mechanism=eval_mechanism, pct=f"{eval_nan_percentage:.0%}"
                )
            )
            residuals = y_true - y_pred

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            bias = residuals.mean()
            if len(y_true) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
            else:
                slope = np.nan

            st.divider()
            st.subheader("📊 Performance Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("R² Score", f"{r2:.3f}", help="Closer to 1.0 is better fit")
            c2.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error - unit follows target variable")
            c3.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error - unit follows target variable")
            c4.metric("Bias (MBE)", f"{bias:+.3f}", help="Mean Bias Error: mean(observed - predicted). Should be close to 0 -- a nonzero value means the model systematically over- or under-estimates.")
            c5.metric("Slope", f"{slope:.3f}", help="Slope of observed-vs-predicted regression. Should be close to 1.0; <1 means the model compresses extremes (under-predicts highs, over-predicts lows).")
            style_metric_cards()

            st.subheader("Predicted vs Actual Scatter Plot")
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scattergl(
                x=y_true, y=y_pred, mode='markers',
                marker=dict(color=colors["primary"], opacity=0.5, size=5),
                name=tr('Test points')
                ))
            fig_scat.add_shape(type="line",
                    x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                    line=dict(color=colors["highlight"], width=2, dash="dash"),
                )
            fig_scat.update_layout(
                title=tr("Observed vs Predicted (R²={r2})").format(r2=f"{r2:.2f}"),
                xaxis_title=tr("Observed (Ground Truth)"),
                yaxis_title=tr("Predicted (Gap-Filled)"),
                )
            style_fig(fig_scat, unified_hover=False, height=480)
            st.plotly_chart(fig_scat, use_container_width=True)

            # Residual histogram: the secondary diagnostic -- reveals systematic
            # over/under-estimation (skew) and outliers the scatter plot can hide.
            st.subheader("Residuals Histogram")
            fig_resid = px.histogram(
                residuals,
                nbins=30,
                title=tr("Distribution of Residuals (Observed − Predicted)"),
                labels={"value": tr("Residual")},
                marginal="box",
                opacity=0.85,
                color_discrete_sequence=[colors["primary"]],
            )
            fig_resid.add_vline(x=0, line_dash="dash", line_color=colors["highlight"], annotation_text="0", row=2, col=1)
            style_fig(fig_resid, unified_hover=False)
            fig_resid.update_layout(bargap=0.1, showlegend=False)
            st.plotly_chart(fig_resid, use_container_width=True)

            # Downloadable report
            st.subheader("💾 Download Evaluation Report")
            df_report = pd.DataFrame({
                "datetime": y_true.index,
                "original": y_true.values,
                "filled": y_pred.values,
                "residual": residuals.values
            })
            df_report['mechanism'] = eval_mechanism
            df_report['nan_percentage'] = eval_nan_percentage
            df_report['target_variable'] = eval_target
            df_report['dependency_col'] = eval_dependency_feature if eval_mechanism == 'MAR' else "N/A"
            csv_data = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download (CSV)",
                data=csv_data,
                file_name=f"eval_report_{eval_target}_{eval_mechanism}.csv",
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
        filled_target = st.session_state.get("filled_target")
        if filled_target:
            st.caption(
                tr("ℹ️ Only **{target}** was gap-filled on the Gap-Filling tab. "
                   "Other variables shown below may still contain missing periods -- gap-filled "
                   "points for that variable are marked distinctly where shown.").format(target=filled_target)
            )
        plot_flux_partitioning(df_flux, filled_target=filled_target)

# About
elif st.session_state.active_tab == "About":
    # Page header
    colored_header(
        label="How it works",
        description="Understanding the logic behind the Gap-Filling Machine",
        color_name="green-70"
    )

    # Intro
    st.markdown(tr("""
    ### 🎯 Objective
    The goal of this platform is to transform your raw, gapped times series and Eddy Covariance data into a **continuous, high-quality time series**.
    To achieve this, we train machine learning models specifically on **your uploaded data**.
    """))
    st.divider()

    # Section 0: Supported data & end-to-end workflow
    st.subheader("📚 Supported Data & Workflow")

    col_fmt, col_flow = st.columns([1, 1])
    with col_fmt:
        st.markdown(tr("""
        **Supported input formats**
        * **FLUXNET** (hourly, half-hourly, or daily resolution)
        * **AmeriFlux**
        * **ICOS**
        * **Custom / general time series** -- any CSV with a recognizable timestamp column (`datetime`, `date`, `timestamp`, etc.)

        Format is auto-detected on upload. Both `-9999` sentinel values (the FLUXNET/AmeriFlux convention) and native `NaN` are treated as missing.
        """))
    with col_flow:
        st.markdown(tr("""
        **The 7-step workflow**
        1. **Upload & Explore** -- load data, inspect missingness and distributions
        2. **Data Preprocessing** -- QA/QC filtering, outlier removal, gap analysis
        3. **Model Training** -- train the full-feature and time-only models
        4. **Gap-Filling** -- apply the trained models to your real gaps
        5. **Gap-Fill Evaluation** -- test accuracy on artificial gaps before trusting the result
        6. **Advanced Flux Visualization** -- energy balance, carbon flux, Bowen ratio
        7. **About** -- you are here
        """))

    st.caption(tr("🔒 Data you upload is processed only within your current session and is not permanently stored by this app."))
    st.divider()

    # Section 1: The Power of Time
    st.subheader("1. 🕰️ The Temporal Features (feature engineering)")
    st.markdown(tr("""
    One of the most critical steps in our pipeline is extracting information from the timestamp.
    The model doesn't just see a date; we break it down into potential predictors that capture natural cycles:

    * **Diurnal Cycle (Hour of Day):** Helps the model understand that photosynthesis peaks at noon and respiration dominates at night.
    * **Seasonality (Month/Week):** Captures phenology, such as leaf-out in spring or senescence in autumn.

    > **Why is this important?** Even if all your meteorological sensors fail, **time never stops**. By learning these temporal patterns, the model can make a reasonable prediction based solely on "what usually happens at this hour in this month."
    """))

    add_vertical_space(1)

    # Section 2: The Smart Fallback Strategy
    st.subheader("2. 🧠 The 'Smart Fallback' Strategy")
    st.markdown(tr("""
    Real-world data is messy. Sometimes you have full meteorological data, sometimes you don't.
    To handle this, we train **two parallel models** and switch between them dynamically for every single gap:
    """))

    col1, col2 = st.columns(2)

    with col1:
        st.info(tr("""
        **🦄 Model A: The Precision Expert**

        * **Inputs:** Solar Radiation, Temperature, **AND** Time Features.
        * **When it's used:** Whenever meteorological data is available.
        * **Accuracy:** ⭐⭐⭐⭐⭐ (Highest)
        """))

    with col2:
        st.warning(tr("""
        **⏰ Model B: The Reliable Backup**

        * **Inputs:** **ONLY** Time Features (Hour, Month, etc.).
        * **When it's used:** When meteorological sensors failed (NaNs).
        * **Accuracy:** ⭐⭐⭐ (Good baseline)
        """))

    st.markdown(tr("""
    ### 🔄 The Workflow
    1.  **You upload** your data (with gaps).
    2.  The system **trains both models** on the clean parts of your data.
    3.  It scans for gaps. For each gap, it asks: *"Do I have meteo data?"*
        * **Yes?** -> Use Model A (Precision).
        * **No?** -> Use Model B (Reliability).
    4.  The result is a gap-filled dataset that maximizes accuracy without leaving any holes.
    """))

    st.divider()

    # Section 3: Metrics glossary -- the same terms used on the Gap-Fill Evaluation
    # and Gap-Filling tabs, explained once so users don't have to guess.
    st.subheader("3. 📊 Understanding Your Results")
    st.markdown(tr("Terms you'll see on the **Gap-Fill Evaluation** and **Gap-Filling** tabs:"))

    metric_col, fill_col = st.columns(2)
    with metric_col:
        st.markdown(tr("""
        **Accuracy metrics**
        * **R² Score** -- variance explained; closer to 1.0 is better.
        * **RMSE** -- root mean squared error, in the target's own units; penalizes large errors more heavily.
        * **MAE** -- mean absolute error, in the target's own units; easier to interpret than RMSE.
        * **Bias (MBE)** -- mean(observed − predicted); should be near 0. A large positive/negative value means the model systematically over- or under-estimates.
        * **Slope** -- slope of observed-vs-predicted regression; should be near 1.0. Below 1 means the model compresses extremes (under-predicts highs, over-predicts lows).
        """))
    with fill_col:
        st.markdown(tr("""
        **Fill confidence**
        * **Filled by full model** -- all meteorological predictors were available; the higher-confidence estimate (Model A).
        * **Filled by time-only fallback** -- meteorological drivers were missing too, so only cyclical time features were used (Model B); treat these with more caution, especially over long gaps.
        * **Short / medium / long gaps** -- classified by duration (≤2h / 2h–1 day / >1 day) on the Data Preprocessing tab. Short gaps are the easiest to fill reliably; long gaps carry the most uncertainty regardless of model.
        """))

    add_vertical_space(1)

    # Section 4: Limitations & best practices
    st.subheader("4. ⚠️ Limitations & Best Practices")
    st.markdown(tr("""
    * **Always check the Evaluation tab before trusting production fills.** Run it with a missingness mechanism (MCAR/MAR/MNAR) that matches how gaps actually occur in your data -- results can differ substantially between mechanisms.
    * **Long gaps are inherently less reliable**, since the model has fewer nearby observations to anchor its estimate. Treat long-gap fills as a reasonable approximation, not ground truth.
    * **The full-feature model can only be trusted as far as its own inputs are** -- if a meteorological predictor is itself gap-filled or unreliable, downstream fills inherit that uncertainty.
    * **Advanced Flux Visualization only reflects the one target you gap-filled.** Other variables shown there may still contain real gaps, flagged with a missingness caption on each tab.
    """))

    st.divider()

    # Section 5: Technical Details (collapsed for clarity)
    with st.expander(tr("🛠️ Under the Hood (Technical Details)")):
        st.markdown(tr("""
        * **Algorithms:** We currently support **XGBoost** and **Random Forest**. These are ensemble tree-based methods excellent at capturing non-linear relationships in ecological data.
        * **Categorical Encoding:** Features like `wind_direction` or `stability_class` are automatically one-hot encoded (converted to binary numbers) so the math works.
        * **Validation:** We use internal testing (RMSE, R²) to check model health, but we highly recommend using the **Evaluation Tab** to simulate artificial gaps and see how the model performs on *your* specific dataset structure.
        """))
#Footer
if st.session_state.active_tab != "Home":
    st.markdown(
        f"""
        <div class='footer'>
            <p>© 2025 Max Anjos • Eddy System | Version 1.0</p>
            <p>
                <a href="https://github.com/ByMaxAnjos/eddy-covariance-gap-filling-system/issues" target="_blank">
                    {tr("Have Feedback or Suggestions? Open an Issue on GitHub.")}
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
