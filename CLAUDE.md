# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eddy Covariance Gap-Filling System — a Streamlit web app for processing and gap-filling time series data from flux tower networks (FLUXNET, AmeriFlux, ICOS) using machine learning. Live deployment: https://eddy-gap-filling.streamlit.app/

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app/eddy_app.py

# No test suite or linter is configured
```

## Architecture

The app has two source files:

**`app/eddy_app.py`** — Streamlit UI with 8 navigation tabs (Home, Upload & Explore, Preprocessing, Model Training, Gap-Filling, Evaluation, Flux Visualization, About). All session state, user controls, and rendering logic lives here. Tab state flows linearly: data must be uploaded before preprocessing, models must be trained before gap-filling.

**`app/eddy_functions.py`** — Pure backend logic with no Streamlit imports. Key responsibilities:
- **Dataset detection**: `detect_and_preprocess_dataset()` auto-detects FLUXNET/ICOS/AmeriFlux/custom column naming conventions
- **Feature engineering**: cyclical time encoding (`hour_sin/cos`, `month_sin/cos`), lag features (1–168 periods), rolling statistics, VPD and potential ET derived from meteorological inputs
- **Dual-model training**: `train_model()` always trains *two* models — a full-feature model (all meteorological + derived inputs) and a time-only fallback model (pure temporal features). Gap-filling uses the full model when all features are available and falls back to the time-only model otherwise
- **Artificial gap injection**: `introduce_nan()` supports MCAR/MAR/MNAR mechanisms for evaluation

**`data/eddy_covariance_data.csv`** — Bundled example dataset (June 2018, 30-min resolution, 27 columns) used when users click "Load Example Data".

**`static/`** — CSS (`custom.css`) and image assets. The forest-green theme (`#1E5631`) is defined in both `custom.css` and `.streamlit/config.toml`.

## Key Design Patterns

- `st.session_state` is the primary data bus between tabs. Variables like `st.session_state['df']`, `st.session_state['model']`, `st.session_state['filled_df']` are set in earlier tabs and read in later ones.
- All plots use Plotly (interactive); Matplotlib/Seaborn are imported but used only for supplementary static charts.
- XGBoost and scikit-learn Random Forest are both supported; model selection is a user radio button. Training calls are identical for both via a shared `train_model()` interface.
- Missing value handling distinguishes between -9999 sentinel values (FLUXNET/AmeriFlux convention) and true `NaN` — both are converted to `NaN` on load.
