# Eddy Covariance Gap-Filling System 🌱

![Eddy Covariance Gap-Filling System](static/eddy_photo.png)

---

### 🚀 An interactive platform for processing, filling, and evaluating gaps in flux tower datasets using Machine Learning.

This app provides an end-to-end solution for working with **eddy covariance data**, including:

✅ Uploading and preprocessing raw flux tower data (FLUXNET, AmeriFlux, ICOS)  
✅ Training gap-filling models using advanced ML techniques (XGBoost, Random Forest)  
✅ Visualizing filled vs. original data interactively  
✅ Evaluating model performance (MAE, RMSE, R²)  
✅ Supporting external integrations (e.g., weather, traffic, satellite data)

---

### 🌍 Supported Datasets

| Dataset    | Description                                                                                           |
|------------|------------------------------------------------------------------------------------------------------|
| [FLUXNET](https://fluxnet.org/)   | Global network of micrometeorological tower sites measuring ecosystem fluxes.                    |
| [AmeriFlux](https://ameriflux.lbl.gov/) | North and South American flux data on ecosystem–atmosphere exchanges.                         |
| [ICOS](https://www.icos-cp.eu/)         | Integrated Carbon Observation System: harmonized GHG flux data across Europe.                   |

---

### 🔧 Features

- Modular machine learning architecture with fallback models
- Detailed energy balance, carbon flux, and Bowen ratio visualizations
- Integration-ready design for adding external predictors (weather, traffic, satellite data)
- Deployable via [Streamlit Cloud](https://streamlit.io/cloud)

---

### 🌐 Live App

👉 **Try it here:** [eddy-gap-filling.streamlit.app](https://eddy-gap-filling.streamlit.app/)

---

### 📦 Setup

```bash
git clone https://github.com/ByMaxAnjos/eddy-covariance-gap-filling-system.git
cd eddy-covariance-gap-filling-system
pip install -r requirements.txt
streamlit run app/eddy_app.py