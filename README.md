# EA_PROJECT
An Epidemiology-Aware Framework for Dynamic Epidemic Forecasting
# GNN-Based COVID-19 Forecasting Models

This repository contains three graph-based models designed to forecast COVID-19 infections by leveraging spatio-temporal patterns and mobility-driven dynamics. The models are built using PyTorch and PyTorch Geometric, with data sourced from Johns Hopkins University and mobility logs.

---

## 📁 Contents

- `sir_gnn_model.py`: GNN-informed SIR model using mobility data between countries.
- `stgcn_Global.py`: Spatio-Temporal GCN model for 188 countries with residual connections and attention.
- `stgcn_us.py`: U.S.-centric STGCN using Gaussian-weighted graphs based on inter-state distances.

---

## 🧠 Models Overview

### 1. GNN-Informed SIR Model (`sir_gnn_model.py`)
- **Scope**: 5 countries (e.g., Japan, China, Australia)
- **Graph**: Built from flight data using ICAO codes.
- **Output**: 121-day infection trajectory using an ODE-based SIR model.
- **Architecture**: Graph-based parameter estimation + differentiable SIR solver.
- **Loss**: Mean Squared Logarithmic Error (MSLE)

### 2. Global STGCN (`stgcn_Global.py`)
- **Scope**: 188 countries
- **Graph**: Distance-based adjacency (thresholded at 2000 km).
- **Input**: Infection trends + temporal encodings + smoothed derivatives.
- **Output**: 7-day forecast using residual prediction (delta from last known value).
- **Architecture**: GCN + GRU + Attention + Residual layers

### 3. U.S. STGCN (`stgcn_us.py`)
- **Scope**: 50 U.S. states
- **Graph**: Weighted using a Gaussian kernel over geographic distances (σ = 300).
- **Input**: Daily infections + temporal encodings (month, weekday).
- **Output**: 7-day forecast
- **Extra**: Neighbor influence analysis for interpretability

---

## 🗃️ Data Sources

- **COVID-19 Case Data**:
  - Global: `time_series_covid19_confirmed_global.csv`
  - U.S.: `time_series_covid19_confirmed_US.csv`
- **Mobility Data**:
  - ICAO flight data from OpenSky (used in `sir_gnn_model.py`)

Ensure data files are structured according to the scripts, or update paths as needed.

---

## 🔧 Setup Instructions

1. Install required packages:
   ```bash
   pip install torch torchvision torchaudio torch-geometric matplotlib seaborn pandas tqdm
