import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torchdiffeq
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


# --- DATA LOADING AND PREPROCESSING ---
def load_covid_data(start_date_str='1/1/20', end_date_str='5/1/20'):
    url = "/kaggle/input/covid-data/time_series_covid19_confirmed_global (1).csv"
    df = pd.read_csv(url)
    df = df.drop(columns=["Province/State", "Lat", "Long"], errors="ignore")
    df = df.groupby("Country/Region").sum()

    start_date = datetime.strptime(start_date_str, '%m/%d/%y')
    end_date = datetime.strptime(end_date_str, '%m/%d/%y')

    date_cols = []
    for col in df.columns:
        try:
            col_date = datetime.strptime(col, '%m/%d/%y')
            if start_date <= col_date <= end_date:
                date_cols.append(col)
        except ValueError:
            continue

    df = df[date_cols]
    return df


POPULATIONS = {
    'Japan': 126_500_000,
    'China': 1_400_000_000,
    'Indonesia': 273_500_000,
    'Australia': 25_500_000,
    'New Zealand': 5_000_000,
}


def load_mobility_data(countries, start_date_str='2020-01-01', end_date_str='2020-05-01'):
    flight_files = [
        "/kaggle/input/flight-data/flightlist_20200101_20200131.csv",
        "/kaggle/input/flight-data/flightlist_20200201_20200229.csv",
        "/kaggle/input/flight-data/flightlist_20200301_20200331.csv",
        "/kaggle/input/flight-data/flightlist_20200401_20200430.csv"
    ]
    flightlist_dfs = []
    for file_path in flight_files:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=["firstseen", "lastseen", "day"])
                flightlist_dfs.append(df)
            except Exception as e:
                print(f"Could not load or parse {file_path}: {e}")
        else:
            print(f"Flight data file not found: {file_path}.")

    if not flightlist_dfs:
        print("No flight data loaded. Using placeholder graph.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    try:
        flightlist = pd.concat(flightlist_dfs, ignore_index=True)
    except ValueError:
        print("Error concatenating flight data. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    flightlist = flightlist.drop(columns=["Unnamed: 0"], errors="ignore")

    try:
        if not pd.api.types.is_datetime64_any_dtype(flightlist['day']):
            flightlist['day'] = pd.to_datetime(flightlist['day'], errors='coerce')
            flightlist.dropna(subset=['day'], inplace=True)
        if flightlist.empty: raise ValueError("Flight list empty after date conversion.")

        if flightlist['day'].dt.tz is not None:
            start_date_dt = pd.to_datetime(start_date_str).tz_localize('UTC')
            end_date_dt = pd.to_datetime(end_date_str).tz_localize('UTC')
        else:
            start_date_dt = pd.to_datetime(start_date_str)
            end_date_dt = pd.to_datetime(end_date_str)
        flightlist = flightlist[(flightlist['day'] >= start_date_dt) & (flightlist['day'] <= end_date_dt)]
    except (AttributeError, KeyError, ValueError) as e:
        print(f"Error processing datetime in flight data: {e}. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    icao_codes_path = "/kaggle/input/flight-data/icao_codes.xlsx"
    if not os.path.exists(icao_codes_path):
        print(f"ICAO codes file not found: {icao_codes_path}. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)
    try:
        icao_codes = pd.read_excel(icao_codes_path, usecols=["ICAO", "Country"])
    except Exception as e:
        print(f"Error reading ICAO codes: {e}. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    if 'origin' not in flightlist.columns or 'destination' not in flightlist.columns:
        print("Origin/Destination columns missing. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    flightlist = pd.merge(flightlist, icao_codes, how="left", left_on="origin", right_on="ICAO").rename(
        columns={"Country": "country_origin"}).drop(columns=["ICAO"], errors='ignore')
    flightlist = pd.merge(flightlist, icao_codes, how="left", left_on="destination", right_on="ICAO").rename(
        columns={"Country": "country_destination"}).drop(columns=["ICAO"], errors='ignore')
    flightlist = flightlist[flightlist['country_origin'].notna() & flightlist['country_destination'].notna()]
    flightlist['international'] = (flightlist['country_origin'] != flightlist['country_destination'])
    flightlist = flightlist[flightlist['international'] & flightlist['country_origin'].isin(countries) & flightlist[
        'country_destination'].isin(countries)]

    if flightlist.empty:
        print("No relevant flights after filtering. Using placeholder.")
        adj_matrix = np.ones((len(countries), len(countries)));
        np.fill_diagonal(adj_matrix, 0);
        adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
        return pd.DataFrame(adj_matrix, index=countries, columns=countries)

    flight_counts = flightlist.groupby(['country_origin', 'country_destination']).size().unstack(fill_value=0).reindex(
        index=countries, columns=countries, fill_value=0)
    adj_matrix = flight_counts.values + flight_counts.values.T
    adj_matrix = adj_matrix / adj_matrix.max() if adj_matrix.max() > 0 else np.zeros_like(adj_matrix)
    return pd.DataFrame(adj_matrix, index=countries, columns=countries)


# --- MODEL DEFINITIONS (Time-Varying Beta) ---
class BatchedSIRFuncTimeBeta(nn.Module):
    def __init__(self):
        super(BatchedSIRFuncTimeBeta, self).__init__()

    def forward(self, t, y_batch, beta_initial_batch, beta_decay_rate_batch, gamma_batch, N_batch):
        S_b, I_b, R_b = y_batch[:, 0], y_batch[:, 1], y_batch[:, 2]
        S_b = torch.clamp(S_b, min=0.0);
        I_b = torch.clamp(I_b, min=0.0)
        N_b_clamped = torch.max(N_batch.to(S_b.device), torch.tensor(1.0, device=S_b.device))

        current_beta_batch = beta_initial_batch * torch.exp(-beta_decay_rate_batch * t)
        current_beta_batch = torch.clamp(current_beta_batch, min=1e-7, max=5.0)

        dS_dt_b = -current_beta_batch * S_b * I_b / N_b_clamped
        dI_dt_b = current_beta_batch * S_b * I_b / N_b_clamped - gamma_batch * I_b
        dR_dt_b = gamma_batch * I_b

        if torch.isnan(dS_dt_b).any() or torch.isnan(dI_dt_b).any() or torch.isnan(dR_dt_b).any():
            print(f"!!! NaN in BatchedSIRFuncTimeBeta: t={t.item()}")
            nan_mask = torch.isnan(dS_dt_b) | torch.isnan(dI_dt_b) | torch.isnan(dR_dt_b)
            if nan_mask.any():
                problem_indices = torch.where(nan_mask)[0]
                print(f"  Problem indices: {problem_indices}")
                # print(f"  S: {S_b[nan_mask]}, I: {I_b[nan_mask]}, CurrentBeta: {current_beta_batch[nan_mask]}, Gamma: {gamma_batch[nan_mask]}, N: {N_b_clamped[nan_mask]}")
            return torch.zeros_like(y_batch)

        return torch.stack([dS_dt_b, dI_dt_b, dR_dt_b], dim=1)


class ParameterPredictorGNNTimeBeta(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_gnn_features=0):  # num_gnn_features for actual data
        super(ParameterPredictorGNNTimeBeta, self).__init__()
        self.num_gnn_features = num_gnn_features
        # If using actual features, GCN input dim changes
        gcn_input_dim = in_feats + num_gnn_features if num_gnn_features > 0 else in_feats

        self.conv1 = GCNConv(gcn_input_dim, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats // 2)
        self.fc_beta_initial = nn.Linear(hidden_feats // 2, 1)
        self.fc_beta_decay = nn.Linear(hidden_feats // 2, 1)
        self.fc_gamma = nn.Linear(hidden_feats // 2, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_embeddings, edge_index, edge_weight=None, x_features=None):
        # x_embeddings are the learnable nn.Embedding outputs
        # x_features are optional actual data features
        if x_features is not None and self.num_gnn_features > 0:
            # Ensure x_features is on the same device
            x_combined = torch.cat([x_embeddings, x_features.to(x_embeddings.device)], dim=1)
        else:
            x_combined = x_embeddings

        x = self.conv1(x_combined, edge_index, edge_weight).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight).relu()

        beta_initial = torch.sigmoid(self.fc_beta_initial(x)).squeeze(-1) * 2.0  # Max initial beta 2.0
        beta_decay_rate = torch.sigmoid(self.fc_beta_decay(x)).squeeze(-1) * 0.1  # Decay rate range 0-0.1
        gamma = torch.sigmoid(self.fc_gamma(x)).squeeze(-1) * 0.3  # Gamma range 0-0.3

        beta_initial = torch.clamp(beta_initial, min=1e-4, max=2.0)
        beta_decay_rate = torch.clamp(beta_decay_rate, min=1e-6, max=0.1)
        gamma = torch.clamp(gamma, min=1e-4, max=0.3)
        return beta_initial, beta_decay_rate, gamma


class GNNInformedSIRTimeBeta(nn.Module):
    def __init__(self, num_countries, gnn_embedding_size, gnn_hidden_feats, populations_tensor, num_timesteps,
                 num_gnn_features=0):
        super(GNNInformedSIRTimeBeta, self).__init__()
        self.param_gnn = ParameterPredictorGNNTimeBeta(gnn_embedding_size, gnn_hidden_feats, num_gnn_features)
        self.sir_ode_time_beta = BatchedSIRFuncTimeBeta()

        self.populations = populations_tensor
        self.num_timesteps = num_timesteps
        self.integration_time = torch.linspace(0, num_timesteps - 1, num_timesteps)
        self.node_embeddings = nn.Embedding(num_countries, gnn_embedding_size)
        nn.init.xavier_uniform_(self.node_embeddings.weight.data, gain=0.5)
        self.num_gnn_features = num_gnn_features

    def forward(self, initial_conditions_S0_I0_R0, edge_index, edge_weight, actual_data_features=None):
        device = edge_index.device
        node_ids = torch.arange(initial_conditions_S0_I0_R0.size(0)).to(device)
        current_node_embeddings = self.node_embeddings(node_ids)

        beta_initial_pred, beta_decay_rate_pred, gamma_pred = self.param_gnn(
            current_node_embeddings, edge_index, edge_weight, x_features=actual_data_features
        )

        if torch.isnan(beta_initial_pred).any() or torch.isnan(beta_decay_rate_pred).any() or torch.isnan(
                gamma_pred).any():
            print("!!! NaN detected in parameters from GNN output !!!")
            return torch.zeros(initial_conditions_S0_I0_R0.size(0), self.num_timesteps, 3, device=device) + float('nan')

        y0_batch = initial_conditions_S0_I0_R0.float().to(device)
        populations_batch = self.populations.to(device)
        current_integration_time = self.integration_time.to(device)

        bad_N = populations_batch <= 0
        bad_S0I0 = (y0_batch[:, 0] < 0) | (y0_batch[:, 1] < 0) | (
                    (y0_batch[:, 0] + y0_batch[:, 1]) > populations_batch * 1.01)
        any_bad_initial_state = bad_N | bad_S0I0
        if any_bad_initial_state.any():
            print(f"Warning: Bad initial states for countries: {torch.where(any_bad_initial_state)[0]}")
            y0_batch[:, 1] = torch.clamp(y0_batch[:, 1], min=0.0, max=populations_batch)
            y0_batch[:, 2] = torch.clamp(y0_batch[:, 2], min=0.0, max=populations_batch - y0_batch[:, 1])
            y0_batch[:, 0] = populations_batch - y0_batch[:, 1] - y0_batch[:, 2]
            y0_batch = torch.clamp(y0_batch, min=0.0)

        try:
            solution_batch = torchdiffeq.odeint(
                lambda t, y: self.sir_ode_time_beta(t, y, beta_initial_pred, beta_decay_rate_pred, gamma_pred,
                                                    populations_batch),
                y0_batch, current_integration_time,
                method='dopri5',  # Switched to dopri5
                rtol=1e-5, atol=1e-6
            )
            solution_batch = solution_batch.permute(1, 0, 2)
        except Exception as e:
            print(f"Batched ODEINT (TimeBeta) Error: {e}")
            # Check if specific parameters caused the issue
            print(
                f"  Problematic parameters (mean): beta_init={beta_initial_pred.mean().item():.3f}, beta_decay={beta_decay_rate_pred.mean().item():.4f}, gamma={gamma_pred.mean().item():.3f}")
            return torch.zeros(initial_conditions_S0_I0_R0.size(0), self.num_timesteps, 3, device=device) + float('nan')
        return solution_batch


# Run Model
def run_sir_gnn_informed():
    torch.autograd.set_detect_anomaly(True)

    countries = ['Japan', 'China', 'Indonesia', 'Australia', 'New Zealand']
    start_date_str, end_date_str = '1/1/20', '5/1/20'

    confirmed_raw_df = load_covid_data(start_date_str, end_date_str)
    available_countries = [c for c in countries if c in confirmed_raw_df.index]
    if len(available_countries) != len(countries):
        print(f"Warning: Using subset of countries: {available_countries} from requested {countries}")
        countries = available_countries
    if not countries: print("No specified countries found in COVID data. Exiting."); return
    confirmed_raw_df = confirmed_raw_df.loc[countries]

    actual_infected_counts = torch.tensor(confirmed_raw_df.values, dtype=torch.float)
    num_countries, num_timesteps = actual_infected_counts.shape
    if num_countries == 0: print("Zero countries to model. Exiting."); return

    # --- Create GNN Features from early data ---
    num_initial_days_for_features = 7  # Use first 7 days
    gnn_data_features_list = []
    if num_timesteps >= num_initial_days_for_features:
        for i in range(num_countries):
            country_early_cases = actual_infected_counts[i, :num_initial_days_for_features].clone()
            # Log1p transform and normalize (simple per-country min-max)
            country_early_cases_log = torch.log1p(country_early_cases)
            min_val, max_val = country_early_cases_log.min(), country_early_cases_log.max()
            if (max_val - min_val) > 1e-6:
                normalized_features = (country_early_cases_log - min_val) / (max_val - min_val)
            else:  # If all values are same (e.g. all zeros)
                normalized_features = torch.zeros_like(country_early_cases_log)
            gnn_data_features_list.append(normalized_features)
        gnn_data_features = torch.stack(gnn_data_features_list)  # (num_countries, num_initial_days_for_features)
        num_actual_gnn_features = num_initial_days_for_features
    else:
        print(
            f"Warning: Not enough timesteps ({num_timesteps}) to extract {num_initial_days_for_features} days for GNN features. Using only embeddings.")
        gnn_data_features = None
        num_actual_gnn_features = 0

    populations_list = [POPULATIONS[c] for c in countries]
    populations_tensor = torch.tensor(populations_list, dtype=torch.float)

    initial_I0 = actual_infected_counts[:, 0] + 1.0
    initial_R0 = torch.zeros(num_countries, dtype=torch.float)
    initial_S0 = populations_tensor - initial_I0 - initial_R0
    initial_I0 = torch.min(initial_I0, populations_tensor);
    initial_I0 = torch.clamp(initial_I0, min=1.0)
    initial_S0 = populations_tensor - initial_I0 - initial_R0;
    initial_S0 = torch.clamp(initial_S0, min=0.0)
    initial_conditions_S0_I0_R0 = torch.stack([initial_S0, initial_I0, initial_R0], dim=1)

    mobility_df = load_mobility_data(countries, '2020-01-01', '2020-05-01')
    edge_index_list = [];
    edge_weight_list = []
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if c1 in mobility_df.index and c2 in mobility_df.columns:
                weight = mobility_df.loc[c1, c2]
                if weight > 1e-6 and i != j:
                    edge_index_list.append([i, j]);
                    edge_weight_list.append(weight)
    if not edge_index_list:
        print("Mobility graph empty. Using fully connected graph fallback.")
        for i in range(num_countries):
            for j in range(num_countries):
                if i != j: edge_index_list.append([i, j]); edge_weight_list.append(0.1)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    gnn_embedding_size = 16  # Size of the learnable part of node features
    gnn_hidden_feats = 64
    model = GNNInformedSIRTimeBeta(
        num_countries, gnn_embedding_size, gnn_hidden_feats,
        populations_tensor, num_timesteps, num_gnn_features=num_actual_gnn_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.5, min_lr=1e-7)

    def msle_loss(pred, target):
        pred_c = torch.clamp(pred, min=0.0);
        target_c = torch.clamp(target, min=0.0)
        return torch.mean((torch.log1p(pred_c) - torch.log1p(target_c)) ** 2)

    loss_fn = msle_loss

    epochs = 300  # Increased epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    print(f"Using device: {device}")
    model.to(device);
    initial_conditions_S0_I0_R0 = initial_conditions_S0_I0_R0.to(device)
    edge_index = edge_index.to(device);
    edge_weight = edge_weight.to(device)
    actual_infected_counts = actual_infected_counts.to(device)
    model.populations = model.populations.to(device)
    model.integration_time = model.integration_time.to(device)
    if gnn_data_features is not None:
        gnn_data_features = gnn_data_features.to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_sir_dynamics = model(initial_conditions_S0_I0_R0, edge_index, edge_weight,
                                  actual_data_features=gnn_data_features)

        current_loss_val = float('nan')

        if torch.isnan(pred_sir_dynamics).any():
            print(f"Epoch {epoch}: NaN detected in pred_sir_dynamics. Skipping update.")
            if epoch > 10 and np.isnan(current_loss_val): print("Persistent NaNs in model output, stopping."); break
            continue

        pred_infected_counts = pred_sir_dynamics[:, :, 1];
        pred_infected_counts = torch.clamp(pred_infected_counts, min=0.0)
        loss = loss_fn(pred_infected_counts, actual_infected_counts)
        current_loss_val = loss.item()

        if torch.isnan(loss):
            print(
                f"Epoch {epoch}: Loss is NaN. Pred range: [{pred_infected_counts.min().item():.2e}, {pred_infected_counts.max().item():.2e}]")
            if epoch > 10: print("Persistent NaN loss, stopping."); break
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr: print(f"Epoch {epoch}: LR reduced from {old_lr:.7f} to {new_lr:.7f}")

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
            with torch.no_grad():
                node_ids = torch.arange(num_countries).to(device)
                current_node_embeddings = model.node_embeddings(node_ids)
                b_init_check, b_decay_check, g_p_check = model.param_gnn(
                    current_node_embeddings, edge_index, edge_weight, x_features=gnn_data_features
                )
                if torch.isnan(b_init_check).any() or torch.isnan(b_decay_check).any() or torch.isnan(g_p_check).any():
                    print("  !!! Params from GNN are NaN during eval print !!!")
                else:
                    print(
                        f"  BetaInit mean: {b_init_check.mean().item():.3f} (range: [{b_init_check.min().item():.3f}, {b_init_check.max().item():.3f}])")
                    print(
                        f"  BetaDecay mean: {b_decay_check.mean().item():.4f} (range: [{b_decay_check.min().item():.4f}, {b_decay_check.max().item():.4f}])")
                    print(
                        f"  Gamma mean: {g_p_check.mean().item():.3f} (range: [{g_p_check.min().item():.3f}, {g_p_check.max().item():.3f}])")

    torch.autograd.set_detect_anomaly(False)

    # Evaluation & Plotting
    model.eval()
    with torch.no_grad():
        pred_sir_dynamics_eval = model(
            initial_conditions_S0_I0_R0, edge_index, edge_weight, actual_data_features=gnn_data_features
        )
        if torch.isnan(pred_sir_dynamics_eval).any():
            print("NaNs in final evaluation output. Plotting might be affected.")
            pred_infected_unscaled = np.full_like(actual_infected_counts.cpu().numpy(), np.nan)
        else:
            pred_infected_unscaled = pred_sir_dynamics_eval[:, :, 1].cpu().numpy()
            pred_infected_unscaled = np.maximum(pred_infected_unscaled, 0)

    actual_unscaled = actual_infected_counts.cpu().numpy()

    if np.isnan(pred_infected_unscaled).any():
        mse, mae = float('nan'), float('nan')
    else:
        mse = np.mean((pred_infected_unscaled - actual_unscaled) ** 2)
        mae = np.mean(np.abs(pred_infected_unscaled - actual_unscaled))
    print(f"\nFinal MSE: {mse:.2f}, MAE: {mae:.2f}")

    t = np.arange(num_timesteps)
    for i, country in enumerate(countries):
        plt.figure(figsize=(12, 7))
        plt.plot(t, actual_unscaled[i, :], 'b-', label=f"{country} (Actual)", linewidth=2)
        if np.isnan(pred_infected_unscaled[i, :]).any():
            plt.plot(t, np.zeros_like(t), 'r--', label=f"{country} (Predicted - NaN)", linewidth=2)
        else:
            plt.plot(t, pred_infected_unscaled[i, :], 'r--', label=f"{country} (Predicted)", linewidth=2)

        with torch.no_grad():
            node_ids = torch.arange(num_countries).to(device)
            current_node_embeddings = model.node_embeddings(node_ids)
            b_init, b_decay, g_p = model.param_gnn(
                current_node_embeddings, edge_index, edge_weight, x_features=gnn_data_features
            )
            b_init_val = f"{b_init[i].item():.3f}" if not torch.isnan(b_init[i]).any() else "NaN"
            b_decay_val = f"{b_decay[i].item():.4f}" if not torch.isnan(b_decay[i]).any() else "NaN"
            g_val = f"{g_p[i].item():.3f}" if not torch.isnan(g_p[i]).any() else "NaN"

        plt.title(
            f"Actual vs Predicted ({country}) | $\\beta_{{init}}={b_init_val}, \\beta_{{decay}}={b_decay_val}, \\gamma={g_val}, N={populations_list[i] / 1e6:.1f}M$")
        plt.xlabel(f"Time (Days since {start_date_str})");
        plt.ylabel("Confirmed Cases");
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        print(f"Displaying plot for {country}")
        plt.show();
        plt.close()

    # Plot learned Beta(t) and R_eff(t)
    print("\nPlotting learned dynamic parameters...")
    model.eval()
    with torch.no_grad():
        node_ids = torch.arange(num_countries).to(device)
        current_node_embeddings = model.node_embeddings(node_ids)
        b_init_final, b_decay_final, g_p_final = model.param_gnn(
            current_node_embeddings, edge_index, edge_weight, x_features=gnn_data_features
        )
        time_points_np = model.integration_time.cpu().numpy()

        for i, country in enumerate(countries):
            country_b_init = b_init_final[i].item()
            country_b_decay = b_decay_final[i].item()
            country_gamma = g_p_final[i].item()
            if torch.isnan(torch.tensor(country_b_init)) or torch.isnan(torch.tensor(country_b_decay)) or torch.isnan(
                    torch.tensor(country_gamma)):
                print(f"Skipping parameter plot for {country} due to NaN parameters.")
                continue

            beta_t_country = country_b_init * np.exp(-country_b_decay * time_points_np)
            r_eff_t_country = beta_t_country / country_gamma if country_gamma > 1e-7 else np.zeros_like(
                beta_t_country) + float('inf')

            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:red';
            ax1.set_xlabel('Time (Days)');
            ax1.set_ylabel('Beta(t)', color=color)
            ax1.plot(time_points_np, beta_t_country, color=color, linestyle='-', label=f'Beta(t) - {country}')
            ax1.tick_params(axis='y', labelcolor=color);
            ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
            ax2 = ax1.twinx();
            color = 'tab:blue';
            ax2.set_ylabel('R_eff(t)', color=color)
            ax2.plot(time_points_np, r_eff_t_country, color=color, linestyle='--', label=f'R_eff(t) - {country}')
            ax2.tick_params(axis='y', labelcolor=color);
            ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5)
            plt.title(
                f"Learned Parameters for {country}\n$\\beta_{{init}}={country_b_init:.3f}, \\beta_{{decay}}={country_b_decay:.4f}, \\gamma={country_gamma:.3f}$")
            fig.tight_layout();
            lines, labels = ax1.get_legend_handles_labels();
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right');
            plt.show();
            plt.close()

    if np.isnan(pred_infected_unscaled).any():
        print("Skipping mobility impact heatmap due to NaNs in predictions.")
    else:
        avg_predicted_infected = np.nanmean(pred_infected_unscaled, axis=1)
        if num_countries > 0 and isinstance(mobility_df, pd.DataFrame) and mobility_df.shape[0] == num_countries and \
                mobility_df.shape[1] == num_countries:
            mobility_df_reindexed = mobility_df.reindex(index=countries, columns=countries, fill_value=0)
            mobility_matrix_for_plot = mobility_df_reindexed.values
            impact_matrix_values = mobility_matrix_for_plot * avg_predicted_infected.reshape(-1, 1)
            impact_df = pd.DataFrame(impact_matrix_values, index=countries, columns=countries)
            plt.figure(figsize=(10, 8))
            sns.heatmap(impact_df, annot=True, cmap="Reds", fmt=".0f")
            plt.title("Estimated Mobility-Driven Infection Potential");
            plt.xlabel("Destination Country");
            plt.ylabel("Source Country (avg. cases)")
            plt.tight_layout();
            print("Displaying mobility impact heatmap.");
            plt.show();
            plt.close()

            total_incoming_impact = impact_df.sum(axis=0)
            if not total_incoming_impact.empty:
                most_impacted_country = total_incoming_impact.idxmax()
                max_impact_value = total_incoming_impact.max()
                print(
                    f"\nMost impacted country by potential infections: {most_impacted_country} (Total incoming impact score: {max_impact_value:.0f})")
                print("Breakdown of incoming impact:");
                print(total_incoming_impact.sort_values(ascending=False))
            else:
                print("Could not determine most impacted country (empty impact sums).")
        else:
            print("Skipping heatmap/impact calculation: mobility_df issues or country list mismatch.")


if __name__ == "__main__":
    run_sir_gnn_informed()