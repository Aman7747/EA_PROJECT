import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# --- 1. Helper Functions (Unchanged) ---
def load_and_process_global_data(confirmed_path):
    df = pd.read_csv(confirmed_path)
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
    value_vars = [col for col in df.columns if col not in id_vars]
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Date', value_name='Confirmed')
    df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%m/%d/%y')
    country_agg = df_melted.groupby(['Country/Region', 'Date']).agg(Confirmed=('Confirmed', 'sum'), Lat=('Lat', 'mean'), Long=('Long', 'mean')).reset_index()
    ts_df = country_agg.pivot(index='Country/Region', columns='Date', values='Confirmed').fillna(0)
    country_metadata = country_agg.groupby('Country/Region').agg(Lat=('Lat', 'first'), Long=('Long', 'first')).to_dict('index')
    daily_confirmed = ts_df.diff(axis=1).fillna(0)
    daily_confirmed[daily_confirmed < 0] = 0
    daily_confirmed_smoothed = daily_confirmed.T.rolling(window=7, min_periods=1, center=True).mean().T.fillna(0)
    countries = sorted(list(country_metadata.keys()))
    return daily_confirmed_smoothed.reindex(countries), country_metadata

def create_global_graph_tensors(country_meta, dist_thresh=2000):
    countries = list(country_meta.keys())
    n = len(countries)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            c1_name, c2_name = countries[i], countries[j]
            c1, c2 = country_meta[c1_name], country_meta[c2_name]
            lat1, lon1, lat2, lon2 = map(np.radians, [c1['Lat'], c1['Long'], c2['Lat'], c2['Long']])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            dist = 2 * np.arcsin(np.sqrt(a)) * 6371
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    adj = np.where((dist_matrix < dist_thresh) & (dist_matrix > 0), 1, 0)
    return torch.tensor(np.array(np.where(adj)), dtype=torch.long)

def create_sequences(data, input_seq_len, output_seq_len):
    xs, ys = [], []
    for i in range(data.shape[1] - input_seq_len - output_seq_len + 1):
        x = data[:, i:(i + input_seq_len), :]
        y = data[:, (i + input_seq_len):(i + input_seq_len + output_seq_len), 0]
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


# --- 2. IMPROVEMENT: New Model with Residual Connection ---
class STGCN_Residual(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout_rate=0.6):
        super(STGCN_Residual, self).__init__()
        # Increased dropout for more regularization
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.attention_weights = nn.Linear(hidden_channels, 1)
        # The output layer now predicts the residual (the change)
        self.fc_residual = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_seq, edge_index):
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        x_seq_dev = x_seq.device
        edge_index_dev = edge_index.to(x_seq_dev)

        # Spatio-Temporal block
        gcn_out_seq = []
        for t in range(seq_len):
            x_t_flat = x_seq[:, t, :, :].reshape(-1, num_features)
            gcn_out_t = self.relu(self.gcn(x_t_flat, edge_index_dev))
            gcn_out_seq.append(gcn_out_t.view(batch_size, num_nodes, -1))

        gcn_out_seq = torch.stack(gcn_out_seq, dim=1)
        gru_input = gcn_out_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        gru_out, _ = self.gru(gru_input)

        # Attention mechanism
        attn_scores = self.softmax(self.attention_weights(gru_out))
        context_vector = torch.sum(attn_scores * gru_out, dim=1)

        # Predict the residual (the change from the last known value)
        predicted_residual = self.fc_residual(context_vector).reshape(batch_size, num_nodes, -1)

        # --- The Residual Connection ---
        # Get the last known value from the input sequence (infection count is feature 0)
        last_known_value = x_seq[:, -1, :, 0].unsqueeze(-1)

        # Final prediction is the sum of the last value and the predicted change
        final_prediction = last_known_value + predicted_residual

        return final_prediction

if __name__ == '__main__':
    CONFIRMED_PATH = "/kaggle/input/covid-data/time_series_covid19_confirmed_global (1).csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading global data and creating features...")
    daily_infections_df, country_metadata = load_and_process_global_data(CONFIRMED_PATH)

    COUNTRIES = list(country_metadata.keys())
    NUM_COUNTRIES = len(COUNTRIES)

    daily_infections_df_log = np.log1p(daily_infections_df)

    # --- IMPROVEMENT: Smoothed Derivative Features ---
    first_derivative = daily_infections_df_log.diff(axis=1).fillna(0)
    first_derivative_smoothed = first_derivative.T.rolling(window=5, min_periods=1, center=True).mean().T

    second_derivative = first_derivative_smoothed.diff(axis=1).fillna(0)
    second_derivative_smoothed = second_derivative.T.rolling(window=5, min_periods=1, center=True).mean().T

    # Normalize smoothed derivatives
    first_derivative_norm = (first_derivative_smoothed - first_derivative_smoothed.mean()) / (first_derivative_smoothed.std() + 1e-8)
    second_derivative_norm = (second_derivative_smoothed - second_derivative_smoothed.mean()) / (second_derivative_smoothed.std() + 1e-8)

    # Transpose for stacking
    first_derivative_norm_T = first_derivative_norm.T.values
    second_derivative_norm_T = second_derivative_norm.T.values

    # --- Prepare all features for model input ---
    df_T = daily_infections_df_log.T
    infection_data = df_T.values
    month_data = np.sin(2 * np.pi * df_T.index.month / 12).values
    day_of_week_data = np.sin(2 * np.pi * df_T.index.dayofweek / 7).values

    infection_mean_per_country = infection_data.mean(axis=0)
    infection_std_per_country = infection_data.std(axis=0)
    infection_data_normalized = (infection_data - infection_mean_per_country) / (infection_std_per_country + 1e-8)

    month_data_normalized = (month_data - month_data.mean()) / month_data.std()
    day_data_normalized = (day_of_week_data - day_of_week_data.mean()) / day_of_week_data.std()

    all_features = np.stack([
        infection_data_normalized,
        np.tile(month_data_normalized[:, np.newaxis], (1, NUM_COUNTRIES)),
        np.tile(day_data_normalized[:, np.newaxis], (1, NUM_COUNTRIES)),
        first_derivative_norm_T,
        second_derivative_norm_T
    ], axis=2)

    edge_index = create_global_graph_tensors(country_metadata)

    INPUT_SEQ_LEN = 28
    OUTPUT_SEQ_LEN = 7

    X, y = create_sequences(all_features.transpose(1, 0, 2), INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    X = X.permute(0, 2, 1, 3)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = STGCN_Residual(
        num_nodes=NUM_COUNTRIES,
        in_channels=X.size(3),
        hidden_channels=128,
        out_channels=OUTPUT_SEQ_LEN
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4) # Slightly lower LR
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    print(f"Starting training on {device}...")
    epochs = 200
    batch_size = 32
    pbar = tqdm(range(epochs))

    X_train_dev, y_train_dev = X_train.to(device), y_train.to(device)
    X_test_dev, y_test_dev = X_test.to(device), y_test.to(device)
    edge_index_dev = edge_index.to(device)

    for epoch in pbar:
        model.train()
        total_train_loss = 0
        permutation = torch.randperm(X_train_dev.size(0))
        for i in range(0, X_train_dev.size(0), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_dev[indices], y_train_dev[indices]

            # The model now outputs the final prediction directly
            pred_normalized = model(batch_x, edge_index_dev)

            active_countries_mask = batch_y.std(dim=(0, 2)) > 1e-6
            if active_countries_mask.sum() > 0:
                batch_loss = criterion(pred_normalized[:, active_countries_mask, :], batch_y[:, active_countries_mask, :])
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / (len(X_train_dev) / batch_size) if total_train_loss > 0 else 0

        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_dev, edge_index_dev)
            val_active_mask = y_test_dev.std(dim=(0, 2)) > 1e-6
            val_loss = criterion(val_preds[:, val_active_mask, :], y_test_dev[:, val_active_mask, :])

        scheduler.step(val_loss)
        pbar.set_description(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")

    # --- Evaluation and Plotting ---
    print("Evaluating model and generating plots...")
    model.eval()
    with torch.no_grad():
        test_preds_normalized = model(X_test_dev, edge_index_dev).cpu().numpy()

    y_test_np = y_test.cpu().numpy()

    pred_day_1_normalized = test_preds_normalized[:, :, 0]
    truth_day_1_normalized = y_test_np[:, :, 0]

    # De-normalize using the saved mean and std
    pred_log = pred_day_1_normalized * (infection_std_per_country + 1e-8) + infection_mean_per_country
    truth_log = truth_day_1_normalized * (infection_std_per_country + 1e-8) + infection_mean_per_country

    # Reverse the log transform
    prediction_rescaled = np.expm1(pred_log)
    ground_truth_rescaled = np.expm1(truth_log)

    prediction_rescaled[prediction_rescaled < 0] = 0

    countries_to_plot = ['India', 'Tanzania']
    fig, axes = plt.subplots(len(countries_to_plot), 1, figsize=(16, 7 * len(countries_to_plot)), sharex=True)
    if len(countries_to_plot) == 1: axes = [axes]

    start_date_idx = train_size + INPUT_SEQ_LEN
    num_test_points = len(ground_truth_rescaled)
    plot_dates = daily_infections_df.columns[start_date_idx : start_date_idx + num_test_points]

    for i, country_name in enumerate(countries_to_plot):
        ax = axes[i]
        try:
            country_idx = COUNTRIES.index(country_name)
            gt = ground_truth_rescaled[:, country_idx]
            pred = prediction_rescaled[:, country_idx]
            mae = np.mean(np.abs(gt - pred))

            ax.plot(plot_dates, gt, label='Ground Truth', color='tab:blue', linewidth=2)
            ax.plot(plot_dates, pred, label='Prediction', linestyle='--', color='tab:orange', linewidth=2)
            ax.set_title(f"Forecast for {country_name} (MAE: {mae:.2f})", fontsize=16)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Daily New Infections", fontsize=12)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        except ValueError:
            print(f"--- WARNING: Country '{country_name}' not found. Skipping plot. ---")

    plt.xlabel("Date", fontsize=12)
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Global COVID-19 Forecast with Residual Attention Model", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()