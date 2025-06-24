import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tqdm import tqdm

# --- 1. Constants and Data Loading ---
STATE_DATA = {
    'Alabama': {'pop': 5024279, 'lat': 32.806671, 'lon': -86.791130},
    'Arizona': {'pop': 7151502, 'lat': 34.168219, 'lon': -111.930907},
    'Arkansas': {'pop': 3011524, 'lat': 34.751928, 'lon': -92.131378},
    'California': {'pop': 39538223, 'lat': 37.271875, 'lon': -119.270415},
    'Colorado': {'pop': 5773714, 'lat': 39.064603, 'lon': -105.311104},
    'Connecticut': {'pop': 3605944, 'lat': 41.518783, 'lon': -72.757507},
    'Delaware': {'pop': 989948, 'lat': 39.145251, 'lon': -75.418921},
    'Florida': {'pop': 21538187, 'lat': 28.932040, 'lon': -81.928961},
    'Georgia': {'pop': 10711908, 'lat': 32.678125, 'lon': -83.222976},
    'Idaho': {'pop': 1839106, 'lat': 45.494576, 'lon': -114.142430},
    'Illinois': {'pop': 12812508, 'lat': 40.000000, 'lon': -89.000000},
    'Indiana': {'pop': 6785528, 'lat': 39.766219, 'lon': -86.441277},
    'Iowa': {'pop': 3190369, 'lat': 41.938317, 'lon': -93.389798},
    'Kansas': {'pop': 2937880, 'lat': 38.498779, 'lon': -98.320078},
    'Kentucky': {'pop': 4505836, 'lat': 37.822294, 'lon': -85.768240},
    'Louisiana': {'pop': 4657757, 'lat': 31.068933, 'lon': -92.645524},
    'Maine': {'pop': 1362359, 'lat': 45.254229, 'lon': -69.000000},
    'Maryland': {'pop': 6177224, 'lat': 38.806352, 'lon': -77.268416},
    'Massachusetts': {'pop': 7029917, 'lat': 42.062940, 'lon': -71.718067},
    'Michigan': {'pop': 10077331, 'lat': 44.943560, 'lon': -86.415805},
    'Minnesota': {'pop': 5706494, 'lat': 46.441859, 'lon': -93.365515},
    'Mississippi': {'pop': 2961279, 'lat': 32.585106, 'lon': -89.877220},
    'Missouri': {'pop': 6154913, 'lat': 38.304664, 'lon': -92.437099},
    'Montana': {'pop': 1084225, 'lat': 47.052784, 'lon': -109.633803},
    'Nebraska': {'pop': 1961504, 'lat': 41.500000, 'lon': -99.680902},
    'Nevada': {'pop': 3104614, 'lat': 39.493241, 'lon': -117.023061},
    'New Hampshire': {'pop': 1377529, 'lat': 44.000000, 'lon': -71.500000},
    'New Jersey': {'pop': 9288994, 'lat': 40.143006, 'lon': -74.731116},
    'New Mexico': {'pop': 2117522, 'lat': 34.166204, 'lon': -106.026068},
    'New York': {'pop': 20201249, 'lat': 43.000000, 'lon': -75.000000},
    'North Carolina': {'pop': 10439393, 'lat': 35.214563, 'lon': -79.891267},
    'North Dakota': {'pop': 779094, 'lat': 47.467882, 'lon': -100.336982},
    'Ohio': {'pop': 11799448, 'lat': 40.190362, 'lon': -82.669252},
    'Oklahoma': {'pop': 3959353, 'lat': 35.309765, 'lon': -98.716559},
    'Oregon': {'pop': 4237256, 'lat': 44.141905, 'lon': -120.538099},
    'Pennsylvania': {'pop': 13002700, 'lat': 40.878120, 'lon': -77.861786},
    'Rhode Island': {'pop': 1097379, 'lat': 41.582728, 'lon': -71.506451},
    'South Carolina': {'pop': 5118425, 'lat': 33.625050, 'lon': -80.947038},
    'South Dakota': {'pop': 886667, 'lat': 44.212699, 'lon': -100.247544},
    'Tennessee': {'pop': 6910840, 'lat': 35.830521, 'lon': -85.978599},
    'Texas': {'pop': 29145505, 'lat': 31.169336, 'lon': -99.683641},
    'Utah': {'pop': 3271616, 'lat': 39.499761, 'lon': -111.547028},
    'Vermont': {'pop': 643077, 'lat': 43.871755, 'lon': -72.447783},
    'Virginia': {'pop': 8631393, 'lat': 37.123224, 'lon': -78.492772},
    'Washington': {'pop': 7705281, 'lat': 47.751076, 'lon': -120.740135},
    'West Virginia': {'pop': 1793716, 'lat': 38.468041, 'lon': -80.969625},
    'Wisconsin': {'pop': 5893718, 'lat': 44.786297, 'lon': -89.826705},
    'Wyoming': {'pop': 576851, 'lat': 43.000325, 'lon': -107.554567}
}
STATES = sorted(list(STATE_DATA.keys()))
NUM_STATES = len(STATES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_process_covid_data(confirmed_path):
    confirmed_df = pd.read_csv(confirmed_path)

    def aggregate_by_state(df, states):
        df = df[df['Province_State'].isin(states)]
        date_cols = df.columns[df.columns.str.match(r'\d+/\d+/\d+')]
        return df.groupby('Province_State')[date_cols].sum().reindex(states)

    confirmed = aggregate_by_state(confirmed_df, STATES)
    confirmed.columns = pd.to_datetime(confirmed.columns, format='%m/%d/%y')

    daily_confirmed = confirmed.diff(axis=1).fillna(0)
    daily_confirmed[daily_confirmed < 0] = 0
    daily_confirmed = daily_confirmed.T.rolling(window=7, min_periods=1, center=True).mean().T.fillna(0)
    return daily_confirmed


### --- MODIFIED --- ###
# This function now creates a weighted graph
def create_graph_tensors_weighted(dist_thresh=1000, sigma=300):
    """
    Creates a graph where edge weights are determined by an inverse exponential
    of the distance between states.

    Returns:
        edge_index: The sparse graph connectivity.
        edge_weight: The weight for each edge in edge_index.
    """
    n = len(STATES)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j: continue
            s1, s2 = STATES[i], STATES[j]
            lat1, lon1, lat2, lon2 = map(np.radians,
                                         [STATE_DATA[s1]['lat'], STATE_DATA[s1]['lon'], STATE_DATA[s2]['lat'],
                                          STATE_DATA[s2]['lon']])
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            dist = 2 * np.arcsin(np.sqrt(a)) * 6371  # Distance in km
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # Create adjacency matrix based on distance threshold
    adj = np.where(dist_matrix < dist_thresh, 1, 0)
    np.fill_diagonal(adj, 1)  # States are connected to themselves

    # Create edge weights using a Gaussian kernel
    # w_ij = exp(- (dist_ij^2) / (2 * sigma^2) )
    # This makes closer states have weights near 1, and distant states near 0.
    weights = np.exp(-np.square(dist_matrix) / (2 * np.square(sigma)))

    # Only keep weights for edges that actually exist in the graph
    weights = weights * adj

    # Convert to sparse format (edge_index and edge_weight)
    edge_index_np = np.array(np.where(adj))
    edge_weight_np = weights[edge_index_np[0], edge_index_np[1]]

    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32).to(device)

    return edge_index, edge_weight


def create_sequences(data, input_seq_len, output_seq_len):
    xs, ys = [], []
    for i in range(data.shape[1] - input_seq_len - output_seq_len + 1):
        x = data[:, i:(i + input_seq_len), :]
        y = data[:, (i + input_seq_len):(i + input_seq_len + output_seq_len), 0]
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.array(xs), dtype=torch.float32).to(device), torch.tensor(np.array(ys),
                                                                                    dtype=torch.float32).to(device)


class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    ### --- MODIFIED --- ###
    # Forward pass now accepts edge_weight
    def forward(self, x_seq, edge_index, edge_weight=None):
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        gcn_out_seq = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :]
            x_t_flat = x_t.reshape(-1, num_features)

            # Pass edge_weight to the GCN layer
            gcn_out_t = self.relu(self.conv1(x_t_flat, edge_index, edge_weight=edge_weight))

            gcn_out_t_reshaped = gcn_out_t.view(batch_size, num_nodes, -1)
            gcn_out_seq.append(gcn_out_t_reshaped)

        gcn_out_seq = torch.stack(gcn_out_seq, dim=1)

        gru_input = gcn_out_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        gru_out, _ = self.gru(gru_input)

        last_hidden_state = gru_out[:, -1, :]
        x = last_hidden_state.reshape(batch_size, num_nodes, -1)
        x = self.fc(x)

        return self.relu(x)


def analyze_neighbor_influence(model, target_state_name, data_tensor, edge_index, edge_weight, states_list, mean, std):
    model.eval()
    target_idx = states_list.index(target_state_name)

    with torch.no_grad():
        ### --- MODIFIED --- ###
        base_pred_normalized = model(data_tensor, edge_index, edge_weight)

    base_prediction = base_pred_normalized[0, target_idx, 0] * std[0] + mean[0]

    neighbors = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src == target_idx and dst != target_idx: neighbors.add(dst)
        if dst == target_idx and src != target_idx: neighbors.add(src)

    print(f"\nAnalyzing influence on '{target_state_name}'. Its neighbors are: {[states_list[n] for n in neighbors]}")
    influences = {}

    for neighbor_idx in neighbors:
        neighbor_name = states_list[neighbor_idx]
        perturbed_data = data_tensor.clone()
        perturbed_data[0, :, neighbor_idx, 0] = 0.0

        with torch.no_grad():
            ### --- MODIFIED --- ###
            perturbed_pred_normalized = model(perturbed_data, edge_index, edge_weight)

        perturbed_prediction = perturbed_pred_normalized[0, target_idx, 0] * std[0] + mean[0]
        influence_score = torch.abs(base_prediction - perturbed_prediction).item()
        influences[neighbor_name] = influence_score

    sorted_influences = {k: v for k, v in sorted(influences.items(), key=lambda item: item[1], reverse=True)}
    return sorted_influences


if __name__ == '__main__':
    CONFIRMED_PATH = "/kaggle/input/us-dataset/time_series_covid19_confirmed_US.csv"

    print("Loading data and creating features...")
    daily_infections_df = load_and_process_covid_data(CONFIRMED_PATH)

    df_T = daily_infections_df.T
    infection_data = df_T.values
    month_data = np.sin(2 * np.pi * df_T.index.month / 12).values
    day_of_week_data = np.sin(2 * np.pi * df_T.index.dayofweek / 7).values

    all_features = np.stack([
        infection_data,
        np.tile(month_data[:, np.newaxis], (1, NUM_STATES)),
        np.tile(day_of_week_data[:, np.newaxis], (1, NUM_STATES))
    ], axis=2)

    mean = all_features.mean(axis=(0, 1))
    std = all_features.std(axis=(0, 1))
    data_normalized = (all_features - mean) / std

    ### --- MODIFIED --- ###
    # Call the new weighted graph function
    edge_index, edge_weight = create_graph_tensors_weighted()

    INPUT_SEQ_LEN = 14
    OUTPUT_SEQ_LEN = 7
    X, y = create_sequences(data_normalized.transpose(1, 0, 2), INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    X = X.permute(0, 2, 1, 3)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    model = STGCN(num_nodes=NUM_STATES, in_channels=X.size(3), hidden_channels=64, out_channels=OUTPUT_SEQ_LEN).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False)

    print(f"Starting training on {device}...")
    epochs = 200
    batch_size = 32
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        model.train()
        total_train_loss = 0
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            ### --- MODIFIED --- ###
            # Pass edge weights during training
            pred = model(batch_x, edge_index, edge_weight)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / (len(X_train) / batch_size)

        model.eval()
        with torch.no_grad():
            ### --- MODIFIED --- ###
            # Pass edge weights during validation
            val_preds = model(X_test, edge_index, edge_weight)
            val_loss = criterion(val_preds, y_test)

        scheduler.step(val_loss)
        pbar.set_description(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")

    # --- EVALUATION ---
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        ### --- MODIFIED --- ###
        test_preds = model(X_test, edge_index, edge_weight).cpu().numpy()

    prediction_rescaled = test_preds[:, :, 0] * std[0] + mean[0]
    ground_truth_rescaled = y_test.cpu().numpy()[:, :, 0] * std[0] + mean[0]
    prediction_rescaled[prediction_rescaled < 0] = 0

    # ... (Plotting code remains the same) ...
    states_to_plot = ['California', 'New York', 'Florida', 'Illinois']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    for i, state_name in enumerate(states_to_plot):
        ax = axes[i]
        state_idx = STATES.index(state_name)
        ax.plot(ground_truth_rescaled[:, state_idx], label='Ground Truth', color='tab:blue', linewidth=2)
        ax.plot(prediction_rescaled[:, state_idx], label='Prediction', linestyle='--', color='tab:orange', linewidth=2)
        ax.set_title(f"Forecast for {state_name}", fontsize=14)
        ax.legend();
        ax.grid(True);
        ax.set_ylim(bottom=0)
    fig.suptitle("COVID-19 Daily Infections Forecast for Various US States", fontsize=20)
    fig.text(0.5, 0.04, 'Days into the Future (Test Set)', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, 'Daily New Infections', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.08, 0.05, 0.95, 0.95]);
    plt.show()

    # --- NEIGHBOR INFLUENCE ANALYSIS ---
    print("\n" + "=" * 50);
    print("RUNNING NEIGHBOR INFLUENCE ANALYSIS (WEIGHTED GRAPH)");
    print("=" * 50)

    sample_to_analyze = X_test[0:1]
    target_state = 'Illinois'

    ### --- MODIFIED --- ###
    # Pass edge_weight to the analysis function
    influences = analyze_neighbor_influence(
        model=model,
        target_state_name=target_state,
        data_tensor=sample_to_analyze,
        edge_index=edge_index,
        edge_weight=edge_weight,  # Pass the new weights
        states_list=STATES,
        mean=mean,
        std=std
    )
    print(f"\nInfluence Scores for {target_state} (higher score = more impact):")
    for neighbor, score in influences.items():
        print(f"  - {neighbor}: {score:.2f}")

    # ... (Visualization code for influence remains the same) ...
    fig, ax = plt.subplots(figsize=(10, 6))
    neighbors = list(influences.keys());
    scores = list(influences.values())
    bars = ax.bar(neighbors, scores, color='skyblue')
    ax.set_ylabel('Influence Score (Change in Predicted Infections)', fontsize=12)
    ax.set_title(f"Model-Learned Influence of Neighboring States on {target_state}'s Forecast (Weighted Graph)",
                 fontsize=14)
    ax.tick_params(axis='x', rotation=45);
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    plt.tight_layout();
    plt.show()