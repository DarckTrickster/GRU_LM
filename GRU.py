import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Pointer
from geopy.distance import geodesic


# 1. Загрузка данных
def load_trajectory_files(folder_path, max_files=5):
    all_trajectories = []
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.plt'):
                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath, skiprows=6, header=None)
                df = df[[0, 1]]  # lat, lon
                df.columns = ['lat', 'lon']
                all_trajectories.append(df)
                count += 1
                if count >= max_files:
                    break
    print(f"Loaded {len(all_trajectories)} trajectory files")
    return all_trajectories


# 2. Формирование последовательностей
def create_sequences(trajectories, window_size=5):
    X, y = [], []
    for traj in trajectories:
        coords = traj[['lat', 'lon']].values
        if len(coords) < window_size + 1:
            continue
        for i in range(len(coords) - window_size):
            X.append(coords[i:i + window_size])
            y.append(coords[i + window_size])
    return np.array(X), np.array(y)


# 3. Нормализация
def normalize_data(X, y):
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))
    X_norm = (X - mean) / std
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_norm = (y - y_mean) / y_std
    return X_norm, y_norm, mean, std, y_mean, y_std


# 4. Dataset
class TrajDatasetGRU(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # [batch, seq_len, 2]
        self.y = torch.tensor(y, dtype=torch.float32)  # [batch, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 5. Модель на GRU
class GRUTrajPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Output: lat, lon

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])  # Используем последний hidden state
        return out



# 6. Метрики
def mean_trajectory_error(y_true, y_pred):
    return np.mean([geodesic(a, b).meters for a, b in zip(y_true, y_pred)])


def evaluate(y_true, y_pred):
    mte = mean_trajectory_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 Evaluation Metrics:")
    print(f"MTE  = {mte:.2f} meters")
    print(f"MAE  = {mae:.6f} (lat/lon units)")
    print(f"R²   = {r2:.4f}")


#  7. Визуализация
def visualize_predictions(actual, predicted, title="Trajectory Predictions"):
    actual = np.array(actual)
    predicted = np.array(predicted)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(actual[:, 1], actual[:, 0], 'bo-', label="Actual")
    axs[1].plot(predicted[:, 1], predicted[:, 0], 'rx-', label="Predicted")
    axs[2].plot(actual[:, 1], actual[:, 0], 'bo-', label="Actual")
    axs[2].plot(predicted[:, 1], predicted[:, 0], 'rx--', label="Predicted2")

    axs[0].set_title("🟦 Actual Trajectory")
    axs[1].set_title("🟥 Predicted Trajectory")
    axs[2].set_title("🟪 Overlay: Actual vs Predicted")
    for ax in axs:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# 8. Основной pipeline
def main():
    folder = "./Geolife Trajectories 1.3/Data"  # путь к .plt
    trajectories = load_trajectory_files(folder, max_files=5)

    X, y = create_sequences(trajectories, window_size=5)
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

    train_loader = DataLoader(TrajDatasetGRU(X_train, y_train), batch_size=32, shuffle=True)
    test_dataset = TrajDatasetGRU(X_test, y_test)

    model = GRUTrajPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Обучение
    for epoch in range(50):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/50, Loss: {total_loss:.6f}")
    torch.save(model.state_dict(), "gru_trajectory_model.pth")
    print("Модель сохранена как 'gru_trajectory_model.pth'")

    # Оценка
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_tensor).numpy()

    # Денормализация
    y_pred_denorm = y_pred * y_std + y_mean
    y_true_denorm = y_test * y_std + y_mean

    evaluate(y_true_denorm, y_pred_denorm)
    visualize_predictions(y_true_denorm, y_pred_denorm)




if __name__ == "__main__":

    main()
