"""CNN espacial pequena para previsão causal recursiva dos mapas de SPI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .datasets import make_spatial_sequences


class SpatialResidualCNN(nn.Module):
    """Usa os 12 mapas passados como canais e aprende a inovação espacial."""

    def __init__(self, lookback_months: int = 12, width: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(lookback_months, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:, :, :] + self.network(inputs)


def spatial_gradient_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    dx_pred = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]
    dx_obs = observed[:, :, :, 1:] - observed[:, :, :, :-1]
    dy_pred = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
    dy_obs = observed[:, :, 1:, :] - observed[:, :, :-1, :]
    return torch.mean(torch.abs(dx_pred - dx_obs)) + torch.mean(torch.abs(dy_pred - dy_obs))


@dataclass
class DeepForecastResult:
    observed: np.ndarray
    predicted: np.ndarray
    target_times: pd.DatetimeIndex
    best_validation_loss: float
    epochs_trained: int


def train_recursive_spatial_cnn(
    spi,
    lookback_months: int = 12,
    train_end: str = "2013-12-31",
    validation_start: str = "2014-01-01",
    validation_end: str = "2014-12-31",
    test_year: int = 2015,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_weight: float = 0.10,
    max_epochs: int = 400,
    patience: int = 35,
    random_state: int = 42,
) -> DeepForecastResult:
    """Treina até 2013, escolhe época em 2014 e prevê 2015 recursivamente."""
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True, warn_only=True)

    X, y, target_times = make_spatial_sequences(spi, lookback_months, horizon_months=1)
    train_mask = target_times <= pd.Timestamp(train_end)
    validation_mask = (target_times >= pd.Timestamp(validation_start)) & (
        target_times <= pd.Timestamp(validation_end)
    )
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Não há sequências suficientes para treino e validação.")

    calibration = np.concatenate([X[train_mask].ravel(), y[train_mask].ravel()])
    mean = float(np.mean(calibration))
    std = float(np.std(calibration))
    if std == 0:
        raise ValueError("A série de treino é constante.")
    X_scaled = (X - mean) / std
    y_scaled = (y - mean) / std

    train_data = TensorDataset(
        torch.from_numpy(X_scaled[train_mask]).float(),
        torch.from_numpy(y_scaled[train_mask]).float(),
    )
    train_loader = DataLoader(train_data, batch_size=min(16, len(train_data)), shuffle=True)
    X_validation = torch.from_numpy(X_scaled[validation_mask]).float()
    y_validation = torch.from_numpy(y_scaled[validation_mask]).float()

    model = SpatialResidualCNN(lookback_months=lookback_months)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse = nn.MSELoss()
    best_state = deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = mse(prediction, batch_y) + gradient_weight * spatial_gradient_loss(
                prediction, batch_y
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(X_validation)
            validation_loss = float(
                (
                    mse(validation_prediction, y_validation)
                    + gradient_weight * spatial_gradient_loss(validation_prediction, y_validation)
                ).item()
            )
        if validation_loss < best_validation - 1e-6:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    history = np.asarray(
        spi.sel(time=slice(f"{test_year - 1}-01-01", f"{test_year - 1}-12-31")).values,
        dtype=np.float32,
    )[-lookback_months:]
    if history.shape[0] != lookback_months or not np.isfinite(history).all():
        raise ValueError("Histórico imediatamente anterior ao teste está incompleto.")
    history_scaled = (history - mean) / std
    predictions = []
    for _ in range(12):
        tensor = torch.from_numpy(history_scaled[None, :, :, :]).float()
        with torch.no_grad():
            next_map = model(tensor).numpy()[0, 0]
        predictions.append(next_map * std + mean)
        history_scaled = np.concatenate([history_scaled[1:], next_map[None, :, :]], axis=0)

    observed = np.asarray(
        spi.sel(time=slice(f"{test_year}-01-01", f"{test_year}-12-31")).values,
        dtype=float,
    )
    if observed.shape[0] != 12:
        raise ValueError("O ano de teste não contém 12 meses.")
    return DeepForecastResult(
        observed=observed,
        predicted=np.asarray(predictions),
        target_times=pd.date_range(f"{test_year}-01-01", periods=12, freq="MS"),
        best_validation_loss=best_validation,
        epochs_trained=epoch,
    )
