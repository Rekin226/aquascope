"""
Lightweight LSTM model for hydrological sequence forecasting.

CPU-friendly PyTorch LSTM with an encoder–decoder architecture.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aquascope.models.base import BaseHydroModel
from aquascope.utils.imports import require

logger = logging.getLogger(__name__)


class LSTMModel(BaseHydroModel):
    """Lightweight LSTM for hydrological time-series forecasting.

    Architecture::

        Input → LSTM(hidden, layers) → Dropout → Linear → Output

    Parameters
    ----------
    seq_length : int
        Number of look-back time steps.
    hidden_size : int
        LSTM hidden-state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate between LSTM layers.
    batch_size : int
        Training mini-batch size.
    learning_rate : float
        Adam optimiser learning rate.
    """

    MODEL_ID = "lstm"
    SUPPORTS_UNCERTAINTY = False
    SUPPORTS_MULTIVARIATE = True

    def __init__(
        self,
        target_variable: str = "value",
        seq_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ):
        super().__init__(target_variable)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self._scaler_mean: float | None = None
        self._scaler_std: float | None = None
        self._last_sequence: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, epochs: int = 50, **kwargs) -> LSTMModel:
        """Train the LSTM on a time-series DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with a ``value`` column (or first column).
        epochs : int
            Number of training epochs.
        """
        require("torch", feature="LSTM forecasting")
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)

        series = self._prepare_series(df)

        self._scaler_mean = float(series.mean())
        self._scaler_std = float(series.std()) + 1e-8
        values = ((series - self._scaler_mean) / self._scaler_std).values.astype(np.float32)

        x_arr, y_arr = self._create_sequences(values, self.seq_length)
        x_tensor = torch.FloatTensor(x_arr).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y_arr)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = self._build_net(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        logger.info("Training LSTM: %d sequences, %d epochs", len(x_arr), epochs)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self._model(batch_x).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info("Epoch [%d/%d] Loss: %.4f", epoch + 1, epochs, avg_loss)

        self._last_sequence = values[-self.seq_length :]
        self._is_fitted = True
        return self

    def predict(self, horizon: int = 7, **kwargs) -> pd.DataFrame:
        """Recursive one-step-ahead forecast for *horizon* days."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        require("torch", feature="LSTM forecasting")
        import torch

        self._model.eval()
        sequence = self._last_sequence.copy()
        predictions: list[float] = []

        with torch.no_grad():
            for _ in range(horizon):
                x = torch.FloatTensor(sequence[-self.seq_length :]).unsqueeze(0).unsqueeze(-1)
                y_norm = self._model(x).item()
                y_pred = y_norm * self._scaler_std + self._scaler_mean
                predictions.append(y_pred)
                sequence = np.append(sequence, y_norm)

        future_dates = self._future_dates(horizon)
        result = pd.DataFrame(
            {"yhat": predictions, "yhat_lower": np.nan, "yhat_upper": np.nan},
            index=future_dates[: len(predictions)],
        )
        result.index.name = "datetime"
        return result

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _create_sequences(values: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in range(len(values) - seq_length):
            x.append(values[i : i + seq_length])
            y.append(values[i + seq_length])
        return np.array(x), np.array(y)

    @staticmethod
    def _build_net(input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """Construct a lightweight PyTorch LSTM network."""
        import torch.nn as nn

        class _HydroLSTMNet(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.dropout = nn.Dropout(dropout)
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.linear(out)

        return _HydroLSTMNet(input_size, hidden_size, num_layers, dropout)
