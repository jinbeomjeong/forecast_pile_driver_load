import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl












class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (sequence_length, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                 output_sequence_length=1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.input_linear = nn.Linear(n_features, d_model)  # 입력 특성 수를 d_model로 매핑
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=False)  # (seq_len, batch, feature)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 트랜스포머 인코더의 출력을 받아 회귀 값을 예측하는 선형 레이어
        # 인코더의 마지막 레이어 출력 (d_model)을 받아 n_features * output_sequence_length 로 예측
        self.output_linear = nn.Linear(d_model, n_features * output_sequence_length)
        self.output_sequence_length = output_sequence_length

    def forward(self, src):
        # src: (batch_size, sequence_length, n_features)
        # TransformerEncoder는 (sequence_length, batch_size, n_features) 형태를 기대하므로 transpose
        src = src.permute(1, 0, 2)  # (sequence_length, batch_size, n_features)

        src = self.input_linear(src)  # (sequence_length, batch_size, d_model)
        src = self.positional_encoding(src)

        output = self.transformer_encoder(src)  # (sequence_length, batch_size, d_model)

        # 마지막 타임스텝의 출력을 사용하여 예측
        # 회귀 분석에서는 보통 마지막 시점의 출력을 사용하거나, 모든 시점의 출력을 결합하여 사용
        # 여기서는 마지막 시점의 출력을 사용합니다.
        output = output[-1, :, :]  # (batch_size, d_model)

        predictions = self.output_linear(output)  # (batch_size, n_features * output_sequence_length)

        # 예측 값을 (batch_size, output_sequence_length, n_features) 형태로 reshape
        predictions = predictions.view(predictions.size(0), self.output_sequence_length, self.n_features)

        return predictions


class LitTransformerRegressor(pl.LightningModule):
    def __init__(self, n_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                 output_sequence_length, learning_rate):
        super().__init__()
        self.save_hyperparameters() # 하이퍼파라미터 저장
        self.model = TransformerRegressor(n_features, d_model, nhead, num_encoder_layers,
                                           dim_feedforward, dropout, output_sequence_length)
        self.criterion = nn.MSELoss() # 회귀 문제에 적합한 MSE 손실 함수

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
