import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm




def load_logging_data(data_root_path: str):
    file_name_list = os.listdir(data_root_path)
    dataset = pd.DataFrame()

    for file_name in tqdm(file_name_list, desc='Loading data...'):
        if file_name[0:2] == 'in':
            work_mode = 'pile_in'
        else:
            work_mode = 'pile_out'

        txt_list = []

        with open(os.path.join(data_root_path, file_name), 'r') as file:
            while True:
                txt = file.readline()

                if txt == '':
                    break
                else:
                    txt_list.append(txt)

        column = txt_list[8]
        column = column.split('\t')
        data_list = txt_list[37:]

        data_arr = []

        for data in data_list:
            data = data.split('\t')[:-1]
            data = [float(x) for x in data]
            data_arr.append(np.array(data))

        data_pd = pd.DataFrame(np.array(data_arr), columns=column)
        data_pd = pd.concat([data_pd, pd.DataFrame([work_mode]*data_pd.shape[0], columns=['work_mode'])], axis=1)
        dataset = pd.concat([dataset, data_pd])

    return dataset[::30]


def create_lstm_dataset(data: np.array, seq_len=1, pred_distance=0, target_idx_pos=1):
    feature, target = [], []

    for i in range(data.shape[0] - pred_distance):
        if i+1 >= seq_len:
            feature.append(data[i+1-seq_len:i+1, :])

            if target_idx_pos >= 0:
                target.append(data[i + pred_distance, target_idx_pos])

    return np.array(feature), np.array(target)


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.array, seq_len: int, pred_distance: int, target_idx_pos: int):
        self.__data = data
        self.__seq_len = seq_len
        self.__pred_distance = pred_distance
        self.__target_idx_pos = target_idx_pos

    def __len__(self):
        return len(self.__data) - self.__pred_distance

    def __getitem__(self, idx):
        if idx+1 >= self.__seq_len:
            x = self.__data[idx+1-self.__seq_len : idx+1, :]
            y = self.__data[idx+self.__pred_distance, self.__target_idx_pos]

            return x, y


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, train_data: Dataset, val_data: Dataset, seq_len, batch_size):
        super().__init__()
        self.__train_data = train_data
        self.__val_data = val_data
        self.__seq_len = seq_len
        self.__batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.__train_data, batch_size=self.__batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.__val_data, batch_size=self.__batch_size, shuffle=False)
