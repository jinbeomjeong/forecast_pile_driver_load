import json, os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

#with open("data_attribute_name.json", "r") as file_handle:
#    data_attribute = json.load(file_handle)

#feature_name_list = data_attribute['feature_name']
#target_name = data_attribute['target_name']


def load_logging_data(data_root_path: str):
    file_name_list = os.listdir(data_root_path)
    dataset = pd.DataFrame()

    for file_name in tqdm(file_name_list, desc='Loading data...'):
        if file_name[0:2] == 'in':
            work_mode = 'pile_in'
        else:
            work_mode = 'pile_out'

        txt_list = []

        with open(os.path.join('data', file_name), 'r') as file:
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
            feature.append(data[i+1-seq_len:i+1, 0:target_idx_pos])

            if target_idx_pos >= 0:
                target.append(data[i + pred_distance, target_idx_pos])

    return np.array(feature), np.array(target)