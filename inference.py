import keras, argparse, os
import numpy as np
import pandas as pd

from utils.Dataset import create_lstm_dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='model.keras')
parser.add_argument("--input_file_path", type=str, default='data')
args = parser.parse_args()

best_lstm_model = keras.models.load_model('model_30.keras')


file_name = os.path.basename(args.input_file_path)
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
dataset = data_pd[::30]

angle_name_list = list(dataset.columns)[22:24] + list(dataset.columns)[26:28]
pressure_name_list = list(dataset.columns)[18:20] + list(dataset.columns)[24:26]
dataset['power'] = dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'].values * dataset['caloutput_rotate_velocity CH=25'].values

dataset = dataset[['pressure_1_pressure_transmitter_1_drive1 CH=23']+angle_name_list+['caloutput_rotate_velocity CH=25', 'caloutput_drill_depth CH=26', 'power']]

for angle_name in angle_name_list:
    dataset[angle_name] = dataset[angle_name]/33

dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] = dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] / 256
dataset['caloutput_rotate_velocity CH=25'] = dataset['caloutput_rotate_velocity CH=25'] / 50
dataset['caloutput_drill_depth CH=26'] = dataset['caloutput_drill_depth CH=26'] / 31
dataset['power'] = dataset['power']/4100

extract_data_df = dataset[['pressure_1_pressure_transmitter_1_drive1 CH=23']+angle_name_list+['caloutput_rotate_velocity CH=25', 'caloutput_drill_depth CH=26', 'power']]

seq_len = 30
pred_distance = 30
hidden_size = 256

feature, target = create_lstm_dataset(extract_data_df.values, seq_len=seq_len, pred_distance=pred_distance, target_idx_pos=7)

best_lstm_model = keras.models.load_model(args.model_path)
pred = np.squeeze(best_lstm_model.predict(feature, verbose=1))

for i, val in enumerate(pred):
    if val < 0:
        pred[i] = 0

print(r2_score(target, pred))
print(mean_absolute_error(target, pred))
print(mean_absolute_percentage_error(target, pred))

