import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.Dataset import load_logging_data, create_lstm_dataset
from utils.ForecastModel import lstm_est_model
from tensorflow import keras


dataset = load_logging_data(data_root_path='data')

angle_name_list = list(dataset.columns)[22:24] + list(dataset.columns)[26:28]

for angle_name in angle_name_list:
    dataset = dataset[dataset[angle_name] < 5]

for angle_name in angle_name_list:
    dataset = dataset[dataset[angle_name] > -5]

dataset.reset_index(drop=True, inplace=True)

pressure_name_list = list(dataset.columns)[18:20] + list(dataset.columns)[24:26]

for pressure_name in pressure_name_list:
    dataset = dataset[dataset[pressure_name] > 0]

dataset.reset_index(drop=True, inplace=True)

dataset = dataset[dataset['caloutput_drill_depth CH=26'] > 3]
dataset.reset_index(drop=True, inplace=True)

dataset = dataset[dataset['caloutput_rotate_velocity CH=25'] > 0]
dataset.reset_index(drop=True, inplace=True)

dataset['power'] = dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] * dataset['caloutput_rotate_velocity CH=25']
dataset['power'] = (dataset['power']-dataset['power'].min()) / (dataset['power'].max()-dataset['power'].min())
dataset = dataset[dataset['power'] != 0]
dataset.reset_index(drop=True, inplace=True)

dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] = dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] / 200
dataset['caloutput_rotate_velocity CH=25'] = dataset['caloutput_rotate_velocity CH=25'] / 35

extract_data = dataset[['pressure_1_pressure_transmitter_1_drive1 CH=23']+angle_name_list+['caloutput_rotate_velocity CH=25', 'caloutput_drill_depth CH=26', 'power']].to_numpy()

grad_data_list = list()

for data in extract_data.T:
    grad_data_list.append(np.gradient(data))

grad_data_arr = np.array(grad_data_list).T
extract_data = np.concatenate([extract_data, grad_data_arr], axis=1)


pred_distance_list = [10, 20, 30]

for pred_distance in pred_distance_list:
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=0)
    csv_logger = keras.callbacks.CSVLogger(filename='log_'+str(pred_distance)+'.csv', append=False, separator=',')
    model_chk_point = keras.callbacks.ModelCheckpoint(filepath='model_'+str(pred_distance)+'.keras', monitor="val_loss", verbose=0,
                                                      save_best_only=True, save_weights_only=False, mode="min", save_freq="epoch",
                                                      initial_value_threshold=None)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    seq_len = 30
    hidden_size = 500
    n_output = 1

    feature, target = create_lstm_dataset(extract_data, seq_len=seq_len, pred_distance=pred_distance, target_idx_pos=7)

    model = lstm_est_model(feature=feature, seq_len=seq_len, hidden_size=hidden_size, n_outputs=n_output)

    model.fit(x=feature, y=target, validation_data=(feature, target), epochs=1000000,
              batch_size=10000, verbose=0, callbacks=[early_stop, csv_logger, model_chk_point, tensorboard_callback])




