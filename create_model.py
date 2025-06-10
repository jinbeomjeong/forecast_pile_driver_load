import datetime, argparse
import numpy as np
import tensorflow as tf

from utils.Dataset import load_logging_data, create_lstm_dataset
from utils.ForecastModel import lstm_est_model_v2
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='data')
args = parser.parse_args()

dataset = load_logging_data(data_root_path=args.path)

angle_name_list = list(dataset.columns)[22:24] + list(dataset.columns)[26:28]
pressure_name_list = list(dataset.columns)[18:20] + list(dataset.columns)[24:26]
dataset['power'] = dataset['pressure_1_pressure_transmitter_1_drive1 CH=23'] * dataset['caloutput_rotate_velocity CH=25']

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
lstm_model = lstm_est_model_v2(input_tensor=feature, seq_len=seq_len, hidden_size=hidden_size)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=0)
csv_logger = keras.callbacks.CSVLogger(filename='log_'+str(pred_distance)+'.csv', append=False, separator=',')
model_chk_point = keras.callbacks.ModelCheckpoint(filepath='model_'+str(pred_distance)+'.keras', monitor="val_loss", verbose=0,
                                                  save_best_only=True, save_weights_only=False, mode="min", save_freq="epoch",
                                                  initial_value_threshold=None)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

lstm_model.fit(x=feature, y=target, validation_data=(feature, target), epochs=10000000,
               batch_size=10000, verbose=2, callbacks=[early_stop, csv_logger, model_chk_point, tensorboard_callback])

best_lstm_model = keras.models.load_model('model_30.keras')
pred = np.squeeze(best_lstm_model.predict(feature, verbose=1))

for i, val in enumerate(pred):
    if val < 0:
        pred[i] = 0

print(r2_score(target, pred))
print(mean_absolute_error(target, pred))
print(mean_absolute_percentage_error(target, pred))