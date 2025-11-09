import tensorflow as tf
from tensorflow import keras
from utils.model import time_mixer_block
from utils.miscellaneous import count_divisions_by_two
from tensorflow.keras.metrics import Precision, Recall, AUC


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    def build_model(input_shape, d_dims=64, dropout_rate=0.2, learning_rate=0.001):
        input_layer = keras.layers.Input(shape=input_shape)

        pos_tensor = input_layer[:, :, 0:4]
        pressure_rotation_speed_tensor = input_layer[:, :, 4:6]

        pos_tensor = keras.layers.BatchNormalization()(pos_tensor)
        pressure_rotation_speed_tensor = keras.layers.LayerNormalization()(pressure_rotation_speed_tensor)

        x = keras.layers.concatenate([pos_tensor, pressure_rotation_speed_tensor], axis=2)
        x_res = keras.layers.Dense(units=d_dims, activation='gelu')(x)

        for i in range(count_divisions_by_two(input_shape[0])+1):
            dilation_rate = 2 ** i
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x_res)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x)
            x = keras.layers.Dropout(dropout_rate)(x)

            x_res = keras.layers.BatchNormalization()(x + x_res)
            x_res = keras.layers.Activation('gelu')(x_res)

        y = keras.layers.Flatten()(x_res)
        y2, y3 = tf.split(y, num_or_size_splits=2, axis=1)
        y3 = keras.layers.LayerNormalization()(y3)
        y3 = keras.layers.Activation('gelu')(y3)

        y_res = keras.layers.Dense(units=input_shape[0]*3, activation='linear')(y2*y3)

        for j in range(3):
            y = time_mixer_block(input_layer=y_res, pred_len=input_shape[0]*3, dropout_rate=dropout_rate)
            y_res = y + y_res

        y = keras.layers.Dropout(dropout_rate)(y_res)

        y2, y3 = tf.split(y, num_or_size_splits=2, axis=1)
        y3 = keras.layers.LayerNormalization()(y3)
        y3 = keras.layers.Activation('gelu')(y3)

        y = keras.layers.LayerNormalization()(y2 * y3)
        y = keras.layers.Dense(units=1, activation='sigmoid')(y)

        model = keras.models.Model(inputs=input_layer, outputs=y)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), AUC()])

        return model
