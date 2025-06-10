import tensorflow as tf
from tensorflow import keras


def lstm_est_model_v1(feature, seq_len, hidden_size, n_outputs):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    input_tensor = keras.layers.Input(shape=(seq_len, feature.shape[2]), dtype=tf.float32)
    input_flatten_layer = keras.layers.Flatten()(input_tensor)

    mean_tensor = tf.reduce_mean(input_tensor, axis=1)
    max_tensor = tf.reduce_max(input_tensor, axis=1)
    min_tensor = tf.reduce_min(input_tensor, axis=1)
    sum_tensor = tf.reduce_sum(input_tensor, axis=1)
    std_tensor = tf.math.reduce_std(input_tensor, axis=1)

    avg_low_bool = tf.less(input_tensor, tf.expand_dims(mean_tensor, axis=1))
    avg_low_count = tf.math.count_nonzero(avg_low_bool, axis=1, dtype=tf.float32)
    avg_high_count = tf.math.count_nonzero(tf.math.logical_not(avg_low_bool), axis=1, dtype=tf.float32)

    feature_eng_tensor = keras.layers.concatenate(inputs=[mean_tensor, max_tensor, min_tensor, sum_tensor, std_tensor, avg_low_count, avg_high_count])

    # Convolutional Layers
    conv_layer = keras.layers.Conv1D(filters=64, kernel_size=7, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2(0.03))(input_tensor)

    conv_layer = keras.layers.Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2(0.03))(conv_layer)
    conv_layer = keras.layers.MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = keras.layers.Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2(0.03))(conv_layer)
    conv_layer = keras.layers.Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2(0.03))(conv_layer)
    conv_flatten_layer = keras.layers.Flatten()(conv_layer)

    # LSTM Layers
    lstm_output_1 = keras.layers.LSTM(units=hidden_size, return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.03), name='lstm_1')(conv_layer)
    lstm_output_2 = keras.layers.LSTM(units=hidden_size, return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.03), name='lstm_2')(conv_layer)
    lstm_output_2 = keras.layers.LSTM(units=int(hidden_size /2), return_sequences=False, kernel_regularizer=keras.regularizers.l2(0.03), name='lstm_3')(lstm_output_2)

    # Concatenation of all features
    concat_tensor = keras.layers.concatenate(inputs=[input_flatten_layer, conv_flatten_layer, lstm_output_1, lstm_output_2, feature_eng_tensor], axis=-1)

    # Output Layer
    output_tensor = keras.layers.Dense(n_outputs, activation='linear')(concat_tensor)

    # Model Compilation
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model



def lstm_est_model_v2(input_tensor, seq_len, hidden_size):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    input_layer = keras.layers.Input(shape=(seq_len, input_tensor.shape[2]), dtype=tf.float32)

    conv_layer = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
    #conv_layer = keras.layers.MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv_layer)
    #conv_layer = keras.layers.MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_layer)
    #conv_layer = keras.layers.MaxPooling1D(pool_size=2)(conv_layer)

    lstm_layer = keras.layers.LSTM(units=hidden_size, return_sequences=False, name='lstm')(conv_layer)

    lstm_output_layer = keras.layers.Dense(1, activation='linear')(lstm_layer)

    model = keras.Model(inputs=input_layer, outputs=lstm_output_layer)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model