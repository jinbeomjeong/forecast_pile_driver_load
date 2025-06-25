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


def resnet_block(input_tensor, filters: int, kernel_size=3, stride=1):
    x = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    shortcut = input_tensor

    if stride > 1 or input_tensor.shape[-1]!= filters:
        shortcut = keras.layers.Conv1D(filters=filters, kernel_size=1, strides=stride, padding='same', activation=None)(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    return x


def regression_model(n_of_features: int, seq_len: int):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    input_layer = keras.layers.Input(shape=(seq_len, n_of_features), dtype=tf.float32)
    input_layer = keras.layers.BatchNormalization()(input_layer)

    conv_layer = keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation=None)(input_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=128, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=256, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=512, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    flat_layer = keras.layers.Flatten()(conv_layer)
    flat_layer = keras.layers.BatchNormalization()(flat_layer)

    nn_layer = keras.layers.Dense(units=1, activation='sigmoid')(flat_layer)

    model = keras.Model(inputs=input_layer, outputs=nn_layer)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


def predict_model_v3(n_of_features: int, seq_len, hidden_size, learing_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=learing_rate)

    input_layer = keras.layers.Input(shape=(seq_len, n_of_features), dtype=tf.float32)
    conv_layer = keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv_layer = keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(conv_layer)

    resnet_layer = resnet_block(conv_layer, filters=32)
    resnet_layer = resnet_block(resnet_layer, filters=32)
    resnet_layer = resnet_block(resnet_layer, filters=64)
    resnet_layer = resnet_block(resnet_layer, filters=64)
    resnet_layer = keras.layers.MaxPooling1D(pool_size=2)(resnet_layer)

    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size, return_sequences=True, name='lstm_1'))(resnet_layer)
    lstm_layer = keras.layers.Dropout(0.2)(lstm_layer)
    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_size, return_sequences=False, name='lstm_2'))(lstm_layer)
    lstm_layer = keras.layers.Dropout(0.2)(lstm_layer)

    lstm_layer = keras.layers.BatchNormalization()(lstm_layer)
    nn_layer = keras.layers.Dense(1, activation='linear')(lstm_layer)

    model = keras.Model(inputs=input_layer, outputs=nn_layer)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model


def predict_model_v4(reg_model_path: str, n_of_features: int, seq_len, hidden_size):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    reg_model = keras.models.load_model(reg_model_path)
    reg_model.trainable = False

    input_layer = keras.layers.Input(shape=(seq_len, n_of_features), dtype=tf.float32)
    est_load_layer = reg_model(input_layer, training=False)
    est_load_layer = keras.layers.RepeatVector(seq_len)(est_load_layer)

    concat_layer = keras.layers.concatenate(inputs=[input_layer, est_load_layer], axis=-1)

    conv_layer = keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation=None)(concat_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.Conv1D(filters=128, kernel_size=3, padding='valid', activation=None)(conv_layer)
    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    conv_layer = keras.layers.Activation('swish')(conv_layer)

    conv_layer = keras.layers.BatchNormalization()(conv_layer)
    lstm_layer = keras.layers.LSTM(units=hidden_size, return_sequences=False, name='lstm_1')(conv_layer)

    lstm_layer = keras.layers.BatchNormalization()(lstm_layer)
    lstm_output_layer = keras.layers.Dense(units=1, activation='sigmoid')(lstm_layer)

    model = keras.Model(inputs=input_layer, outputs=lstm_output_layer)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model