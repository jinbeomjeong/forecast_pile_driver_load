import keras, tf2onnx, logging
import tensorflow as tf

from utils.layer import DecompositionLayer
from utils.metric import smape

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_path = '../model_30.keras'
best_model = keras.models.load_model(model_path, custom_objects={'DecompositionLayer': DecompositionLayer,
                                                                 'smape': smape})

logging.info(f'Model loaded from {model_path}')

spec = (tf.TensorSpec(best_model.inputs[0].shape, tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec)
logging.info('converted ONNX model')

with open('../model.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

logging.info('saved ONNX model')
