import tensorflow as tf, tf2onnx, numpy as np

KERAS_FILE = "gpe_cnn_jimin.h5"
ONNX_FILE  = "gpe_cnn_jimin.onnx"
OPSET      = 13          # TRT 8.5+ handles opset ≥ 13 well
WINDOW = 80

# 1) load the Sequential model from disk
seq = tf.keras.models.load_model(KERAS_FILE, compile=False)

# 2) wrap it as a Functional graph with an explicit Input tensor
inp  = tf.keras.Input(shape=(WINDOW, 8), name="input")   # (time, channels)
out  = seq(inp) # call once ➜ built

model = tf.keras.Model(inp, out, name="gpe_cnn")

# 3) convert to ONNX
spec = (tf.TensorSpec((None, WINDOW, 8), tf.float32, name="input"),)

onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature = spec,
        opset           = OPSET,
        output_path     = ONNX_FILE)

print(f"✓ saved {ONNX_FILE}")
