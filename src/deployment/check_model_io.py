from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

root      = Path(__file__).resolve().parent
model_l_dir   = root / "IND_model" / f"gpe_cnn_left.onnx"
model_r_dir   = root / "IND_model" / f"gpe_cnn_right.onnx"

model_l = load_model(model_l_dir, compile=False)
model_r = load_model(model_r_dir, compile=False)

model_l.summary(line_length=120)

# dummy_batch = np.zeros((32, 40, 8), dtype=np.float32)  # 32 samples
# y_pred = model_l.predict(dummy_batch)
# print(y_pred.shape)   # (32, 2)

print("Input  :", model_l.input_shape)    # e.g. (None, 256, 8)
print("Output :", model_l.output_shape)   # e.g. (None, 2)
