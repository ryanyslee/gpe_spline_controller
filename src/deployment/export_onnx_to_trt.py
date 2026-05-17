'''
Converts the model in onnx to trt with static batching
input: (1, 80, 8) ; output: (1, 2)
'''

import tensorrt as trt

logger   = trt.Logger(trt.Logger.WARNING)
builder  = trt.Builder(logger)
network  = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser   = trt.OnnxParser(network, logger)

with open("gpe_cnn_ryan.onnx", "rb") as f:
    parser.parse(f.read())

config   = builder.create_builder_config()
profile  = builder.create_optimization_profile()
profile.set_shape("input", min=(1,80,8), opt=(1,80,8), max=(1,80,8))
config.add_optimization_profile(profile)

engine = builder.build_engine(network, config)
with open("gpe_cnn_ryan.trt", "wb") as f:
    f.write(engine.serialize())
    

# One line code to convert model with the static (1, 80, 8) or (2, 80, 8) profile
# Need to call it using its absolute path
# /usr/src/tensorrt/bin/trtexec --onnx=gpe_cnn_ryan.onnx --saveEngine=gpe_cnn_ryan.trt --minShapes=input:1x80x8 --optShapes=input:1x80x8 --maxShapes=input:1x80x8
# /usr/src/tensorrt/bin/trtexec --onnx=gpe_cnn_ryan.onnx --saveEngine=gpe_cnn_ryan.trt --minShapes=input:2x80x8 --optShapes=input:2x80x8 --maxShapes=input:2x80x8

# **Copy the .onnx file to the new Jetson and run trtexec locally on the new board