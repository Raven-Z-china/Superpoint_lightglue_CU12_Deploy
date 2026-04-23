#!/usr/bin/env python3
import tensorrt as trt
import os

os.chdir('/home/raven/code/SuperPoint-LightGlue-TensorRT')

for engine_file in ['weights/offical_sp_FP32_v1.engine', 'weights/offical_lg_FP32_v1.engine']:
    print(f"\n{'='*50}")
    print(f"Engine: {engine_file}")
    print('='*50)
    
    with open(engine_file, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        print("Failed to load engine")
        continue
    
    print(f"Precision:")
    
    dtype_map = {
        trt.DataType.FLOAT: "FP32",
        trt.DataType.HALF: "FP16",
        trt.DataType.INT8: "INT8",
    }
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
        print(f"  {name} ({mode_str}): {dtype_map.get(dtype, str(dtype))}")
