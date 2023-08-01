from ixrt import IxRT, RuntimeConfig, RuntimeContext
import torch
import os
import subprocess
from loguru import logger
import numpy as np
import time


def config_init_engine(config):
    model = config.perf_dir + "/" + config.onnx_path
    quant_file = None

    runtime_config = RuntimeConfig()

    input_shapes = [config.batch_size, 3, config.image_size, config.image_size]    
    runtime_config.input_shapes = [("input", input_shapes)]
    runtime_config.device_idx = 0

    precision = "float16"
    if precision=="int8":
        assert quant_file, "Quant file must provided for int8 inferencing."

    runtime_config.runtime_context = RuntimeContext(
        precision,
        "nhwc",
        use_gpu=True,
        pipeline_sync=True,
        input_types={"input": "float32"},
        output_types={"output": "float32"},
        input_device="gpu",
        output_device="gpu",
    )

    runtime = IxRT.from_onnx(model, quant_file, runtime_config)
    return runtime


def build_engine(config):
    output_path = config.log_dir + "/" + config.ixrt_tmp_path

    time.sleep(10)

    dir_output_path = os.path.dirname(output_path)
    os.makedirs(dir_output_path, exist_ok=True)

    runtime = config_init_engine(config)
    print(f"Build Engine File: {output_path}")
    runtime.BuildEngine()
    runtime.SerializeEngine(output_path)
    print("Build Engine done!")

    runtime = IxRT()
    runtime.LoadEngine(output_path, config.batch_size)
    return runtime


def get_inference_toolkits(config):
    engine = build_engine(config)
    return (engine, allocate_buffers, inference)


class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    output_map = engine.GetOutputShape()
    output_io_buffers = []   
    output_types = {}
    config = engine.GetConfig()
    for key, val in config.runtime_context.output_types.items():
        output_types[key] = str(val)
    for name, shape in output_map.items():
        # 1. apply memory buffer for output of the shape
        if output_types[name] =="float32":
            buffer = np.zeros(shape.dims, dtype=np.float32)
        elif output_types[name] =="int32":
            buffer = np.zeros(shape.dims, dtype=np.int32)
        elif output_types[name] =="float16":
            buffer = np.zeros(shape.dims, dtype=np.float16)
        else:
            raise RuntimeError("need to add a {} datatype of output".format(output_types[name]))
        buffer = torch.tensor(buffer).cuda()
        # 2. put the buffer to a list
        output_io_buffers.append([name, buffer, shape])
    
    engine.BindIOBuffers(output_io_buffers)
    return output_io_buffers


def inference(context, inputs, outputs):
    input_map = context.GetInputShape()
    input_io_buffers = []

    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if k not in input_map.keys():
                raise RuntimeError("input name err ", k)
            if not v.is_contiguous():
                v = v.contiguous()
            shape = input_map[k]
            _shape, _padding = shape.dims, shape.padding
            _shape = [i + j for i, j in zip(_shape, _padding)]
            _shape = [_shape[0], *_shape[2:], _shape[1]]
            input_io_buffers.append([k, v, shape])
    elif isinstance(inputs, torch.Tensor):
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        name, shape = list(input_map.items())[0]
        _shape, _padding = shape.dims, shape.padding
        _shape = [i + j for i, j in zip(_shape, _padding)]
        _shape = [_shape[0], *_shape[2:], _shape[1]]
        input_io_buffers.append([name, inputs, shape])
    else:
        raise RuntimeError("input invalid!")

    context.BindIOBuffers(outputs)
    context.LoadInput(input_io_buffers)

    torch.cuda.synchronize()
    context.Execute()
    torch.cuda.synchronize()

    gpu_io_buffers = []
    for buffer in outputs:
        gpu_io_buffers.append([buffer[0], buffer[1], buffer[2]])

    return gpu_io_buffers
