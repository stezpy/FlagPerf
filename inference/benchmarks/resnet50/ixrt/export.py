import torch
import os
import importlib

def export_model(model, config):
    model.cpu()
    del model

    benchmark_module = importlib.import_module(
        "benchmarks." + config.case + "." + config.framework, __package__)
    model = benchmark_module.create_model(config)
    
    dummy_input = torch.randn(config.batch_size, 3, 224, 224)

    dummy_input = dummy_input.cuda()

    onnx_path = config.perf_dir + "/" + config.onnx_path

    dir_onnx_path = os.path.dirname(onnx_path)
    os.makedirs(dir_onnx_path, exist_ok=True)

    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      verbose=False,
                      input_names=["input"],
                      output_names=["output"],
                      do_constant_folding=True,
                      opset_version=11)
