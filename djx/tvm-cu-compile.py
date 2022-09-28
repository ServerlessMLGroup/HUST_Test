import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_no', type=int, default=0)
args = parser.parse_args()

gpu_no = args.gpu_no

import sys

sys.path.append("..")

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ("%d" % args.gpu_no)

if __name__ == '__main__':
    model_url = (
        "https://github.com/onnx/models/raw/main/"
        "vision/classification/resnet/model/"
        "resnet50-v2-7.onnx"
    )

    model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
    print("download_testdata finish!")
    onnx_model = onnx.load(model_path)
    print("onnx.load finish!")
    target = tvm.target.cuda()
    # The input name may vary across model types. You can use a tool
    # like Netron to check input names
    input_name = "data"
    shape_dict = {input_name: (1, 3, 224, 224)}
    print("shape_dict", shape_dict)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    source_module = lib.get_lib().imported_modules[0]

    source_code = source_module.get_source()

    with open("resnet50_source_code_cuda_base.txt", "a") as f:
        print(source_code, file=f)