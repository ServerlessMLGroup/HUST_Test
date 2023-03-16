import numpy as np
import torch

from transformers import BertForSequenceClassification

import tvm
from tvm import relay

model = BertForSequenceClassification.from_pretrained('bert-large-uncased', return_dict=False)

batch_size = 1
seq_len = 128
inputs = (torch.ones(batch_size, seq_len, dtype=torch.int64),
          torch.ones(batch_size, seq_len, dtype=torch.int64),
          torch.ones(batch_size, seq_len, dtype=torch.int64))

input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                ("attention_mask", (inputs[1].shape, "int64")),
                ("token_type_ids", (inputs[2].shape, "int64"))]

with torch.no_grad():
    out = model(*inputs)

script_module = torch.jit.trace(model, inputs).eval()

import time
t1 = time.time()
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
t2 = time.time()

print(relay.transform.InferType()(mod))

print("PT import time:", t2 - t1)

target = "llvm -mcpu=cascadelake"

with tvm.transform.PassContext(opt_level=3):
    # opt_mod, opt_params = relay.optimize(mod, target=target, params=params)
    # print(opt_mod["main"])
    lib = relay.build(mod, target=target, params=params)

runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

runtime.set_input("input_ids", inputs[0].numpy())
runtime.set_input("attention_mask", inputs[1].numpy())
runtime.set_input("token_type_ids", inputs[2].numpy())

runtime.run()