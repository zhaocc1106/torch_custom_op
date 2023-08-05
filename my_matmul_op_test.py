import torch
from my_matmul_op import *

model = MyMatMul()
a = torch.ones(4, 3)
b = torch.ones(3, 5)
torch_output = model(a, b).detach()
print('torch_output:{}'.format(torch_output))

# 导出为onnx
torch.onnx.export(model, (a, b), 'my_matmul.onnx', input_names=['a', 'b'], output_names=['out'])
