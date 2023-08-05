import torch
from my_add_op import *

model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), 'my_add.onnx', input_names=['a', 'b'], output_names=['out'])
torch_output = model(input, input).detach().numpy()
# print('torch_output:{}'.format(torch_output))

import onnxruntime
import numpy as np
sess = onnxruntime.InferenceSession('my_add.onnx')
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]
# print('ort_output:{}'.format(ort_output))

assert np.allclose(torch_output, ort_output)
