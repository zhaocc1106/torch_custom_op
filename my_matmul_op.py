import torch
import my_lib2


class MyMatMulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib2.my_matmul(a, b)

    @staticmethod
    def symbolic(g, a, b):
        return g.op('com.microsoft::MyMatMul', a, b)  # 自定义onnx算子


my_matmul = MyMatMulFunction.apply


class MyMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_matmul(a, b)
