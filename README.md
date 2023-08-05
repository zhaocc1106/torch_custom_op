# 实现自己的torch算子

## 自定义算子
* my_add: c = 2 * a +b
* my_matmul: c = a * b，使用cublas库实现矩阵乘

## 安装
```bash
python3 setup.py install
```

## 测试
```bash
# 测试my_add并导出为onnx
python3 my_add_op_test.py

# 测试my_matmul并导出为onnx
python3 my_matmul_op_test.py  
```