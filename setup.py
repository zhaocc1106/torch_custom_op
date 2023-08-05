from setuptools import setup
from torch.utils import cpp_extension

setup(name='my_add',
      ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='my_matmul',
      ext_modules=[cpp_extension.CppExtension('my_lib2', ['my_matmul.cpp'], include_dirs=[
          '/usr/local/cuda/include', '/usr/local/cuda/x86_64-linux/include'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
