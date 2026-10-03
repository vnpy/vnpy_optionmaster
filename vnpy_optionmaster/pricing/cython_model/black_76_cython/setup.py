"""编译 Black-76 定价的 Cython 扩展。"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='black_76_cython',
    ext_modules=cythonize("black_76_cython.pyx"),
)
