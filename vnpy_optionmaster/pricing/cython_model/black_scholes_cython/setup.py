"""编译 Black-Scholes 定价的 Cython 扩展。"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='black_scholes_cython',
    ext_modules=cythonize("black_scholes_cython.pyx"),
)
