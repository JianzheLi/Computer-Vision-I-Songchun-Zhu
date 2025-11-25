from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# 根据平台选择合适的编译选项
if sys.platform.startswith('linux'):
    extra_compile_args = ['-fopenmp', '-O3', '-march=native', '-fno-math-errno']
    extra_link_args = ['-fopenmp']
else:
    extra_compile_args = ['-fopenmp', '-O3']
    extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "gibbs_optimized",
        ["gibbs_optimized.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': '3',
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
        'nonecheck': False,
    })
)