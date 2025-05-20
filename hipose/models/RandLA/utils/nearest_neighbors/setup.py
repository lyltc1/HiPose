from setuptools import setup, Extension
import numpy
import os

# 确保当前工作目录是setup.py所在的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ext_modules = [Extension(
       "nearest_neighbors",
       sources=["knn_.cxx"],
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = ["-std=c++11", "-fopenmp", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
       extra_link_args=["-std=c++11", '-fopenmp'],
       define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
  )]

setup(
    name="KNN_NanoFLANN",
    ext_modules=ext_modules,
)