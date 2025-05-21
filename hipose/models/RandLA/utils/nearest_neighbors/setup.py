from setuptools import setup, Extension
import pybind11
import os

setup(
    name='nearest_neighbors',
    version='0.1',
    ext_modules=[
        Extension(
            'nearest_neighbors',
            sources=[
                'knn_pybind.cpp',
                'knn_.cxx',
            ],
            include_dirs=[
                pybind11.get_include(),
                os.path.dirname(os.path.abspath(__file__)),
                # Add other include paths as needed
            ],
            language='c++',
            extra_compile_args=[
                '-fopenmp',
                '-std=c++11',
                '-O3'
            ],
            extra_link_args=[
                '-fopenmp'
            ],
        )
    ],
    zip_safe=False,
)