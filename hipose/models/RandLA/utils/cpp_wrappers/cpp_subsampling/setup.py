from setuptools import setup, Extension
import numpy as np

# Module name
m_name = "grid_subsampling"

# Source files
SOURCES = ["../cpp_utils/cloud/cloud.cpp",
           "grid_subsampling/grid_subsampling.cpp",
           "wrapper.cpp"]

# Create extension module
module = Extension(
    name=m_name,
    sources=SOURCES,
    include_dirs=[np.get_include()],  # Add NumPy include directory
    extra_compile_args=['-std=c++11', '-D_GLIBCXX_USE_CXX11_ABI=0'],
    language='c++'
)

# Setup configuration
setup(
    name=m_name,
    ext_modules=[module],
    zip_safe=False,  # Required for C extensions
)

