#!/bin/bash
cd utils/nearest_neighbors
pip isntall .
cd ../../

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

