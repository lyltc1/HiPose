#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "knn_.h"

namespace py = pybind11;

py::array_t<long> knn(py::array_t<float> pts, py::array_t<float> queries, int K, bool omp = false) {
    // Get array info
    py::buffer_info pts_buf = pts.request();
    py::buffer_info queries_buf = queries.request();

    if (pts_buf.ndim != 2 || queries_buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be 2");

    size_t npts = pts_buf.shape[0];
    size_t dim = pts_buf.shape[1];
    size_t nqueries = queries_buf.shape[0];

    if (queries_buf.shape[1] != dim)
        throw std::runtime_error("Points and queries must have the same dimension");

    // Create output array
    py::array_t<long> indices({nqueries, (size_t)K});
    py::buffer_info indices_buf = indices.request();

    // Get pointers to data
    float* pts_ptr = static_cast<float*>(pts_buf.ptr);
    float* queries_ptr = static_cast<float*>(queries_buf.ptr);
    long* indices_ptr = static_cast<long*>(indices_buf.ptr);

    // Call appropriate C++ function
    if (omp) {
        cpp_knn_omp(pts_ptr, npts, dim, queries_ptr, nqueries, K, indices_ptr);
    } else {
        cpp_knn(pts_ptr, npts, dim, queries_ptr, nqueries, K, indices_ptr);
    }

    return indices;
}

py::array_t<long> knn_batch(py::array_t<float> pts, py::array_t<float> queries, int K, bool omp = false) {
    // Get array info
    py::buffer_info pts_buf = pts.request();
    py::buffer_info queries_buf = queries.request();

    if (pts_buf.ndim != 3 || queries_buf.ndim != 3)
        throw std::runtime_error("Number of dimensions must be 3");

    size_t batch_size = pts_buf.shape[0];
    size_t npts = pts_buf.shape[1];
    size_t dim = pts_buf.shape[2];
    size_t nqueries = queries_buf.shape[1];

    if (queries_buf.shape[0] != batch_size || queries_buf.shape[2] != dim)
        throw std::runtime_error("Points and queries dimensions mismatch");

    // Create output array
    py::array_t<long> indices({batch_size, nqueries, (size_t)K});
    py::buffer_info indices_buf = indices.request();

    // Get pointers to data
    float* pts_ptr = static_cast<float*>(pts_buf.ptr);
    float* queries_ptr = static_cast<float*>(queries_buf.ptr);
    long* indices_ptr = static_cast<long*>(indices_buf.ptr);

    // Call appropriate C++ function
    if (omp) {
        cpp_knn_batch_omp(pts_ptr, batch_size, npts, dim, queries_ptr, nqueries, K, indices_ptr);
    } else {
        cpp_knn_batch(pts_ptr, batch_size, npts, dim, queries_ptr, nqueries, K, indices_ptr);
    }

    return indices;
}

std::tuple<py::array_t<long>, py::array_t<float>> knn_batch_distance_pick(
    py::array_t<float> pts, int nqueries, int K, bool omp = false) {
    
    // Get array info
    py::buffer_info pts_buf = pts.request();

    if (pts_buf.ndim != 3)
        throw std::runtime_error("Number of dimensions must be 3");

    size_t batch_size = pts_buf.shape[0];
    size_t npts = pts_buf.shape[1];
    size_t dim = pts_buf.shape[2];

    // Create output arrays
    py::array_t<long> indices({batch_size, (size_t)nqueries, (size_t)K});
    py::array_t<float> queries({batch_size, (size_t)nqueries, dim});
    
    py::buffer_info indices_buf = indices.request();
    py::buffer_info queries_buf = queries.request();

    // Get pointers to data
    float* pts_ptr = static_cast<float*>(pts_buf.ptr);
    float* queries_ptr = static_cast<float*>(queries_buf.ptr);
    long* indices_ptr = static_cast<long*>(indices_buf.ptr);

    // Call appropriate C++ function
    if (omp) {
        cpp_knn_batch_distance_pick_omp(pts_ptr, batch_size, npts, dim, 
            queries_ptr, nqueries, K, indices_ptr);
    } else {
        cpp_knn_batch_distance_pick(pts_ptr, batch_size, npts, dim, 
            queries_ptr, nqueries, K, indices_ptr);
    }

    return std::make_tuple(indices, queries);
}

PYBIND11_MODULE(nearest_neighbors, m) {
    m.doc() = "KNN search implementation using nanoflann"; // optional module docstring

    m.def("knn", &knn, py::arg("pts"), py::arg("queries"), py::arg("K"), py::arg("omp") = false,
          "Perform k-nearest neighbor search");
          
    m.def("knn_batch", &knn_batch, py::arg("pts"), py::arg("queries"), py::arg("K"), py::arg("omp") = false,
          "Perform batch k-nearest neighbor search");
          
    m.def("knn_batch_distance_pick", &knn_batch_distance_pick, 
          py::arg("pts"), py::arg("nqueries"), py::arg("K"), py::arg("omp") = false,
          "Perform batch k-nearest neighbor search with distance picking");
}