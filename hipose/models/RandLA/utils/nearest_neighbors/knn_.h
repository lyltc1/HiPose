#ifndef KNN_H
#define KNN_H

#include <cstddef> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

// Basic KNN search
void cpp_knn(const float* points, size_t npts, size_t dim, 
             const float* queries, size_t nqueries,
             size_t K, long* indices);

// OpenMP parallel version
void cpp_knn_omp(const float* points, size_t npts, size_t dim,
                const float* queries, size_t nqueries,
                size_t K, long* indices);

// Batch KNN search
void cpp_knn_batch(const float* batch_data, size_t batch_size, 
                  size_t npts, size_t dim,
                  const float* queries, size_t nqueries,
                  size_t K, long* batch_indices);

// OpenMP parallel batch version
void cpp_knn_batch_omp(const float* batch_data, size_t batch_size,
                      size_t npts, size_t dim,
                      const float* queries, size_t nqueries,
                      size_t K, long* batch_indices);

// Distance-based KNN pick
void cpp_knn_batch_distance_pick(const float* batch_data, size_t batch_size,
                                size_t npts, size_t dim,
                                float* queries, size_t nqueries,
                                size_t K, long* batch_indices);

// OpenMP parallel distance-based version
void cpp_knn_batch_distance_pick_omp(const float* batch_data, size_t batch_size,
                                    size_t npts, size_t dim,
                                    float* batch_queries, size_t nqueries,
                                    size_t K, long* batch_indices);

#ifdef __cplusplus
}
#endif

#endif // KNN_H