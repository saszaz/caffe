#ifndef CAFFE_UTIL_EKF_MATH_H_
#define CAFFE_UTIL_EKF_MATH_H_

#include <stdint.h>

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"
 
namespace caffe {

template<typename Dtype>
void tracker_gpu_csr_gemm_cusparse(const CBLAS_TRANSPOSE TransA,
                          const CBLAS_TRANSPOSE TransB, const int M, const int N,
                          const int K, const Dtype alpha, int nzz, const Dtype* A,
                          const int* indices, const int* ptr, const Dtype* B,
                          const Dtype beta, Dtype* C, const CBLAS_ORDER orderC);

}  // namespace caffe

#endif  // CAFFE_UTIL_EKF_MATH_H_
