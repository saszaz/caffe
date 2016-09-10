#ifndef EKF_FLOW_WARPING_LAYER_HPP_
#define EKF_FLOW_WARPING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {

template <typename Dtype>
class FlowWarpingLayer : public Layer<Dtype> {
 public:
  explicit FlowWarpingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FlowWarping"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void ForwardJv_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void ForwardJv_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);  
  int channels_;
  int height_;
  int width_;
  Dtype scale_;

  Blob<Dtype> partial_i_blob_;
  Blob<Dtype> partial_j_blob_;
  Blob<Dtype> interp_coefs_blob_;
  Blob<int> indices_blob_;
  Blob<int> ptrs_blob_;
  Blob<Dtype> scale_blob_;
};
}  // namespace caffe

#endif  // EKF_FLOW_WARPING_LAYER_HPP_
