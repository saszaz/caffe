#include <cfloat>

#include "caffe/layers/flow_warping_layer.hpp"

namespace caffe {

template <typename Dtype>
void FlowWarpingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  vector<int> ones_size(1, channels_);
  scale_blob_.Reshape(ones_size);
  scale_ = this->layer_param().flow_warping_param().scale();
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(channels_, bottom[0]->channels());
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  CHECK_EQ(height_, bottom[1]->height()) << "disp height = " << bottom[1]->height() << " is not equal to image height = " << height_;
  CHECK_EQ(width_, bottom[1]->width()) << "disp width = " << bottom[1]->width() << " is not equal to image width = " << width_;
  
  vector<int> size_nz;
  size_nz.push_back(bottom[0]->num());
  size_nz.push_back(width_ * height_ * 4);
  
  vector<int> size_ptrs;
  size_ptrs.push_back(bottom[0]->num());
  size_ptrs.push_back(width_ * height_ + 1);
  
  interp_coefs_blob_.Reshape(size_nz);
  partial_i_blob_.Reshape(size_nz);
  partial_j_blob_.Reshape(size_nz);
  indices_blob_.Reshape(size_nz);
  ptrs_blob_.Reshape(size_ptrs);
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void FlowWarpingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FlowWarpingLayer);
#endif

INSTANTIATE_CLASS(FlowWarpingLayer);
REGISTER_LAYER_CLASS(FlowWarping);

}  // namespace caffe
