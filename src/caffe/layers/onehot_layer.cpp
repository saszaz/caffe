#include <algorithm>
#include <vector>

#include "caffe/layers/onehot_layer.hpp"

using std::max;

namespace caffe {
  

template <typename Dtype>
	void OnehotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OnehotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int dim = this->layer_param_.inner_product_param().num_output();
  top[0]->Reshape(bottom[0]->num(), dim, 1, 1);  
}


template <typename Dtype>
void OnehotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int count = top[0]->count();
  const int dim = count / num;

  int label;
  memset(top_data, 0, sizeof(Dtype) * count);
  for (int i = 0; i < num; ++i) {
    label = static_cast<int>(bottom_data[i]);
    if (label >=0 && label < dim)
    //       top_data[i + num * label] = 1.;
      top_data[i * dim + label] = 1.;
    else
      LOG(FATAL) << "Label " << label << " outside of expected range [0," << dim-1 << "]";
  }

}


template <typename Dtype>
void OnehotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(OnehotLayer);
REGISTER_LAYER_CLASS(Onehot);


}  // namespace caffe
