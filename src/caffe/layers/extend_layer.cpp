#include <vector>

#include "caffe/layers/extend_layer.hpp"

namespace caffe {

template <typename Dtype>
void ExtendLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  group_ = this->layer_param_.convolution_param().group();
  LOG(INFO) << "GROUP in Extend " << group_;
}

template <typename Dtype>
void ExtendLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[1] = group_ * shape[1];
  top[0]->Reshape(shape);
}

template <typename Dtype>
void ExtendLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int n = 0; n < bottom[0]->shape()[0]; ++n) {
    for (int g = 0; g < group_; ++g) {
      caffe_copy(bottom[0]->count(1), bottom_data + n*bottom[0]->count(1), top_data + n*group_*bottom[0]->count(1) + g * bottom[0]->count(1));
    }
  }
}

template <typename Dtype>
void ExtendLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  for (int n = 0; n < bottom[0]->shape()[0]; ++n) {
    for (int g = 0; g < group_; ++g) {
      caffe_copy(bottom[0]->count(1), bottom_data + n*bottom[0]->count(1), top_data + n*group_*bottom[0]->count(1) + g * bottom[0]->count(1));
    }
  }
}

INSTANTIATE_CLASS(ExtendLayer);
REGISTER_LAYER_CLASS(Extend);

}  // namespace caffe
