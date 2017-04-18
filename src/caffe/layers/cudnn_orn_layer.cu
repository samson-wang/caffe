#ifdef USE_CUDNN
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cudnn_orn_layer.hpp"

namespace caffe {

inline int rot_index(int k, int N) {
  k = k % N;
  k = k >= 0 ? k : k + N;
  return k;
}

__global__ void sync_orn_conv_groups() { }

template <typename Dtype>
void CuDNNOrientedConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
//    caffe_gpu_set(top[i]->count(), static_cast<Dtype>(0), top_data);

    for (int n = 0; n < this->group_; ++n) {
      rot_weights_gpu(this->rot_blobs_[0]->mutable_gpu_data(), this->blobs_[0]->gpu_data(),
          n, this->group_, n ? true : false, this->blobs_[0]->shape()[0], this->blobs_[0]->shape()[1], 3, true);
    }
      const Dtype* rot = this->rot_blobs_[0]->gpu_data();
      // Forward through cuDNN in parallel over groups.
      for (int g = 0; g < this->group_; g++) {
        // Filters.
        CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              filter_desc_, rot + this->weight_offset_ * g,
              conv_descs_[i],
              fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              top_descs_[i], top_data + top_offset_ * g));

        // Bias.
        if (this->bias_term_) {
          const Dtype* bias_data = this->blobs_[1]->gpu_data();
          CUDNN_CHECK(cudnnAddTensor(handle_[g],
                cudnn::dataType<Dtype>::one,
                bias_desc_, bias_data + bias_offset_ * g,
                cudnn::dataType<Dtype>::one,
                top_descs_[i], top_data + top_offset_ * g));
        }
      }

      // Synchronize the work across groups, each of which went into its own
      // stream, by launching an empty kernel into the default (null) stream.
      // NOLINT_NEXT_LINE(whitespace/operators)
      sync_orn_conv_groups<<<1, 1>>>();
    
  }
}

template <typename Dtype>
void CuDNNOrientedConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
/*
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }
*/
      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        const Dtype* rot = this->rot_blobs_[0]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, rot + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    for (int n = 0; n < 1; ++n) {
//      Dtype* rot_diff = this->rot_blobs_[0]->mutable_gpu_diff();
      for (int g = 0; g < this->group_; ++g) {
        // Gradient w.r.t. weights.
        if (this->param_propagate_down_[0]) {
          const Dtype* bottom_data = bottom[i]->gpu_data();
          CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle_[1*this->group_ + g],
                cudnn::dataType<Dtype>::one,
                bottom_descs_[i], bottom_data + bottom_offset_ * (g),
                top_descs_[i],    top_diff + top_offset_ * g,
                conv_descs_[i],
                bwd_filter_algo_[i], workspace[1*this->group_ + g],
                workspace_bwd_filter_sizes_[i],
                cudnn::dataType<Dtype>::one,
                filter_desc_, weight_diff + this->weight_offset_ * g));
        }
      }
      sync_orn_conv_groups<<<1, 1>>>();
      /*
      if (this->param_propagate_down_[0]) {
        rot_weights_gpu(weight_diff, rot_diff, -n, this->group_, true, this->blobs_[0]->shape()[0],
            this->blobs_[0]->shape()[1], 3, false);
      }
      */

    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_orn_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNOrientedConvolutionLayer);

}  // namespace caffe
#endif
