#ifndef _CAFFE_UTIL_ROT_HPP_
#define _CAFFE_UTIL_ROT_HPP_
#include <vector>
namespace caffe {

#define D_N 8
#define K_W 3

#define W_N 0.5
#define W_M 0.2071067811865475449817353137405007146298885345458984375
#define W_F 0.0857864376269049933032562194057391025125980377197265625
#define W_O 0.58578643762690507656998306629247963428497314453125
template <typename Dtype>
void rot_weights_gpu(Dtype* F_, const Dtype* F, const int k, const int N, const bool acc, 
    const int channel_out, const int channel_in, const int k_w, const bool spin=false);


template <typename Dtype>
void rot_pi_4(Dtype* F_, const Dtype* F, int K, bool acc);
template <typename Dtype>
void rot_pi_2(Dtype* F_, const Dtype* F, int K, bool acc);
template <typename Dtype>
void rot_weights(Dtype* rot, const Dtype* weights, std::vector<int> const weight_shape, int K, int N, bool acc);

} // namespace caffe
#endif  // CAFFE_UTIL_ROT_HPP_
