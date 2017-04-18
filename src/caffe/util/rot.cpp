#include <vector>
#include "caffe/util/rot.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
static const std::vector<std::vector<int> > ITER_COORD_ = {{0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 2}, {2, 1}, {2, 0}, {1, 0}};
static const std::vector<std::vector<std::vector<int> > > ROT_COORD_ = {
        {{1, 0}},
        {{1, 0}, {1, 1}, {0, 0}, {0, 1}},
        {{0, 1}},
        {{1, 1}, {1, 2}, {0, 1}, {0, 2}},
        {{1, 2}},
        {{2, 1}, {2, 2}, {1, 1}, {1, 2}},
        {{2, 1}}, 
        {{2, 0}, {2, 1}, {1, 0}, {1, 1}}
      };
static const std::vector<std::vector<double> > ROT_WEIGHT_ = {
        {W_O},
        {W_M, W_F, W_N, W_M},
        {W_O},
        {W_F, W_M, W_M, W_N},
        {W_O},
        {W_M, W_N, W_F, W_M},
        {W_O},
        {W_N, W_M, W_M, W_F}
      };

inline int rot_index(int k, int N) {
  k = k % N;
  k = k >= 0 ? k : k + N;
  return k;
}

template <typename Dtype>
void rot_pi_4(Dtype* F_,
      const Dtype* F, int K, bool acc) {

    const int N = 8;
    const int w_ = 3;
//    assert(F.size() == F_.size());
    if (acc) {
      F_[1 * w_ + 1] += F[1 * w_ + 1];
    } else {
      F_[1 * w_ + 1] = F[1 * w_ + 1];
    }
    for (int k = 0; k < ITER_COORD_.size(); ++k) {
        int y = ITER_COORD_[k][0];
        int x = ITER_COORD_[k][1];

        Dtype tmp = 0;
        int rot_idx = rot_index((k - K + 1), N);
        for (int m = 0; m < ROT_COORD_[rot_idx].size(); ++m) {
            int j = ROT_COORD_[rot_idx][m][0];
            int i = ROT_COORD_[rot_idx][m][1];
            tmp += F[i + j*w_] * ROT_WEIGHT_[rot_idx][m];
        }
        if (acc == false) {
          F_[x + y*w_] = tmp;
        } else {
          F_[x + y*w_] += tmp;
        }
    }

}

template <typename Dtype>
void rot_pi_2(Dtype* F_,
      const Dtype* F, int K, bool acc) {
    const int N = 8;
    const int w_ = 3;
    if (acc) {
      F_[1 + 1 * w_] += F[1 + 1 * w_];
    } else {
      F_[1 + 1 * w_] = F[1 + 1 * w_];
    }
    for (int k = 0; k < ITER_COORD_.size(); ++k) {
        int y = ITER_COORD_[k][0];
        int x = ITER_COORD_[k][1];

        int Y = ITER_COORD_[rot_index((k - 2*K), N)][0];
        int X = ITER_COORD_[rot_index((k - 2*K), N)][1];
        if (acc == false) {
          F_[x + y*w_] = F[X + Y*w_];
        } else {
          F_[x + y*w_] += F[X + Y*w_];
        }
    }

}

template <typename Dtype>
void rot_weights(Dtype* rot, const Dtype* weights,
    std::vector<int> const weight_shape, int K, int N, bool acc) {
//  const Dtype* weights = this->blobs_[0]->cpu_data();
//  vector<int> weight_shape(this->blobs_[0]->shape());
//  CHECK_EQ(weight_shape.size(), 4);
//  CHECK_EQ(weight_shape[2], 3);

//  Dtype* rot = this->rot_blobs_[0]->mutable_cpu_data();

  int dim = weight_shape[2] * weight_shape[3];

  // Rotate kernel
  if (K == 0) {
    for (int i = 0; i < dim * weight_shape[0] * weight_shape[1]; i++) {
      if (acc) {
        rot[i] += weights[i];
      } else {
        rot[i] = weights[i];
      }
    }
  } else {
    for (int c_out = 0; c_out < weight_shape[0]; c_out++) {
      for (int c_in = 0; c_in < weight_shape[1]; c_in++) {
        if (N == 4) {
          rot_pi_2(rot+c_in*dim+c_out*weight_shape[1]*dim,
                    weights+c_in*dim+c_out*weight_shape[1]*dim, K, acc);
        } else if (N == 8 && abs(K) % 2 == 0) {
          rot_pi_2(rot+c_in*dim+c_out*weight_shape[1]*dim,
                    weights+c_in*dim+c_out*weight_shape[1]*dim, K/2, acc);
        } else if (N == 8 && abs(K) % 2 == 1) {
          rot_pi_4(rot+c_in*dim+c_out*weight_shape[1]*dim,
                    weights+c_in*dim+c_out*weight_shape[1]*dim, K, acc);
        } else if (N == 2) {
          rot_pi_2(rot+c_in*dim+c_out*weight_shape[1]*dim,
                    weights+c_in*dim+c_out*weight_shape[1]*dim, K*2, acc);
        }
      }
    }
  }
}

template 
void rot_weights<float>(float* rot, const float* weights,
    std::vector<int> const weight_shape, int K, int N, bool acc);

template 
void rot_weights<double>(double* rot, const double* weights,
    std::vector<int> const weight_shape, int K, int N, bool acc);
}  // namespace caffe
