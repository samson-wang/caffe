#include "caffe/common.hpp"
#include "caffe/util/rot.hpp"

namespace caffe {

template <typename Dtype>
__global__ void rot_pi_2_kernel(Dtype* F_, const Dtype* F, const int k, const int N_k, const int N,
        const bool acc, const int channel_out, const int channel_in, const bool spin) {
    const int ITER_COORD[16] = {0,0, 0,1, 0,2, 1,2, 2,2, 2,1, 2,0, 1,0};

    int idx = (threadIdx.x - 2*k) % D_N;
    idx = idx >= 0 ? idx : D_N + idx;
    idx = idx * 2;

    int weight_offset = channel_in * channel_out / N;
    int ci_idx = (threadIdx.y + blockIdx.x * blockDim.y) % weight_offset;
    int co_idx = (threadIdx.y + blockIdx.x * blockDim.y) / weight_offset;
    int offset = (co_idx * weight_offset + ci_idx) * K_W * K_W;
    int spin_offset = spin ? (((N + co_idx - N_k) % N * weight_offset + ci_idx) * K_W * K_W) : offset;

/*
    if (threadIdx.y + blockIdx.x * blockDim.y < channel_out * channel_in) {
    printf("ci %d, co %d, woff %d, (x, y, bx) (%d, %d, %d), (off, spin) (%d, %d) spin(%d, %d, %d)\n", ci_idx, co_idx, weight_offset, threadIdx.x, threadIdx.y, blockIdx.x, offset, spin_offset, N + co_idx - N_k, N_k, k);
    }
*/
//    printf("tid_y: %d, bid.x: %d, blockdim: %d, cindx: %d, %d\n", threadIdx.y, blockIdx.x, blockDim.y, threadIdx.y + blockIdx.x * blockDim.y, offset);
    if (threadIdx.y + blockIdx.x * blockDim.y < channel_out * channel_in) {
        if (acc) {
            F_[offset + ITER_COORD[threadIdx.x * 2 + 1] + ITER_COORD[threadIdx.x * 2] * K_W] +=
                F[spin_offset + ITER_COORD[idx + 1] + ITER_COORD[idx] * K_W];
        } else {
            F_[offset + ITER_COORD[threadIdx.x * 2 + 1] + ITER_COORD[threadIdx.x * 2] * K_W] =
                F[spin_offset + ITER_COORD[idx + 1] + ITER_COORD[idx] * K_W];

        }
        if (threadIdx.x == 0) {
            if (acc) {
                F_[offset + K_W * 1 + 1] += F[spin_offset + K_W * 1 + 1];
            } else {
                F_[offset + K_W * 1 + 1] = F[spin_offset + K_W * 1 + 1];
            }
        }

    }
}

template <typename Dtype>
__global__ void rot_pi_4_kernel(Dtype* F_, const Dtype* F, const int k, const int N_k, const int N,
        const bool acc, const int channel_out, const int channel_in, const bool spin) {
    const int ITER_COORD[16] = {0,0, 0,1, 0,2, 1,2, 2,2, 2,1, 2,0, 1,0};
    int ROT_COORD_1[] = {1,0, 0,1, 1,2, 2,1};
    int ROT_COORD_2[] = {1,0, 1,1, 0,0, 0,1,  1,1, 1,2, 0,1, 0,2,  2,1, 2,2, 1,1, 1,2,
                        2,0, 2,1, 1,0, 1,1};

    Dtype ROT_WEIGHT_1 = W_O;
    Dtype ROT_WEIGHT_2[] = {W_M, W_F, W_N, W_M,  W_F, W_M, W_M, W_N,  W_M, W_N, W_F, W_M,  W_N, W_M, W_M, W_F};
    int idx = (threadIdx.x - k + 1) % D_N;
    idx = idx >= 0 ? idx : idx + D_N;

    int weight_offset = channel_in * channel_out / N;
    int ci_idx = (threadIdx.y + blockIdx.x * blockDim.y) % weight_offset;
    int co_idx = (threadIdx.y + blockIdx.x * blockDim.y) / weight_offset;
    int offset = (co_idx * weight_offset + ci_idx) * K_W * K_W;
    int spin_offset = spin ? (((N + co_idx - N_k) % N * weight_offset + ci_idx) * K_W * K_W) : offset;

    Dtype tmp = 0.0;
    if (threadIdx.y + blockIdx.x * blockDim.y < channel_out * channel_in) {
        int y = ITER_COORD[threadIdx.x * 2];
        int x = ITER_COORD[threadIdx.x * 2 + 1];
        if (idx % 2 == 0) {
            tmp = F[spin_offset + ROT_COORD_1[idx+1] + ROT_COORD_1[idx] * K_W] * ROT_WEIGHT_1;
        } else {
            for (int m = 0; m < 4; m++) {
                tmp += F[spin_offset + ROT_COORD_2[idx/2 * 8 + m * 2 + 1] + ROT_COORD_2[idx/2 * 8 + m * 2] * K_W] * ROT_WEIGHT_2[idx/2 * 4 + m];
            }
        }
        if (acc) {
            F_[offset + x + y * K_W] += tmp;
        } else {
            F_[offset + x + y * K_W] = tmp;
        }
//        printf("TMP: %f\t", tmp);
        if (threadIdx.x == 0) {
            if (acc) {
                F_[offset + K_W * 1 + 1] += F[spin_offset + K_W * 1 + 1];
            } else {
                F_[offset + K_W * 1 + 1] = F[spin_offset + K_W * 1 + 1];
            }
        }

    }

}

template <typename Dtype>
__global__ void rot_pi_0_kernel(Dtype* F_, const Dtype* F, const int k,
        const bool acc, const int channel_out, const int channel_in) {
  int offset = (threadIdx.y + blockIdx.x * blockDim.y) * K_W * K_W;

  if (threadIdx.y + blockIdx.x * blockDim.y < channel_out * channel_in) {
    int y = threadIdx.x / 3;
    int x = threadIdx.x % 3;
    if (acc) {
      F_[offset + x + y * K_W] += F[offset + x + y * K_W];
    } else {
      F_[offset + x + y * K_W] = F[offset + x + y * K_W];
    }
    if (threadIdx.x == 0) {
      if (acc) {
        F_[offset + K_W * 2 + 2] += F[offset + K_W * 2 + 2];
      } else {
        F_[offset + K_W * 2 + 2] = F[offset + K_W * 2 + 2];
      }
    }

  }

}

template <typename Dtype>
void rot_weights_gpu(Dtype* F_, const Dtype* F, const int k, const int N, const bool acc,
    const int channel_out, const int channel_in, const int k_w, const bool spin) {
  int thread_dim_y = CAFFE_CUDA_NUM_THREADS / 8;
  int block_num = (channel_out * channel_in + thread_dim_y - 1) / thread_dim_y;

//  LOG(INFO) << block_num;
//  std::cout << "calling " << k << std::endl;

  if (k == 0) {
    rot_pi_0_kernel<Dtype><<<block_num, dim3(8, thread_dim_y)>>>(F_, F, k, acc, channel_out, channel_in);
    CUDA_POST_KERNEL_CHECK;
  } else if (N == 8 && abs(k) % 2 == 0) {
    rot_pi_2_kernel<Dtype><<<block_num, dim3(8, thread_dim_y)>>>(F_, F, k/2, k, N, acc, channel_out, channel_in, spin);
    CUDA_POST_KERNEL_CHECK;
  } else if (N == 8 && abs(k) % 2 == 1) {
    rot_pi_4_kernel<Dtype><<<block_num, dim3(8, thread_dim_y)>>>(F_, F, k, k, N, acc, channel_out, channel_in, spin);
    CUDA_POST_KERNEL_CHECK;
  } else if (N == 4) {
    rot_pi_2_kernel<Dtype><<<block_num, dim3(8, thread_dim_y)>>>(F_, F, k, k, N, acc, channel_out, channel_in, spin);
    CUDA_POST_KERNEL_CHECK;
  } else if (N == 2) {
    rot_pi_2_kernel<Dtype><<<block_num, dim3(8, thread_dim_y)>>>(F_, F, k*2, k, N, acc, channel_out, channel_in, spin);
    CUDA_POST_KERNEL_CHECK;
  }
}

template void rot_weights_gpu<float>(float* F_, const float* F, const int k, const int N,
    const bool acc, const int channel_out, const int channel_in, const int k_w, const bool spin=false);

template void rot_weights_gpu<double>(double* F_, const double* F, const int k, const int N,
    const bool acc, const int channel_out, const int channel_in, const int k_w, const bool spin=false);
} // namespace caffe
