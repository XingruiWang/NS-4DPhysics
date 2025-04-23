#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>

__device__ float vector_dot_four_weight(
    const float* s_start,
    const float* v_start_0,
    const float* v_start_1,
    const float* v_start_2,
    const float* v_start_3,
    const float w0,
    const float w1,
    const float w2,
    const float w3,
    const int size
){
    float out_val = 0;
    
    for (int i = 0; i < size; ++i){
        out_val += (v_start_0[i] * w0 + v_start_1[i] * w1 + v_start_2[i] * w2 + v_start_3[i] * w3) * s_start[i]; 
    }

    return out_val;
}


__global__ void FastFeatureSimilarityKernel(
    const int* kp_indexs, // (B if aligned else 0, L, N, K_selected, )
    const float* kp_locations, // (B if aligned else 0, L, N, K_selected, 2) => (x, y)
    const float* kp_features, // (L, K_padded, C)
    const float* feature_maps, // (B, H, W, C)
    const float* clutter_score, // (B, H, W, )
    const int aligned,
    const int B,
    const int H,
    const int W,
    const int C,
    const int N,
    const int L,
    const int K,
    const int K_pad,
    float * out_score // (B, L, N, K_selected)
){
    /*
    (xd, yd, w0) -------- (xu, yd, w1)
         |                     |
         |                     |
         |                     |
    (xd, yu, w3) -------- (xu, yu, w2)
    */
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < B * L * N * K; pid += num_threads) {
        const int b = pid / (L * N * K);
        const int l = (pid % (L * N * K)) / (N * K);
        const int n = (pid % (N * K)) / K;
        const int k = pid % K;

        const float x = kp_locations[2 * (b * L * N * K * aligned + l * N * K + n * K + k) + 0];
        const float y = kp_locations[2 * (b * L * N * K * aligned + l * N * K + n * K + k) + 1];

        const float x1 = floorf(x);
        const float x2 = ceilf(x);
        const float y1 = floorf(y);
        const float y2 = ceilf(y);

        const float w0 = (y - y1) * (x - x1);
        const float w1 = (y - y1) * (x2 - x);
        const float w2 = (y2 - y) * (x2 - x);
        const float w3 = (y2 - y) * (x - x1);

        const int xd = __float2int_rd(x);
        const int xu = __float2int_ru(x);
        const int yd = __float2int_rd(y);
        const int yu = __float2int_ru(y);

        const float clutter_ = clutter_score[b * H * W + yd * W + xd] * w0 + clutter_score[b * H * W + yd * W + xu] * w1 + clutter_score[b * H * W + yu * W + xu] * w2 + clutter_score[b * H * W + yu * W + xd] * w3;
        const float score_ = vector_dot_four_weight(
            kp_features + l * K_pad * C + kp_indexs[b * L * N * K * aligned + l * N * K + n * K + k] * C,
            feature_maps + (b * H * W + yd * W + xd) * C,
            feature_maps + (b * H * W + yd * W + xu) * C,
            feature_maps + (b * H * W + yu * W + xu) * C,
            feature_maps + (b * H * W + yu * W + xd) * C,
            w0, w1, w2, w3, C
        );

        out_score[pid] = fmaxf(score_, clutter_);
    }
}


at::Tensor FastFeatureSimilarity(
    const at::Tensor& kp_indexs, // (L, N, K_selected, )
    const at::Tensor& kp_locations, // (L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_features, // (L, K_padded, C)
    const at::Tensor& feature_maps, // (B, H, W, C)
    const at::Tensor& clutter_score // (B, H, W, )
){
    at::cuda::CUDAGuard device_guard(kp_features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int L = kp_locations.size(0);
    const int N = kp_locations.size(1);
    const int K = kp_locations.size(2);
    const int K_pad = kp_features.size(1);
    const int B = feature_maps.size(0);
    const int H = feature_maps.size(1);
    const int W = feature_maps.size(2);
    const int C = feature_maps.size(3);

    auto float_opts = feature_maps.options().dtype(at::kFloat);
    auto int_opts = feature_maps.options().dtype(at::kInt);

    at::Tensor out_score = at::zeros({B, L, N, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    FastFeatureSimilarityKernel<<<blocks, threads, 0, stream>>>(
        kp_indexs.contiguous().data_ptr<int>(), // (L, N, K_padded, 2) => (x, y)
        kp_locations.contiguous().data_ptr<float>(), // (L, N, K_padded, 2) => (x, y)
        kp_features.contiguous().data_ptr<float>(), // (L, K_padded, C)
        feature_maps.contiguous().data_ptr<float>(), // (B, H, W, C)
        clutter_score.contiguous().data_ptr<float>(), // (B, H, W, )
        0, 
        B,
        H,
        W,
        C,
        N,
        L,
        K,
        K_pad,
        out_score.data_ptr<float>() // (B, L, N, K_padded)
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    return out_score;
}


at::Tensor AlignedFastFeatureSimilarity(
    const at::Tensor& kp_indexs, // (B, L, N, K_selected, )
    const at::Tensor& kp_locations, // (B, L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_features, // (L, K_padded, C)
    const at::Tensor& feature_maps, // (B, H, W, C)
    const at::Tensor& clutter_score // (B, H, W, )
){
    at::cuda::CUDAGuard device_guard(kp_features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int B = kp_locations.size(0);
    const int L = kp_locations.size(1);
    const int N = kp_locations.size(2);
    const int K = kp_locations.size(3);
    const int K_pad = kp_features.size(1);
    const int H = feature_maps.size(1);
    const int W = feature_maps.size(2);
    const int C = feature_maps.size(3);

    auto float_opts = feature_maps.options().dtype(at::kFloat);
    auto int_opts = feature_maps.options().dtype(at::kInt);

    at::Tensor out_score = at::zeros({B, L, N, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    FastFeatureSimilarityKernel<<<blocks, threads, 0, stream>>>(
        kp_indexs.contiguous().data_ptr<int>(), // (B, L, N, K_padded, 2) => (x, y)
        kp_locations.contiguous().data_ptr<float>(), // (B, L, N, K_padded, 2) => (x, y)
        kp_features.contiguous().data_ptr<float>(), // (K_padded, C)
        feature_maps.contiguous().data_ptr<float>(), // (B, H, W, C)
        clutter_score.contiguous().data_ptr<float>(), // (B, H, W, )
        1,
        B,
        H,
        W,
        C,
        N,
        L,
        K,
        K_pad,
        out_score.data_ptr<float>() // (B, L, N, K_padded)
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    return out_score;
}


__global__ void FastScoreCollectKernel(
    const int* kp_indexs, // (B if aligned else 0, L, N, K_selected, )
    const float* kp_locations, // (B if aligned else 0, L, N, K_selected, 2) => (x, y)
    const float* kp_score, // (B, K_padded, H, W)
    const float* clutter_score, // (B, H, W, )
    const int aligned,
    const int B,
    const int H,
    const int W,
    const int N,
    const int L,
    const int K,
    const int K_pad,
    float * out_score // (B, L, N, K_selected)
){
    /*
    (xd, yd, w0) -------- (xu, yd, w1)
         |                     |
         |                     |
         |                     |
    (xd, yu, w3) -------- (xu, yu, w2)
    */
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < B * L * N * K; pid += num_threads) {
        const int b = pid / (L * N * K);
        const int l = (pid % (L * N * K)) / (N * K);
        const int n = (pid % (N * K)) / K;
        const int k = pid % K;

        const float x = kp_locations[2 * (b * L * N * K * aligned + l * N * K + n * K + k) + 0];
        const float y = kp_locations[2 * (b * L * N * K * aligned + l * N * K + n * K + k) + 1];

        const float x1 = floorf(x);
        const float x2 = ceilf(x);
        const float y1 = floorf(y);
        const float y2 = ceilf(y);

        const float w0 = (y - y1) * (x - x1);
        const float w1 = (y - y1) * (x2 - x);
        const float w2 = (y2 - y) * (x2 - x);
        const float w3 = (y2 - y) * (x - x1);

        const int xd = __float2int_rd(x);
        const int xu = __float2int_ru(x);
        const int yd = __float2int_rd(y);
        const int yu = __float2int_ru(y);

        const int c_base_ = b * H * W;
        const int k_base_ = b * H * W * K_pad + kp_indexs[b * L * N * K * aligned + l * N * K + n * K + k] * H * W;

        const float clutter_ = clutter_score[c_base_ + yd * W + xd] * w0 + clutter_score[c_base_ + yd * W + xu] * w1 + clutter_score[c_base_ + yu * W + xu] * w2 + clutter_score[c_base_ + yu * W + xd] * w3;
        const float score_ = kp_score[k_base_ + yd * W + xd] * w0 + kp_score[k_base_ + yd * W + xu] * w1 + kp_score[k_base_ + yu * W + xu] * w2 + kp_score[k_base_ + yu * W + xd] * w3;

        out_score[pid] = fmaxf(score_, clutter_);
    }
}


at::Tensor FastScoreCollect(
    const at::Tensor& kp_indexs, // (B, L, N, K_selected, )
    const at::Tensor& kp_locations, // (B, L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_score, // (B, K_padded, H, W)
    const at::Tensor& clutter_score // (B, H, W, )
){
    at::cuda::CUDAGuard device_guard(kp_indexs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int L = kp_locations.size(0);
    const int N = kp_locations.size(1);
    const int K = kp_locations.size(2);
    const int B = kp_score.size(0);
    const int K_pad = kp_score.size(1);
    const int H = kp_score.size(2);
    const int W = kp_score.size(3);

    auto float_opts = kp_score.options().dtype(at::kFloat);
    auto int_opts = kp_score.options().dtype(at::kInt);

    at::Tensor out_score = at::zeros({B, L, N, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    FastScoreCollectKernel<<<blocks, threads, 0, stream>>>(
        kp_indexs.contiguous().data_ptr<int>(), // (L, N, K_padded, 2) => (x, y)
        kp_locations.contiguous().data_ptr<float>(), // (L, N, K_padded, 2) => (x, y)
        kp_score.contiguous().data_ptr<float>(), // (B, K_padded, H, W)
        clutter_score.contiguous().data_ptr<float>(), // (B, H, W, )
        0, 
        B,
        H,
        W,
        N,
        L,
        K,
        K_pad,
        out_score.data_ptr<float>() // (B, L, N, K_padded)
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    return out_score;
}


at::Tensor AlignedFastScoreCollect(
    const at::Tensor& kp_indexs, // (L, N, K_selected, )
    const at::Tensor& kp_locations, // (L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_score, // (B, K_padded, H, W)
    const at::Tensor& clutter_score // (B, H, W, )
){
    at::cuda::CUDAGuard device_guard(kp_indexs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int L = kp_locations.size(1);
    const int N = kp_locations.size(2);
    const int K = kp_locations.size(3);
    const int B = kp_score.size(0);
    const int K_pad = kp_score.size(1);
    const int H = kp_score.size(2);
    const int W = kp_score.size(3);

    auto float_opts = kp_score.options().dtype(at::kFloat);
    auto int_opts = kp_score.options().dtype(at::kInt);

    at::Tensor out_score = at::zeros({B, L, N, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    FastScoreCollectKernel<<<blocks, threads, 0, stream>>>(
        kp_indexs.contiguous().data_ptr<int>(), // (L, N, K_padded, 2) => (x, y)
        kp_locations.contiguous().data_ptr<float>(), // (L, N, K_padded, 2) => (x, y)
        kp_score.contiguous().data_ptr<float>(), // (B, K_padded, H, W)
        clutter_score.contiguous().data_ptr<float>(), // (B, H, W, )
        1, 
        B,
        H,
        W,
        N,
        L,
        K,
        K_pad,
        out_score.data_ptr<float>() // (B, L, N, K_padded)
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    return out_score;
}