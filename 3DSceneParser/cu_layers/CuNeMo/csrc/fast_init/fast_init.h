#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
at::Tensor FastFeatureSimilarity(
    const at::Tensor& kp_indexs, // (N, L, K_selected, )
    const at::Tensor& kp_locations, // (N, L, K_selected, 2) => (x, y)
    const at::Tensor& kp_features, // (L, K_padded, C)
    const at::Tensor& feature_maps, // (B, H, W, C)
    const at::Tensor& clutter_score // (B, H, W, )
);

at::Tensor AlignedFastFeatureSimilarity(
    const at::Tensor& kp_indexs, // (B, N, L, K_selected, )
    const at::Tensor& kp_locations, // (B, N, L, K_selected, 2) => (x, y)
    const at::Tensor& kp_features, // (L, K_padded, C)
    const at::Tensor& feature_maps, // (B, H, W, C)
    const at::Tensor& clutter_score // (B, H, W, )
);

at::Tensor FastScoreCollect(
    const at::Tensor& kp_indexs, // (B, L, N, K_selected, )
    const at::Tensor& kp_locations, // (B, L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_score, // (B, K_padded, H, W)
    const at::Tensor& clutter_score // (B, H, W, )
);

at::Tensor AlignedFastScoreCollect(
    const at::Tensor& kp_indexs, // (L, N, K_selected, )
    const at::Tensor& kp_locations, // (L, N, K_selected, 2) => (x, y)
    const at::Tensor& kp_score, // (B, K_padded, H, W)
    const at::Tensor& clutter_score // (B, H, W, )
);

#else
    AT_ERROR("Not compiled with GPU support");
#endif