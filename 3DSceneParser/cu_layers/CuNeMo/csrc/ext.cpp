#include "gather_features/gather_features.h"
#include "mask_weight/mask_weight.h"
#include "fast_init/fast_init.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_features", &GatherFeatures);
  m.def("gather_features_backward", &GatherFeaturesBackward);
  m.def("gather_idx", &GatherIdx);
  m.def("mask_weight", &MaskWeight);
  m.def("fast_feature_similarity", &FastFeatureSimilarity);
  m.def("aligned_fast_feature_similarity", &AlignedFastFeatureSimilarity);
  m.def("fast_score_collect", &FastScoreCollect);
  m.def("aligned_fast_score_collect", &AlignedFastScoreCollect);
}