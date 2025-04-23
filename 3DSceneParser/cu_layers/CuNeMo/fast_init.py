import torch
from CuNeMo import _C


def check_dims(dims, *args):
    return all([arg_.dim() == dim_ for dim_, arg_ in zip(dims, args) if arg_ is not None])
         

def fast_feature_similarity(kp_locations, kp_features, feature_maps, clutter_score=None, kp_indexs=None):
    """
    kp_indexs (B, L, N, K_selected, ) or (B, N, K_selected, ) or (L, N, K_selected, ) or (N, K_selected, )
    kp_locations (B, L, N, K_selected, 2) or (B, N, K_selected, 2) or (L, N, K_selected, 2) or (N, K_selected, 2)
    Return (B, L, N, K_selected)
    """
    assert check_dims([3, 2, 4, 3, 2], kp_locations, kp_features, feature_maps, clutter_score, kp_indexs) \
        or check_dims([4, 3, 4, 3, 3], kp_locations, kp_features, feature_maps, clutter_score, kp_indexs) \
        or check_dims([4, 2, 4, 3, 3], kp_locations, kp_features, feature_maps, clutter_score, kp_indexs) \
        or check_dims([5, 3, 4, 3, 4], kp_locations, kp_features, feature_maps, clutter_score, kp_indexs) 

    if kp_features.dim() == 2:
        kp_features = kp_features[None]
        kp_locations = kp_locations[:, None] if kp_locations.dim() == 4 else kp_locations[None]

    if kp_indexs is None:
        kp_indexs = torch.arange(kp_locations.shape[-2], device=kp_locations.device).type(torch.int32)[None, None].expand(*kp_locations.shape[:-2], -1)
    elif kp_indexs.dim() == kp_locations.dim() - 2:
        kp_indexs = kp_indexs[:, None]

    if clutter_score is None:
        clutter_score = -torch.ones(feature_maps.shape[:3], device=feature_map.device)

    assert kp_locations.is_cuda
    assert kp_locations.device == kp_features.device == feature_maps.device == clutter_score.device == kp_indexs.device
    
    if check_dims([5, 3, 4, 3, 4], kp_locations, kp_features, feature_maps, clutter_score, kp_indexs):
        return _AlignedFastFeatureSimilarity.apply(kp_indexs, kp_locations, kp_features, feature_maps, clutter_score)
    else:
        return _FastFeatureSimilarity.apply(kp_indexs, kp_locations, kp_features, feature_maps, clutter_score)


class _AlignedFastFeatureSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                kp_indexs, # (B, L, N, K_selected, )
                kp_locations, # (B, L, N, K_selected, 2) => (x, y)
                kp_features, # (L, K_padded, C)
                feature_maps, # (B, H, W, C)
                clutter_score # (B, H, W, )
        ): # => (B, L, N, K_selected)
        args = (
                kp_indexs, 
                kp_locations, 
                kp_features, 
                feature_maps, 
                clutter_score 
        )
        out_score = _C.aligned_fast_feature_similarity(*args)
        ctx.mark_non_differentiable(out_score)
        return out_score


class _FastFeatureSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                kp_indexs, # (L, N, K_selected, )
                kp_locations, # (L, N, K_selected, 2) => (x, y)
                kp_features, # (L, K_padded, C)
                feature_maps, # (B, H, W, C)
                clutter_score # (B, H, W, )
        ): # => (B, L, N, K_selected)
        args = (
                kp_indexs, 
                kp_locations, 
                kp_features,
                feature_maps,
                clutter_score
        )

        out_score = _C.fast_feature_similarity(*args)
        ctx.mark_non_differentiable(out_score)
        return out_score


def fast_collect_score(kp_locations, kp_score, clutter_score=None, kp_indexs=None):
    """
    kp_indexs (B, L, N, K_selected, ) or (B, N, K_selected, ) or (L, N, K_selected, ) or (N, K_selected, )
    kp_locations (B, L, N, K_selected, 2) or (B, N, K_selected, 2) or (L, N, K_selected, 2) or (N, K_selected, 2)
    Return (B, L, N, K_selected)
    """
    assert check_dims([3, 4, 3, 2], kp_locations, kp_score, clutter_score, kp_indexs) \
        or check_dims([4, 4, 3, 3], kp_locations, kp_score, clutter_score, kp_indexs) \
        or check_dims([5, 4, 3, 4], kp_locations, kp_score, clutter_score, kp_indexs) 

    if kp_locations.dim() == 4 and kp_locations.shape[0] == kp_score.shape[0]:
        kp_locations = kp_locations[:, None]

    if kp_locations.dim() == 3:
        kp_locations = kp_locations[None]

    if kp_indexs is None:
        kp_indexs = torch.arange(kp_locations.shape[-2], device=kp_locations.device).type(torch.int32)[None, None].expand(*kp_locations.shape[:-2], -1)
    elif kp_indexs.dim() == kp_locations.dim() - 2:
        kp_indexs = kp_indexs[:, None]

    if clutter_score is None:
        clutter_score = -torch.ones((kp_score.shape[0], kp_score.shape[2], kp_score.shape[3]), device=feature_map.device)

    assert kp_locations.is_cuda
    assert kp_locations.device == kp_score.device == clutter_score.device == kp_indexs.device
    
    if check_dims([5, 4, 3, 4], kp_locations, kp_score, clutter_score, kp_indexs):
        return _AlignedFastScoreCollect.apply(kp_indexs, kp_locations, kp_score, clutter_score)
    else:
        return _FastScoreCollect.apply(kp_indexs, kp_locations, kp_score, clutter_score)


class _AlignedFastScoreCollect(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                kp_indexs, # (B, L, N, K_selected, )
                kp_locations, # (B, L, N, K_selected, 2) => (x, y)
                kp_score, # (B, K_padded, H, W)
                clutter_score # (B, H, W, )
        ): # => (B, L, N, K_selected)
        args = (
                kp_indexs, 
                kp_locations, 
                kp_score,
                clutter_score 
        )
        out_score = _C.aligned_fast_score_collect(*args)
        ctx.mark_non_differentiable(out_score)
        return out_score


class _FastScoreCollect(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                kp_indexs, # (L, N, K_selected, )
                kp_locations, # (L, N, K_selected, 2) => (x, y)
                kp_score, # (B, K_padded, H, W)
                clutter_score # (B, H, W, )
        ): # => (B, L, N, K_selected)
        args = (
                kp_indexs, 
                kp_locations, 
                kp_score,
                clutter_score
        )

        out_score = _C.fast_score_collect(*args)
        ctx.mark_non_differentiable(out_score)
        return out_score


if __name__ == '__main__':
    kp_locations = torch.rand((1, 8, 2, 5, 2)).cuda() * 9
    kp_features = torch.ones((2, 5, 3)).cuda()
    feature_maps = torch.ones((1, 10, 10, 3)).cuda()
    clutter_score = torch.zeros((1, 10, 10, )).cuda()
    print(fast_feature_similarity(kp_locations, kp_features, feature_maps, clutter_score).shape)