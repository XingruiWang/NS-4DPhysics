# Author: Angtian Wang
# Adding support for batch operation 
# Perform in original NeMo manner 
# Support 3D pose as NeMo and VoGE-NeMo, 


import numpy as np
import torch
from pytorch3d.renderer import camera_position_from_spherical_angles
from nemo.utils import construct_class_by_name
from nemo.utils import camera_position_to_spherical_angle
from nemo.utils.general import tensor_linspace
import time

try:
    from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, interpolate_attr
    from VoGE.Utils import Batchifier
    enable_voge = True
except:
    enable_voge=False

if not enable_voge:
    from TorchBatchifier import Batchifier


class StandardAnnoFilter():
    use_anno = {
        3: ['principal', 'distance'],
        4: ['principal'],
        6: []
    }

    def __init__(self, dof, device):
        self.dof = dof
        self.device = device

    def __call__(self, samples):
        return {kk: samples[kk].to(self.device).float() if torch.is_tensor(samples[kk]) else samples[kk] for kk in self.use_anno[self.dof]}


def loss_fg_only(obj_s, clu_s=None, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - reduce_method(obj_s)


def loss_fg_bg(obj_s, clu_s, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    )

def object_loss_fg_bg(obj_s, clu_s, object_height, object_width, reduce_method=lambda x: torch.mean(x)):
    obj_s = obj_s[object_height[0]:object_height[1], object_width[0]:object_width[1]]
    clu_s = clu_s[object_height[0]:object_height[1], object_width[0]:object_width[1]]
    return torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    )


def solve_pose(
    cfg,
    feature_map,
    inter_module,
    proposal_module,
    clutter_bank=None,
    device="cuda",
    principal=None,
    distance=None,
    dof=3,
    **kwargs
):
    b, c, hm_h, hm_w = feature_map.size()
    pred = {}

    # Step 1: Pre-compute foreground and background features
    start_time = time.time()
    clutter_score = None
    if not clutter_bank is None:
        if not isinstance(clutter_bank, list):
            clutter_bank = [clutter_bank]
        for cb in clutter_bank:
            _score = (
                torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3))
                .squeeze(1)
            )
            if clutter_score is None:
                clutter_score = _score
            else:
                clutter_score = torch.max(clutter_score, _score)

    end_time = time.time()
    pred["pre_compute_time"] = end_time - start_time

    # Step 2: Search for initializations
    start_time = end_time
    print(principal)
    init_C, init_theta, init_principal = proposal_module(feature_map=feature_map, clutter_score=clutter_score, principal=principal, distance=distance, **kwargs)
    
    if dof == 6 or principal is not None:
        inter_module.rasterizer.cameras._N = feature_map.shape[0]

    end_time = time.time()
    pred["pre_rendering_time"] = end_time - start_time

    # Step 3: Refine object proposals with pose optimization
    start_time = end_time
    
    C = torch.nn.Parameter(init_C, requires_grad=True)
    theta = torch.nn.Parameter(init_theta, requires_grad=True)
    if dof == 6 or cfg.get('optimize_translation', False):
        principals = torch.nn.Parameter(init_principal, requires_grad=True)
        inter_module.rasterizer.cameras.principal_point = principals
        optim = construct_class_by_name(**cfg.inference.optimizer, params=[C, theta, principals])
    else:
        principals = init_principal.expand(b, -1) if init_principal.shape[0] == 1 else init_principal
        inter_module.rasterizer.cameras.principal_point = init_principal
        optim = construct_class_by_name(**cfg.inference.optimizer, params=[C, theta])

    scheduler_kwargs = {"optimizer": optim}
    scheduler = construct_class_by_name(**cfg.inference.scheduler, **scheduler_kwargs)
    
    for epo in range(cfg.inference.epochs):
        # [b, c, h, w]
        projected_map = inter_module(
            C,
            theta,
            mode=cfg.inference.inter_mode,
            blur_radius=cfg.inference.blur_radius,
        )

        # [b, c, h, w] -> [b, h, w]
        object_score = torch.sum(projected_map * feature_map, dim=1)
        import ipdb; ipdb.set_trace()
        loss = loss_fg_bg(object_score, clutter_score, )
        loss.backward()
        optim.step()
        optim.zero_grad()

        if (epo + 1) % max(cfg.inference.epochs // 3, 1) == 0:
            scheduler.step()

    distance_preds, elevation_preds, azimuth_preds = camera_position_to_spherical_angle(C)
    pred["optimization_time"] = end_time - start_time

    preds = []

    for i in range(b):
        theta_pred, distance_pred, elevation_pred, azimuth_pred = (
            theta[i].item(),
            distance_preds[i].item(),
            elevation_preds[i].item(),
            azimuth_preds[i].item(),
        )
        this_principal = principals[i]
        with torch.no_grad():
            if kwargs.get('bbox', None) is not None:
                object_height = (int(bbox[i][0].item() / kwargs.get('down_sample_rate', 8) + .5), int(bbox[i][1].item() / kwargs.get('down_sample_rate', 8) + .5))
                object_width = (int(bbox[i][2].item() / kwargs.get('down_sample_rate', 8) + .5), int(bbox[i][3].item() / kwargs.get('down_sample_rate', 8) + .5))
                this_loss = object_loss_fg_bg(object_score[i], clutter_score[i], object_height, object_width)
            else:
                this_loss = loss_fg_bg(object_score[i, None], clutter_score[i, None], )
        refined = [{
                "azimuth": azimuth_pred,
                "elevation": elevation_pred,
                "theta": theta_pred,
                "distance": distance_pred,
                "principal": [
                    this_principal[0].item(),
                    this_principal[1].item(),
                ],
                "score": this_loss.item(),}]
        preds.append(dict(final=refined, **{k: pred[k] / b for k in pred.keys()}))

    return preds
