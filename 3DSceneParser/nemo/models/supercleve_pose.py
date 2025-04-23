import numpy as np
import math
import time
import torch
import torch.nn as nn
from pytorch3d.renderer import camera_position_from_spherical_angles
from nemo.utils import construct_class_by_name, call_func_by_name
from nemo.utils import camera_position_to_spherical_angle, flow_warp
from nemo.utils.general import tensor_linspace
from nemo.utils.visualization import visual_clevr, visual_extrema
from nemo.utils.meshloader import superclever_loader
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale, axis_angle_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion
from skimage.feature import peak_local_max
import time
import os 
from matplotlib import pyplot as plt
import cv2
from nemo.utils.pose import cal_rotation_matrix, rotation_matrix
from tqdm import tqdm
def loss_fg_only(obj_s, clu_s=None, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - reduce_method(obj_s)


def loss_fg_bg(obj_s, clu_s, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    )
def loss_fg_bg_prior(obj_s, clu_s, reduce_method=lambda x: torch.mean(x), prior=None):
    return (torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    ))

def get_corr_pytorch(
    px_samples,
    py_samples,
    kpt_score_map,
    kp_coords,
    kp_vis,
    down_sample_rate,
    hm_h,
    hm_w,
    batch_size=12,
    device="cuda",
):
    all_corr = []

    begin, end = 0, batch_size
    while begin < len(px_samples):
        px_s, py_s = torch.from_numpy(px_samples[begin:end]).to(
            device
        ), torch.from_numpy(py_samples).to(device)
        kpc = torch.from_numpy(kp_coords).to(device)
        kpv = torch.from_numpy(kp_vis).to(device)
        kps = torch.from_numpy(kpt_score_map).to(device)

        xv, yv = torch.meshgrid(px_s, py_s)
        principal_samples = (
            torch.stack([xv, yv], dim=2).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)
        )

        kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
        kpc += principal_samples
        kpc = kpc.reshape(-1, kpc.shape[2], 2)
        kpc = torch.round(kpc / down_sample_rate)

        kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
        kpv = kpv.reshape(-1, kpv.shape[2])
        kpv[kpc[:, :, 0] < 0] = 0
        kpv[kpc[:, :, 0] >= hm_w - 1] = 0
        kpv[kpc[:, :, 1] < 0] = 0
        kpv[kpc[:, :, 1] >= hm_h - 1] = 0

        kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w - 1)
        kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h - 1)
        kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()

        corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]
        corr = torch.sum(corr * kpv, dim=1)   

        all_corr.append(corr.reshape(-1, len(px_s), len(py_s)))

        begin += batch_size
        end += batch_size
        if end > len(px_samples):
            end = len(px_samples)

    corr = torch.cat(all_corr, dim=-2).detach().cpu().numpy()
    return corr


def get_corr_pytorch_unknown(px_samples, py_samples, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w, device):
    l = len(px_samples) // 4
    all_corr = []
    for i in range(4):
        if i != 3:
            px_s, py_s = torch.from_numpy(px_samples[i*l:i*l+l]).to(device), torch.from_numpy(py_samples).to(device)
        else:
            px_s, py_s = torch.from_numpy(px_samples[i*l:]).to(device), torch.from_numpy(py_samples).to(device)
        kpc = torch.from_numpy(kp_coords).to(device)
        kpv = torch.from_numpy(kp_vis).to(device)
        kps = torch.from_numpy(kpt_score_map).to(device)
        
        xv, yv = torch.meshgrid(px_s, py_s)
        principal_samples = torch.stack([xv, yv], dim=2).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)

        kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
        kpc += principal_samples
        kpc = kpc.reshape(-1, kpc.shape[2], 2)
        kpc = torch.round(kpc/down_sample_rate)

        kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
        kpv = kpv.reshape(-1, kpv.shape[2])
        kpv[kpc[:, :, 0] < 0] = 0
        kpv[kpc[:, :, 0] >= hm_w-1] = 0
        kpv[kpc[:, :, 1] < 0] = 0
        kpv[kpc[:, :, 1] >= hm_h-1] = 0

        kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w-1)
        kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h-1)
        kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()

        corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]
        corr = torch.sum(corr * kpv, dim=1)

        all_corr.append(corr.reshape(-1, len(px_s), len(py_s)))
    
    corr = torch.cat(all_corr, dim=-2).detach().cpu().numpy()
    return corr

# maxima
def maxima(arr, d=1):
    coordinates = peak_local_max(arr, min_distance=d, exclude_border=False)
    return coordinates

class DebugAnnoFilter():
    def __init__(self, dof):
        pass

    def __call__(self, sample):
        return sample


class GTSuperCleverProposal():
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return kwargs['transforms'][..., :3, :3], kwargs['transforms'][..., 3:, :3] , kwargs['label']
        # return kwargs['transforms'][..., :3, :3], kwargs['transforms'][..., 3:, :3] @ torch.pinverse(kwargs['transforms'][..., :3, :3]), kwargs['label']

def matrix_norm_to_euler_angles(rotation):
    gt_quaternion = matrix_to_quaternion(rotation)
    norm = torch.norm(gt_quaternion, p=2, dim=-1, keepdim=True)
    norm[norm == 0] = 1
   
    gt_quaternion /= norm
    # rotation = matrix_to_axis_angle(quaternion_to_matrix(gt_quaternion))
    angle = matrix_to_euler_angles(quaternion_to_matrix(gt_quaternion), "XYZ")
    return angle, norm

class SuperClevrProposal():
    # proposal_module = SuperClevrProposal()
    def __init__(
        self, 
        init_samples,
        px_samples,
        py_samples,
        kp_features,
        kp_coords, 
        kp_vis,
        down_sample_rate, 
        device,
        **kwargs
        ):
        # Instead of pre-rendering feature maps, we use sparse keypoint features for coarse detection
        self.px_samples = px_samples
        self.py_samples = py_samples
        self.kp_features = kp_features

        self.kp_coords = kp_coords
        self.kp_vis = kp_vis
        
        self.init_samples = init_samples
        self.down_sample_rate = down_sample_rate
        self.device = device


    def vis(self, cfg, feature_map, clutter_score, R, T, pred, **kwargs):
        '''return rotation, translation, label
        '''
        nkpt, c = self.kp_features.size()
        # feature_map = feature_map.expand(nkpt, -1, -1, -1)
        memory = self.kp_features.view(nkpt, c, 1, 1)
        b, c, hm_h, hm_w = feature_map.size()
        if hm_h >= 80 or hm_w >= 56:
            kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
            for i in range(nkpt):
                kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
        else:
            kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
            kpt_score_map = kpt_score_map.detach().cpu().numpy()
        kpt_score_map = kpt_score_map.reshape(nkpt, -1) # (nkpt, H * W)

        this_name = kwargs['this_name'][0]
        # to visualize the kpt_score_map
        kpt_score_map_vis = kpt_score_map.reshape(nkpt, hm_h, hm_w)
        kpt_score_map_vis_2d = np.max(kpt_score_map_vis, axis=0)
        # from matplotlib import pyplot as plt
        h, w = kpt_score_map_vis_2d.shape
        
        ori_img = kwargs['original_img'][0]
        ori_img = ori_img.cpu().numpy()
        ori_img = cv2.resize(ori_img, (w, h))
        pred = cv2.resize(pred, (w, h))

        kpt_score_map_vis_2d = np.repeat( kpt_score_map_vis_2d[:, :, np.newaxis], 3, axis=2)
        kpt_score_map_vis_2d = kpt_score_map_vis_2d / np.max(kpt_score_map_vis_2d) * 255
        kpt_score_map_vis_2d = kpt_score_map_vis_2d.astype(np.uint8)
        kpt_score_map_vis_2d = cv2.applyColorMap(kpt_score_map_vis_2d, cv2.COLORMAP_VIRIDIS)

        # Concatenate images horizontally
        # import ipdb; ipdb.set_trace()
        folder_name = 'kpt_score_map_cont_phy_v3'
        # folder_name = 'kpt_score_map_cont_2'
        concatenated_image = np.hstack((ori_img[:, :, ::-1], kpt_score_map_vis_2d, pred))
        if not os.path.exists(f'vis/{folder_name}/{this_name.split("/")[0]}'):
            os.makedirs(f'vis/{folder_name}/{this_name.split("/")[0]}')
        # return concatenated_image
        # cv2.imwrite(f'vis/{folder_name}/{this_name.split("/")[0]}/{this_name.split("/")[1].split(".")[0]}_{epoch}.png', concatenated_image)
        cv2.imwrite(f'vis/{folder_name}/{this_name}', concatenated_image)


        ######
        # return rotation, kwargs['transforms'][..., 3:, :3] , kwargs['label']
        # return kwargs['transforms'][..., :3, :3], translation, kwargs['label'][:, 0:len(extrema)]
        # return rotation, translation, kwargs['label'][:, 0:len(extrema)]


    def vis2(self, cfg, feature_map, clutter_score, R, T, pred, **kwargs):
        '''return rotation, translation, label
        '''
        # nkpt, c = self.kp_features.size()
        # # feature_map = feature_map.expand(nkpt, -1, -1, -1)
        # memory = self.kp_features.view(nkpt, c, 1, 1)
        # b, c, hm_h, hm_w = feature_map.size()
        # if hm_h >= 80 or hm_w >= 56:
        #     kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
        #     for i in range(nkpt):
        #         kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
        # else:
        #     kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
        #     kpt_score_map = kpt_score_map.detach().cpu().numpy()
        # kpt_score_map = kpt_score_map.reshape(nkpt, -1) # (nkpt, H * W)

        this_name = kwargs['this_name'][0]
        # # to visualize the kpt_score_map
        # kpt_score_map_vis = kpt_score_map.reshape(nkpt, hm_h, hm_w)
        # kpt_score_map_vis_2d = np.max(kpt_score_map_vis, axis=0)
        # from matplotlib import pyplot as plt
        # h, w = kpt_score_map_vis_2d.shape

        
        ori_img = kwargs['original_img'][0]
        ori_img = ori_img.cpu().numpy()
        h, w, _ = ori_img.shape
        h = h // 2
        w = w // 2
        ori_img = cv2.resize(ori_img, (w, h))
        pred = cv2.resize(pred, (w, h))

        # kpt_score_map_vis_2d = np.repeat( kpt_score_map_vis_2d[:, :, np.newaxis], 3, axis=2)
        # kpt_score_map_vis_2d = kpt_score_map_vis_2d / np.max(kpt_score_map_vis_2d) * 255
        # kpt_score_map_vis_2d = kpt_score_map_vis_2d.astype(np.uint8)
        # kpt_score_map_vis_2d = cv2.applyColorMap(kpt_score_map_vis_2d, cv2.COLORMAP_VIRIDIS)

        # Concatenate images horizontally

        folder_name = 'kpt_score_map_cont_phy_v3_2'
        # folder_name = 'kpt_score_map_cont_v3'

        concatenated_image = np.vstack((ori_img[:, :, ::-1], pred))
        if not os.path.exists(f'vis/{folder_name}/{this_name.split("/")[0]}'):
            os.makedirs(f'vis/{folder_name}/{this_name.split("/")[0]}')
        # return concatenated_image
        # cv2.imwrite(f'vis/{folder_name}/{this_name.split("/")[0]}/{this_name.split("/")[1].split(".")[0]}_{epoch}.png', concatenated_image)
        cv2.imwrite(f'vis/{folder_name}/{this_name}', concatenated_image)


        ######
        # return rotation, kwargs['transforms'][..., 3:, :3] , kwargs['label']
        # return kwargs['transforms'][..., :3, :3], translation, kwargs['label'][:, 0:len(extrema)]
        # return rotation, translation, kwargs['label'][:, 0:len(extrema)]


    def gt_proposal(self, cfg, feature_map, clutter_score, R, T, prev_preds, **kwargs):
        '''
        Version 1: 3D estimation
        '''
        # return kwargs['transforms'][..., :3, :3], kwargs['transforms'][..., 3:, :3] , kwargs['label']
        
        B, N = kwargs['angles'].shape[:2]
        
        # initialize a empth angles
        # if prev_preds[0] is None:
        #     angles = torch.zeros(B,N, 3)
            
        # else:
        #     angles = kwargs['angles']
        scales = torch.ones(B,N)
        
        angles = torch.zeros(B,N, 3)
        # Load gt rotation angle
        angles = kwargs['angles']
        scales = kwargs['scales'].view(scales.shape)
        translation = kwargs['translations'].view(B, N, 3)

        visiblity = kwargs['visibility'].view(scales.shape)


        for i in range(B):
            if prev_preds[i] is not None:
                device = angles.device
                prev_vis = prev_preds[i]['visibility'].unsqueeze(-1)
                angles[i] = prev_preds[i]['rotation'] 
                # translation[i] = prev_preds[i]['translation']
                translation[i] = prev_preds[i]['translation'] * prev_vis + translation[i].to(prev_vis.device) * (1 - prev_vis)
                
        # if prev_pred is None:
        #     angles = torch.ones(B,N, 4)*0.5
        # else:
        #     angles = prev_pred['rotation']
        #     translation = prev_pred['translation']
        #     import ipdb; ipdb.set_trace()
  

        return angles, translation, scales, visiblity, kwargs['label']

    def __call__(self, cfg, feature_map, clutter_score, R, T, **kwargs):
        '''
        Version 1: 3D estimation
        '''
        nkpt, c = self.kp_features.size()
        # feature_map = feature_map.expand(nkpt, -1, -1, -1)
        memory = self.kp_features.view(nkpt, c, 1, 1)
        b, c, hm_h, hm_w = feature_map.size()
        if hm_h >= 80 or hm_w >= 56:
            kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
            for i in range(nkpt):
                kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
        else:
            kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
            kpt_score_map = kpt_score_map.detach().cpu().numpy()
        kpt_score_map = kpt_score_map.reshape(nkpt, -1) # (nkpt, H * W)

        # Initialization of the principal point (detection)
        # This is important for 6D version, remove temporarily
        xv, yv = np.meshgrid(self.px_samples, self.py_samples, indexing='ij')

        corr = get_corr_pytorch(self.px_samples, self.py_samples, kpt_score_map, self.kp_coords, self.kp_vis, self.down_sample_rate, hm_h, hm_w, 
                                batch_size=cfg.inference.batch_size,
                                device = self.device)    

        corr = corr.reshape(12, 4, 3, 1, len(xv), len(yv)) # (pose, xcorr, ycorr)
        corr2d = corr.reshape(-1, len(xv), len(yv))
        corr2d_max = np.max(corr2d, axis=0)
        # extrema_2d = maxima(corr2d_max, d=2)
        extrema_2d = [[0, 0]]
        extrema = []
        translation = []
        rotation = []
        boxes = []

        for e in extrema_2d:
            c = corr2d[:, e[0], e[1]].reshape(12, 4, 3, 1)
            e_azim, e_elev, e_the, e_dist = np.unravel_index(np.argmax(c, axis=None), c.shape)
            # print(corr2d_max[e[0], e[1]])
            # if corr2d_max[e[0], e[1]] >= 80.0:
            if corr2d_max[e[0], e[1]] >= 0.0:
                p = self.init_samples[e_azim, e_elev, e_the, e_dist]
                _rotation = compute_rotation(p[0], p[1], p[3])
                
                # compute the translation
                # step_x, step_y = self.px_samples[1] // 2, self.py_samples[1] // 2 
                step_x, step_y = self.px_samples[0], self.py_samples[0]
                _p = np.array([(self.px_samples[e[0]] + step_x - hm_w / 2) / 1400,
                                (self.px_samples[e[1]] + step_y - hm_h / 2) / 1400, 1])
                _trans = np.dot(R[0].numpy().T, _p * p[3]) + T.numpy()
                translation.append(_trans)
                rotation.append(_rotation)
                boxes.append(e)
                extrema.append({
                    'box': e,
                    'rotation': _rotation,
                    'translation': _trans,
                    'distance': p[3],
                    'px': self.px_samples[e[0]],
                    'py': self.py_samples[e[1]],
                    'principal': [self.px_samples[e[0]], self.py_samples[e[1]]]
                })
        
        # this is for padding, to the max obj
        # translation += [torch.zeros((1, 3)) for _ in range(10 - len(extrema_2d))]
        # rotation += [torch.zeros((3, 3)) for _ in range(10 - len(extrema_2d))]

        translation = torch.tensor(translation).unsqueeze(0)
        rotation = torch.tensor(rotation).unsqueeze(0).float()
        # visual_extrema(len(self.px_samples)-1, boxes, cfg.args.save_dir, **kwargs)
        pred = {'pre_render': extrema, 'corr2d': corr2d_max}
        # according to the extrema, get the rotation metrix
        
        # return rotation, kwargs['transforms'][..., 3:, :3] , kwargs['label']
        # return kwargs['transforms'][..., :3, :3], translation, kwargs['label'][:, 0:len(extrema)]
        return rotation, translation, kwargs['label'][:, 0:len(extrema)]
    
# TODO:
# class SuperClevrProposal():
#     # proposal_module = SuperClevrProposal()
#     def __init__(
#         self, 
#         init_samples,
#         px_samples,
#         py_samples,
#         kp_features,
#         kp_coords, 
#         kp_vis,
#         down_sample_rate, 
#         device,
#         **kwargs
#         ):
#         # Instead of pre-rendering feature maps, we use sparse keypoint features for coarse detection
#         self.px_samples = px_samples
#         self.py_samples = py_samples
#         self.kp_features = kp_features

#         self.kp_coords = kp_coords
#         self.kp_vis = kp_vis
        
#         self.init_samples = init_samples
#         self.down_sample_rate = down_sample_rate
#         self.device = device


#     def vis(self, cfg, feature_map, clutter_score, R, T, **kwargs):
#         '''return rotation, translation, label
#         '''
#         nkpt, c = self.kp_features.size()
#         # feature_map = feature_map.expand(nkpt, -1, -1, -1)
#         memory = self.kp_features.view(nkpt, c, 1, 1)
#         b, c, hm_h, hm_w = feature_map.size()
#         if hm_h >= 80 or hm_w >= 56:
#             kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
#             for i in range(nkpt):
#                 kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
#         else:
#             kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
#             kpt_score_map = kpt_score_map.detach().cpu().numpy()
#         kpt_score_map = kpt_score_map.reshape(nkpt, -1) # (nkpt, H * W)

        
#         this_name = kwargs['this_name'][0]
#         # to visualize the kpt_score_map
#         kpt_score_map_vis = kpt_score_map.reshape(nkpt, hm_h, hm_w)
#         kpt_score_map_vis_2d = np.max(kpt_score_map_vis, axis=0)
#         # from matplotlib import pyplot as plt
#         h, w = kpt_score_map_vis_2d.shape
        
#         ori_img = kwargs['original_img'][0]
#         ori_img = ori_img.cpu().numpy()
#         ori_img = cv2.resize(ori_img, (w, h))

#         kpt_score_map_vis_2d = np.repeat( kpt_score_map_vis_2d[:, :, np.newaxis], 3, axis=2)
#         kpt_score_map_vis_2d = kpt_score_map_vis_2d / np.max(kpt_score_map_vis_2d) * 255
#         kpt_score_map_vis_2d = kpt_score_map_vis_2d.astype(np.uint8)
#         kpt_score_map_vis_2d = cv2.applyColorMap(kpt_score_map_vis_2d, cv2.COLORMAP_VIRIDIS)

#         # Concatenate images horizontally
#         concatenated_image = np.hstack((ori_img, kpt_score_map_vis_2d))
#         cv2.imwrite(f'vis/kpt_score_map/{this_name}.png', concatenated_image)


#         ######
#         # return rotation, kwargs['transforms'][..., 3:, :3] , kwargs['label']
#         # return kwargs['transforms'][..., :3, :3], translation, kwargs['label'][:, 0:len(extrema)]
#         # return rotation, translation, kwargs['label'][:, 0:len(extrema)]


#     def __call__(self, cfg, feature_map, clutter_score, R, T, **kwargs):
#         '''return rotation, translation, label
#         '''
#         nkpt, c = self.kp_features.size()
#         # feature_map = feature_map.expand(nkpt, -1, -1, -1)
#         memory = self.kp_features.view(nkpt, c, 1, 1)
#         b, c, hm_h, hm_w = feature_map.size()
#         if hm_h >= 80 or hm_w >= 56:
#             kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
#             for i in range(nkpt):
#                 kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
#         else:
#             kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
#             kpt_score_map = kpt_score_map.detach().cpu().numpy()
#         kpt_score_map = kpt_score_map.reshape(nkpt, -1) # (nkpt, H * W)

#         # Initialization of the principal point (detection)
#         xv, yv = np.meshgrid(self.px_samples, self.py_samples, indexing='ij')

#         corr = get_corr_pytorch(self.px_samples, self.py_samples, kpt_score_map, self.kp_coords, self.kp_vis, self.down_sample_rate, hm_h, hm_w, 
#                                 batch_size=cfg.inference.batch_size,
#                                 device = self.device)

#         corr = corr.reshape(12, 4, 3, 1, len(xv), len(yv)) # (pose, xcorr, ycorr)
#         corr2d = corr.reshape(-1, len(xv), len(yv))
#         corr2d_max = np.max(corr2d, axis=0)
#         # extrema_2d = maxima(corr2d_max, d=2)
#         extrema_2d = [[0, 0]]
#         extrema = []
#         translation = []
#         rotation = []
#         boxes = []

#         for e in extrema_2d:
#             c = corr2d[:, e[0], e[1]].reshape(12, 4, 3, 1)
#             e_azim, e_elev, e_the, e_dist = np.unravel_index(np.argmax(c, axis=None), c.shape)
#             # print(corr2d_max[e[0], e[1]])
#             # if corr2d_max[e[0], e[1]] >= 80.0:
#             if corr2d_max[e[0], e[1]] >= 0.0:
#                 p = self.init_samples[e_azim, e_elev, e_the, e_dist]
#                 _rotation = compute_rotation(p[0], p[1], p[3])
                
#                 # compute the translation
#                 # step_x, step_y = self.px_samples[1] // 2, self.py_samples[1] // 2 
#                 step_x, step_y = self.px_samples[0], self.py_samples[0]
#                 _p = np.array([(self.px_samples[e[0]] + step_x - hm_w / 2) / 1400,
#                                 (self.px_samples[e[1]] + step_y - hm_h / 2) / 1400, 1])
#                 _trans = np.dot(R[0].numpy().T, _p * p[3]) + T.numpy()
#                 translation.append(_trans)
#                 rotation.append(_rotation)
#                 boxes.append(e)
#                 extrema.append({
#                     'box': e,
#                     'rotation': _rotation,
#                     'translation': _trans,
#                     'distance': p[3],
#                     'px': self.px_samples[e[0]],
#                     'py': self.py_samples[e[1]],
#                     'principal': [self.px_samples[e[0]], self.py_samples[e[1]]]
#                 })
        
#         # why add 10?
#         # this is for padding, to the max obj
#         # translation += [torch.zeros((1, 3)) for _ in range(10 - len(extrema_2d))]
#         # rotation += [torch.zeros((3, 3)) for _ in range(10 - len(extrema_2d))]

#         translation = torch.tensor(translation).unsqueeze(0)
#         rotation = torch.tensor(rotation).unsqueeze(0).float()
#         # visual_extrema(len(self.px_samples)-1, boxes, cfg.args.save_dir, **kwargs)
#         pred = {'pre_render': extrema, 'corr2d': corr2d_max}
#         # according to the extrema, get the rotation metrix
        
#         # return rotation, kwargs['transforms'][..., 3:, :3] , kwargs['label']
#         # return kwargs['transforms'][..., :3, :3], translation, kwargs['label'][:, 0:len(extrema)]
#         return rotation, translation, kwargs['label'][:, 0:len(extrema)]


def transform_3d_to_2d( x, y, z, camera):
    R = np.array(camera['R'])[:, :, 0]
    K = np.array(camera['K'])[:, :, 0]
    homo_transform = np.linalg.inv(R)
    homo_intrinsics = np.zeros((3, 4), dtype=np.float32)
    homo_intrinsics[:, :3] = K

    point4d = np.concatenate([[x], [y], [z], [1.]])
    projected = homo_intrinsics @ homo_transform @ point4d
    image_coords = projected / projected[2]
    image_coords[2] = np.sign(projected[2])
    return image_coords[0], image_coords[1]


def multi_object_solve_pose(
    cfg,
    feature_map,
    inter_module,
    proposal_module,
    clutter_bank=None,
    device="cuda",
    R=None,
    T=None,
    dof=3,
    img_name = None,
    prev_pred = None,
    **kwargs
):
    feature_map = feature_map.detach()
    b, c, hm_h, hm_w = feature_map.size()

    gt_angles = kwargs['angles']

    
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
    # add prev poses
    # only support batch_size = 1
    prev_preds = []
    for i in range(b):    
                
        if img_name[i].split('/')[-1] == 'rgba_00000.png':
            frame_id = 0
            prev_pred_input = None
            epochs = cfg.inference.epochs
        elif prev_pred is not None:
            video_name = img_name[i].split('/')[0]
            frame_id = int(img_name[i].split('/')[1].split('.')[0].split('_')[-1])
            # initialize from last frame

            if not cfg.args.physics_prior:
                init_frame_name = f'rgba_{str(frame_id-1).zfill(5)}.png'
            # initialize from physic prediction
            else:
                init_frame_name = f'rgba_{str(frame_id).zfill(5)}.png'
                last_frame_name = f'rgba_{str(frame_id-1).zfill(5)}.png'
                prev_pred[video_name][init_frame_name]['visibility'] = prev_pred[video_name][last_frame_name]['visibility']
            prev_pred_input = prev_pred[video_name][init_frame_name] # format same as pred['final']
            epochs = cfg.inference.epochs_finetune
        else:
            raise ValueError('prev_pred is None')
        prev_preds.append(prev_pred_input)

    

    init_angles, init_translation, scales, visiblity, init_obj_idxs = proposal_module.gt_proposal(cfg, feature_map=feature_map, \
                                                                            clutter_score=clutter_score, R=R, T=T, img_name = img_name, prev_preds = prev_preds, **kwargs)
    start = time.time()
    
    # '''skip for visualization
    inter_module.reset()


    obj_idxs = init_obj_idxs

    R = R.cuda()
    T = T.cuda()
    # init_translation += 0.5
    prior_init_translation =  init_translation.clone().to(device)
    translation = torch.nn.Parameter(init_translation.to(device), requires_grad=True)
    angles = torch.nn.Parameter(init_angles.to(device), requires_grad=True)
    rotation = axis_angle_to_matrix(angles)

    scales = scales.to(device)
    visiblity = visiblity.to(device)

    optim = construct_class_by_name(**cfg.inference.optimizer, params=[translation, angles])

    scheduler_kwargs = {"optimizer": optim}
    scheduler = construct_class_by_name(**cfg.inference.scheduler, **scheduler_kwargs)

    eye_ = torch.cat([torch.eye(3), torch.zeros((3, 1))], dim=1)
    to_4x4 = lambda x, eye3=eye_.to(device)[None, None], zeros4=(torch.eye(4) - eye_.T @ eye_).to(device)[None, None]: eye3.transpose(-2, -1) @ x @ eye3 + zeros4

    feature_map = feature_map.permute(0, 2, 3, 1)
    
    '''
    # print('init', translation, rotation)
    # when you do visualization, you should set the batch_size = 1
    
    # visualize
    # img_pred = visual_clevr(cfg, translation, rotation, init_obj_idxs, R, T, cfg.args.save_dir, distances=scales, **kwargs)
    # img_pred = img_pred.astype(np.uint8)
    # H, W, _ = img_pred.shape
    # if prev_pred is not None:
    #     collisions = prev_pred["collisions"] 
    #     for collision in collisions:
    #         x, y, z = collision['position']
    #         x, y = transform_3d_to_2d(x, y, z, kwargs['camera'])
    #         x = int(x*W)
    #         y = int(y*H)
    #         img_pred = cv2.circle(img_pred, (x, y), 5, (0, 0, 255), -1)
    # s_name, i_name = img_name[i].split('/')
    # os.makedirs(f'vis/collision/{s_name}', exist_ok=True)
    # cv2.imwrite(f"vis/collision/{s_name}/{img_name[i].split('/')[-1]}", img_pred)
    '''
    sigma = cfg.args.sigma
    for epo in range(epochs):
        rotation = axis_angle_to_matrix(angles)
        

        m_matrix = to_4x4(rotation)
        trans_ = [Transform3d(matrix=m_matrix[ii]).compose(Scale(scales[ii])).compose(Translate(translation[ii])) for ii in range(rotation.shape[0])]
        

        projected_map = inter_module(transforms=trans_, indexs=obj_idxs, R=R, T=T)
        object_score = torch.sum(projected_map * feature_map.detach(), dim=3)
        if cfg.args.physics_prior:
            phy_likelihood = 1-torch.exp(-torch.sum((translation - prior_init_translation)**2) / (2*sigma**2)) / ((2*np.pi*sigma**2)**0.5 )
            phy_likelihood = torch.clamp(phy_likelihood, 0, 1)
            loss = loss_fg_bg(object_score, clutter_score.detach()) * phy_likelihood
        else:
            loss = loss_fg_bg(object_score, clutter_score.detach())
        # import ipdb; ipdb.set_trace()
        # loss = loss_fg_bg_prior(object_score, clutter_score.detach(), prior = phy_likelihood)
        # import ipdb; ipdb.set_trace()c
        angles.retain_grad()
        # translation.retain_grad()

        loss.backward()
        optim.step()
        if epo == epochs-1:
            visiblity = (torch.abs(angles.grad).max(2)[0]*100 != 0).int()
            # visiblity2 = (torch.abs(translation.grad).max(2)[0]*100 > 0).int()
            # visiblity 
            
        optim.zero_grad()

        if (epo + 1) % max(cfg.inference.epochs // 3, 1) == 0:
            scheduler.step()
    print('proposal time', time.time()-start)
    frame_id = int(img_name[0].split('/')[1].split('.')[0].split('_')[-1])
    if cfg.args.vis_inference and frame_id % 5 == 0:
        feature_map_vis = feature_map.permute(0, 3, 1, 2)
        img_pred = visual_clevr(cfg, translation, rotation, init_obj_idxs, R, T, cfg.args.save_dir,distances=scales, **kwargs)
        proposal_module.vis(cfg, feature_map=feature_map_vis, clutter_score=clutter_score, R=R, T=T, pred = img_pred, **kwargs)
        
    # '''
    # saving the results

    start = time.time()
    preds = []
    for i in range(b):
        translation_pred, angles_pred = (
            translation[i],
            angles[i]
        )
        pred = {
                "final": {
                    "scene": img_name[i],
                    "translation": translation_pred,
                    "rotation": angles_pred,
                    "scales": scales[i],
                    "visibility": visiblity[i],
                    "num_objects": kwargs["num_objects"][i], # this shouldn't be used for inference
                    "obj_idxs": obj_idxs[i]
                },
            }

        preds.append(pred)

    # TODO: check line 48 in config => focal_length: 1400
    
    # preds = []
    for i in range(b):
        translation_pred, rotation_pred = (
            translation[i],
            rotation[i]
        )
        this_loss = loss_fg_bg(object_score[i, None], clutter_score[i, None], )
        distance_loss = torch.sum((translation_pred.cpu() - init_translation[i]) ** 2) ** 0.5 / kwargs["num_objects"][i]
        preds[i]['final'].update({
            "loss": this_loss.item(),
            "distance_loss": distance_loss.item()
        })
        # refined = {
        #     "scene": kwargs['this_name'][i],
        #     "translation": translation_pred,
        #     "rotation": rotation_pred,
        #     "num_objects": kwargs["num_objects"][i],

        #     }
        # preds.append(dict(final=refined))
    print('post processing time', time.time()-start)
    return preds
    # '''

class BackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x
        ):  
            return x
    
    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.max().item(), grad_x.min().item())
        return grad_x


def compute_rotation(azim, elev, dist):
    azimuth = - azim
    elevation = -(math.pi / 2 - elev)
    Rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )  # rotation by azimuth
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ]
    )  # rotation by elevation  
    R_rot = np.dot(Rx, Rz)
    return math.sqrt(dist) * R_rot
    
def pre_compute_kp_coords_clevr(
    mesh_path,
    azimuth_samples,
    elevation_samples,
    theta_samples,
    distance_samples,
    translation_samples,
    viewport=1400,
):
    """Calculate vertex visibility for cuboid models."""
    mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path)])))
    xvert = []
    for cate in (mesh_sub_cates):
        _xvert, _ = superclever_loader(os.path.join(mesh_path, cate + '.obj'))
        xvert.append(_xvert)
    xvert = np.concatenate(xvert)

    xmin, xmax = np.min(xvert[:, 0]), np.max(xvert[:, 0])
    ymin, ymax = np.min(xvert[:, 1]), np.max(xvert[:, 1])
    zmin, zmax = np.min(xvert[:, 2]), np.max(xvert[:, 2])
    xmean = (xmin + xmax) / 2
    ymean = (ymin + ymax) / 2
    zmean = (zmin + zmax) / 2
    pts = [
        [xmean, ymean, zmin],
        [xmean, ymean, zmax],
        [xmean, ymin, zmean],
        [xmean, ymax, zmean],
        [xmin, ymean, zmean],
        [xmax, ymean, zmean],
    ]

    poses = np.zeros(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            4,
        ),
        dtype=np.float32,
    )
    num_vis_faces = []
    count = 0
    for azim_ in azimuth_samples:
        for elev_ in elevation_samples:
            for theta_ in theta_samples:
                for dist_ in distance_samples:
                    poses[count] = [azim_, elev_, theta_, dist_]
                    count += 1
                    if elev_ == 0:
                        if azim_ in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                            num_vis_faces.append(1)
                        else:
                            num_vis_faces.append(2)
                    else:
                        if azim_ in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                            num_vis_faces.append(2)
                        else:
                            num_vis_faces.append(3)

    kp_coords = np.zeros(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            len(xvert),
            2,
        ),
        dtype=np.float32,
    )
    kp_vis = np.ones(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            len(xvert),
        ),
        dtype=np.float32,
    )
    xvert_ext = np.concatenate((xvert, pts), axis=0)
    for i, pose_ in enumerate(poses):
        azim_, elev_, theta_, dist_ = pose_

        C = np.zeros((3, 1))
        C[0] = dist_ * math.cos(elev_) * math.sin(azim_)
        C[1] = -dist_ * math.cos(elev_) * math.cos(azim_)
        C[2] = dist_ * math.sin(elev_)
        azimuth = -azim_
        elevation = -(math.pi / 2 - elev_)
        Rz = np.array(
            [
                [math.cos(azimuth), -math.sin(azimuth), 0],
                [math.sin(azimuth), math.cos(azimuth), 0],
                [0, 0, 1],
            ]
        )  # rotation by azimuth
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(elevation), -math.sin(elevation)],
                [0, math.sin(elevation), math.cos(elevation)],
            ]
        )  # rotation by elevation
        R_rot = np.dot(Rx, Rz)
        R = np.hstack((R_rot, np.dot(-R_rot, C)))
        P = np.array([[viewport, 0, 0], [0, viewport, 0], [0, 0, -1]])
        x3d_ = np.hstack((xvert_ext, np.ones((len(xvert_ext), 1)))).T
        x3d_ = np.dot(R, x3d_)
        # x3d_r_ = np.dot(P, x3d_)
        x2d = np.dot(P, x3d_)
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        x2d = x2d[0:2, :]
        R2d = np.array(
            [
                [math.cos(theta_), -math.sin(theta_)],
                [math.sin(theta_), math.cos(theta_)],
            ]
        )
        x2d = np.dot(R2d, x2d).T
        x2d[:, 1] *= -1

        # principal = np.array([px_, py_], dtype=np.float32)
        # x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

        x2d = x2d[: len(xvert)]
        kp_coords[i] = x2d

        center3d = x3d_[:, len(xvert) :]
        face_dist = np.sqrt(
            np.square(center3d[0, :])
            + np.square(center3d[1, :])
            + np.square(center3d[2, :])
        )
        ind = np.argsort(face_dist)[: num_vis_faces[i]]

        """
        # 13 13 5 13x12 + 13x12 + 5x12 + 5x12 + 5x13 + 5x13
        # 8 17 6 17x8 + 17x8 + 6x8 + 6x8 + 6x17 + 6x17
        if 0 in ind:
            kp_vis[i, 0:mesh_face_breaks[0]] = 1
        if 1 in ind:
            kp_vis[i, mesh_face_breaks[0]:mesh_face_breaks[1]] = 1
        if 2 in ind:
            kp_vis[i, mesh_face_breaks[1]:mesh_face_breaks[2]] = 1
        if 3 in ind:
            kp_vis[i, mesh_face_breaks[2]:mesh_face_breaks[3]] = 1
        if 4 in ind:
            kp_vis[i, mesh_face_breaks[3]:mesh_face_breaks[4]] = 1
        if 5 in ind:
            kp_vis[i, mesh_face_breaks[4]:mesh_face_breaks[5]] = 1
        """

    poses = poses.reshape(
        len(azimuth_samples),
        len(elevation_samples),
        len(theta_samples),
        len(distance_samples),
        4,
    )
    return poses, kp_coords, kp_vis

