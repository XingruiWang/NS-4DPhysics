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
    enable_voge = False

try:
    from CuNeMo import fast_feature_similarity, fast_collect_score
    enable_cunemo = True
except:
    enable_cunemo = False

if not enable_voge:
    from TorchBatchifier import Batchifier


def loss_fg_only(obj_s, clu_s=None, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - reduce_method(obj_s)


def loss_fg_bg(obj_s, clu_s, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    )


def get_pre_render_samples(inter_module, azum_samples, elev_samples, theta_samples, distance_samples=[5], device='cpu',):
    with torch.no_grad():
        get_c = []
        get_theta = []
        get_samples = [[azum_, elev_, theta_, distance_] for azum_ in azum_samples for elev_ in elev_samples for theta_ in theta_samples for distance_ in distance_samples]
        out_maps = []
        for sample_ in get_samples:
            theta_ = torch.ones(1, device=device) * sample_[2]
            C = camera_position_from_spherical_angles(sample_[3], sample_[1], sample_[0], degrees=False, device=device)

            projected_map = inter_module(C, theta_)
            out_maps.append(projected_map)
            get_c.append(C.detach())
            get_theta.append(theta_)

        get_c = torch.stack(get_c, ).squeeze(1)
        get_theta = torch.cat(get_theta)
        out_maps = torch.stack(out_maps)

    return out_maps, get_c, get_theta


@torch.no_grad()
def align_no_centered(maps_source, distance_source, principal_source, maps_target_shape, distance_target, principal_target, padding_mode='zeros'):
    """
    maps_source: [n, c, h1, w1]
    distance_source: [n, ]
    principal_source: [n, 2]
    """
    n, c, h1, w1 = maps_source.shape
    h0, w0 = maps_target_shape

    # distance source larger, sampling grid wider
    resize_rate = (distance_source / distance_target).float()

    range_x_min = 2 * principal_source[:, 0] / w1 - w0 / (w1 * resize_rate) - principal_target[:, 0] * 2 / w0
    range_x_max = 2 * principal_source[:, 0] / w1 + w0 / (w1 * resize_rate) - principal_target[:, 0] * 2 / w0
    range_y_min = 2 * principal_source[:, 1] / h1 - h0 / (h1 * resize_rate) - principal_target[:, 1] * 2 / h0
    range_y_max = 2 * principal_source[:, 1] / h1 + h0 / (h1 * resize_rate) - principal_target[:, 1] * 2 / h0

    # [n, w0] -> [n, h0, w0]
    grid_x = tensor_linspace(range_x_min, range_x_max, int(w0.item()))[:, None, :].expand(-1, int(h0.item()), -1)
    # [n, h0] -> [n, h0, w0]
    grid_y = tensor_linspace(range_y_min, range_y_max, int(h0.item()))[:, :, None].expand(-1, -1, int(w0.item()))

    grids = torch.cat([grid_x[..., None], grid_y[..., None]], dim=3)

    return torch.nn.functional.grid_sample(maps_source, grids, padding_mode=padding_mode)


def get_init_pos_rendered(samples_maps, samples_pos, samples_theta, predicted_maps, clutter_scores=None, batch_size=32):
    """
    samples_pos: [n, 3]
    samples_theta: [n, ]
    samples_map: [n, c, h, w]
    predicted_map: [b, c, h, w]
    clutter_score: [b, h, w]
    """
    n = samples_maps.shape[0]
    if clutter_scores is None:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    batchifier = Batchifier(batch_size=batch_size, batch_args=('projected_map', 'predicted_map') + (tuple() if clutter_scores is None else ('clutter_map', )), target_dims=(0, 1))

    with torch.no_grad():
        # [n, b, c, h, w] -> [n, b]
        target_shape = (n, *predicted_maps.shape)
        get_loss = batchifier(cal_sim)(projected_map=samples_maps.expand(*target_shape).contiguous(),
                                       predicted_map=predicted_maps[None].expand(*target_shape).contiguous(), 
                                       clutter_map=clutter_scores[None].expand(n, *clutter_scores.shape).contiguous(), )

        # [b]
        use_indexes = torch.min(get_loss, dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(get_loss, dim=0)[0]


def get_init_pos_rendered_dim0(samples_maps, samples_pos, samples_theta, predicted_maps, clutter_scores=None, batch_size=32):
    """
    samples_pos: [n, 3]
    samples_theta: [n, ]
    samples_map: [n, c, h, w]
    predicted_map: [b, c, h, w]
    clutter_score: [b, h, w]
    """
    n = samples_maps.shape[0]
    if clutter_scores is None:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,chw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,chw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    batchifier = Batchifier(batch_size=batch_size, batch_args=('projected_map', ), target_dims=(0, ))

    with torch.no_grad():
        # [n, b, c, h, w] -> [n, b]
        get_loss = []

        for i in range(predicted_maps.shape[0]):
            # [n]
            get_loss.append(batchifier(cal_sim)(projected_map=samples_maps.squeeze(1), predicted_map=predicted_maps[i], clutter_map=clutter_scores[None, i]))

        # b * [n, ] -> [n, b]
        get_loss = torch.stack(get_loss).T

        # [b]
        use_indexes = torch.min(get_loss, dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(get_loss, dim=0)[0]



def get_init_pos(inter_module, samples_pos, samples_theta, predicted_maps, clutter_scores=None, reset_distance=None):
    if clutter_scores is None:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    with torch.no_grad():
        out_scores = []
        for pos_, theta_ in zip(samples_pos, samples_theta):
            if reset_distance is not None:
                maps_ = inter_module(torch.nn.functional.normalize(pos_[None]) * reset_distance[:, None], theta_[None].expand(reset_distance.shape[0], -1))
            else:
                maps_ = inter_module(pos_[None], theta_[None])
            scores_ = cal_sim(maps_, predicted_maps, clutter_scores)
            out_scores.append(scores_)
        use_indexes = torch.min(torch.stack(out_scores), dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(torch.stack(out_scores), dim=0)[0]


class NeMoRenderProposal():
    def __init__(self, inter_module, cfg, azum_samples, elev_samples, theta_samples, distance_samples, px_samples, py_samples, dof, down_sample_rate=1, pre_render=True, device='cpu', **kwargs):
        self.kwargs = kwargs
        self.cfg = cfg
        self.pre_render = pre_render
        self.inter_module = inter_module
        self.device = device
        # out of memory here?
        self.feature_pre_rendered, self.cam_pos_pre_rendered, self.theta_pre_rendered = get_pre_render_samples(
            self.inter_module,
            azum_samples=azum_samples,
            elev_samples=elev_samples,
            theta_samples=theta_samples,
            distance_samples=distance_samples,
            device=self.device
        )
        if dof == 3:
            assert distance_samples.shape[0] == 1
            self.record_distance = distance_samples[0]
        if dof == 6:
            map_shape = inter_module.rasterizer.raster_settings.image_size
            self.samples_principal = torch.stack((torch.from_numpy(px_samples).view(-1, 1).expand(-1, py_samples.shape[0]) * map_shape[1], 
                                                  torch.from_numpy(py_samples).view(1, -1).expand(px_samples.shape[0], -1) * map_shape[0]), ).view(2, -1).T.to(self.device)
        self.dof = dof
        self.down_sample_rate = down_sample_rate

    def __call__(self, feature_map, clutter_score=None, principal=None, distance=None, **kwargs):
        dof = self.dof
        inter_module = self.inter_module

        if principal is not None:
            principal = principal / self.down_sample_rate

        # 3 DoF or 4 DoF
        if dof == 3 or dof == 4:
            if dof == 3:
                assert distance is not None
                distance_source = distance.to(feature_map.device)
                distance_target = self.record_distance * torch.ones(feature_map.shape[0]).to(feature_map.device)
            else:
                distance_source = torch.ones(feature_map.shape[0]).to(feature_map.device)
                distance_target = torch.ones(feature_map.shape[0]).to(feature_map.device)
            
            # Not centered images
            if principal is not None:
                maps_target_shape = inter_module.rasterizer.cameras.image_size 
                t_feature_map = align_no_centered(maps_source=feature_map, principal_source=principal, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_target)
                t_clutter_score = align_no_centered(maps_source=clutter_score[:, None], principal_source=principal, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_target).squeeze(1)
                init_principal = principal.float()
            # Centered images
            else:
                init_principal = inter_module.rasterizer.cameras.principal_point
                t_feature_map = feature_map
                t_clutter_score = clutter_score

            if self.pre_render:
                init_C, init_theta, _ = get_init_pos_rendered_dim0(samples_maps=self.feature_pre_rendered, 
                                                        samples_pos=self.cam_pos_pre_rendered, 
                                                        samples_theta=self.theta_pre_rendered, 
                                                        predicted_maps=t_feature_map, 
                                                        clutter_scores=t_clutter_score, 
                                                        batch_size=self.cfg.get('batch_size_no_grad', 144))
            else:
                init_C, init_theta, _ = get_init_pos(inter_module=self.inter_module, 
                                                        samples_pos=self.cam_pos_pre_rendered, 
                                                        samples_theta=self.theta_pre_rendered, 
                                                        predicted_maps=feature_map, 
                                                        clutter_scores=clutter_score, 
                                                        reset_distance=distance if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device))

            if principal is not None and dof == 3:
                init_C = init_C / init_C.pow(2).sum(-1).pow(.5)[..., None] * distance_source[..., None].float()

        # 6 DoF
        else:
            raise NotImplementedError("6 DoF not implemented")
            assert principal is None
            assert self.pre_render

            principal = self.samples_principal
            maps_target_shape = inter_module.rasterizer.cameras.image_size 

            with torch.no_grad():
                all_init_C, all_init_theta, all_init_loss = [], [], []
                for principal_ in principal:
                    n = feature_map.shape[0]
                    distance_source = torch.ones(feature_map.shape[0]).to(feature_map.device)
                    principal_ = principal_[None].expand(n, -1).float()

                    # Note it is correct to use distance_source as target since we do not want to rescale the feature map here. The actual distance is controlled in C
                    t_feature_map = align_no_centered(maps_source=feature_map, principal_source=principal_, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_source, padding_mode='border')
                    t_clutter_score = align_no_centered(maps_source=clutter_score[:, None], principal_source=principal_, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_source, padding_mode='border').squeeze(1)

                    this_C, this_theta, this_loss = get_init_pos_rendered_dim0(samples_maps=self.feature_pre_rendered, 
                                                            samples_pos=self.cam_pos_pre_rendered, 
                                                            samples_theta=self.theta_pre_rendered, 
                                                            predicted_maps=t_feature_map, 
                                                            clutter_scores=t_clutter_score, 
                                                            batch_size=self.cfg.get('batch_size_no_grad', 144),)
                    
                    all_init_C.append(this_C)
                    all_init_theta.append(this_theta)
                    all_init_loss.append(this_loss)

                use_indexes = torch.min(torch.stack(all_init_loss), dim=0)[1]
                init_C = torch.gather(torch.stack(all_init_C), dim=0, index=use_indexes.view(1, -1, 1).expand(-1, -1, 3)).squeeze(0)
                init_theta = torch.gather(torch.stack(all_init_theta), dim=0, index=use_indexes.view(1, -1)).squeeze(0)
                init_principal = torch.gather(principal, dim=0, index=use_indexes.view(-1, 1).expand(-1, 2)).float()
        
        return init_C, init_theta, init_principal


class NeMoSampleProposal():
    def __init__(self, kp_projector, inter_module, cfg, azum_samples, elev_samples, theta_samples, distance_samples, px_samples, py_samples, dof, down_sample_rate=1, pre_render=True, device='cpu', **kwargs):
        assert enable_cunemo
        map_shape = kp_projector.raster.raster_settings.image_size
        self.map_shape = (map_shape[0] // down_sample_rate, map_shape[1] // down_sample_rate)
        self.dof = dof
        self.verts_features = inter_module.memory_bank.to(device)
        self.cfg = cfg

        if dof == 3:
            assert distance_samples.shape[0] == 1
            self.record_distance = distance_samples[0]

        # (x, y)
        principal_samples = torch.stack((torch.from_numpy(px_samples).view(-1, 1).expand(-1, py_samples.shape[0]) * map_shape[1], torch.from_numpy(py_samples).view(1, -1).expand(px_samples.shape[0], -1) * map_shape[0]), ).view(2, -1).T.float().to(device)
        azum_samples = torch.from_numpy(azum_samples).to(device).float()
        elev_samples = torch.from_numpy(elev_samples).to(device).float()
        theta_samples = torch.from_numpy(theta_samples).to(device).float()
        distance_samples = torch.from_numpy(distance_samples).to(device).float()

        if dof <= 4:
            principal_samples = torch.zeros_like(principal_samples)

        out_shape = (azum_samples.shape[0], elev_samples.shape[0], theta_samples.shape[0], distance_samples.shape[0], )
        
        azum_samples = azum_samples[:, None, None, None, ].expand(*out_shape).contiguous().view(-1)
        elev_samples = elev_samples[None, :, None, None, ].expand(*out_shape).contiguous().view(-1)
        theta_samples = theta_samples[None, None, :, None, ].expand(*out_shape).contiguous().view(-1)
        distance_samples = distance_samples[None, None, None, :, ].expand(*out_shape).contiguous().view(-1)

        self.cam_pos_samples = camera_position_from_spherical_angles(distance_samples, elev_samples, azum_samples, degrees=False, device=device)[:, None].expand(-1, principal_samples.shape[0], -1).contiguous().view(-1, 3)
        self.theta_samples = theta_samples[:, None].expand(-1, principal_samples.shape[0]).contiguous().view(-1)
        self.principal_samples = principal_samples[None].expand(theta_samples.shape[0], -1, -1).contiguous().view(-1, 2) / down_sample_rate
        
        batch_size = cfg.get('batch_size_no_grad', 200)
        batchifier = Batchifier(batch_size=batch_size, batch_args=('azim', 'elev', 'dist', 'theta', ), target_dims=(0, ))

        with torch.no_grad():
            # (y, x) => (x, y)
            # Avoid invisible verts due to out of bound -> 6.5 should be enough test on ori-bus
            kp_get, _ = kp_projector(azim=azum_samples, elev=elev_samples, dist=distance_samples, theta=theta_samples, restrict_to_boundary=False, kp_only=True, down_rate=1)
            _, vis_get = batchifier(kp_projector)(azim=azum_samples, elev=elev_samples, dist=torch.ones_like(distance_samples) * 6.5, theta=theta_samples, restrict_to_boundary=False)
            kp_get = (kp_get[:, None].flip(-1) - kp_projector.cameras.principal_point.view(2)) + principal_samples.view(1, -1, 1, 2)
            kp_get = (kp_get / down_sample_rate).contiguous().view(-1, vis_get.shape[1], 2).contiguous()
            vis_get = vis_get[:, None].expand(-1, principal_samples.shape[0], -1).contiguous().view(-1, vis_get.shape[1])

        self.kp_get = kp_get
        self.vis_get = vis_get

        if dof == 6:
            eps = 1e-5
            max_size = torch.Tensor(self.map_shape[::-1]).to(self.kp_get.device)  # (x, y)
            inner_mask = torch.min(max_size - 1 > self.kp_get, dim=-1)[0] & torch.min(0 < self.kp_get, dim=-1)[0]
            self.vis_get = self.vis_get & inner_mask
            self.kp_get = torch.min(self.kp_get, max_size * torch.ones_like(self.kp_get) - 1 - eps)
            self.kp_get = torch.max(self.kp_get, torch.zeros_like(self.kp_get) + eps)
        self.down_sample_rate = down_sample_rate

    def __call__(self, feature_map, clutter_score=None, principal=None, distance=None, **kwargs):
        # principal -> (x, y)
        dof = self.dof
        
        if principal is not None:
            principal = principal / self.down_sample_rate

        if dof <= 4:
            if dof == 3 and distance is not None:
                assert principal is not None
                # 3D ori
                # (N, K, 2) => (B, N, K, 2)
                kp_get_ = self.kp_get[None] * self.record_distance / distance[:, None, None, None]
            else:
                kp_get_ = self.kp_get
                
            if principal is None:
                principal = torch.Tensor([self.map_shape[1] // 2, self.map_shape[0] // 2]).to(self.kp_get.device)
                kp_get_ = kp_get_ + principal # (N, K, 2)
            else:
                kp_get_ = kp_get_ + principal[:, None, None] # (B, N, K, 2)
            
            vis_get_ = self.vis_get
            eps = 1e-5
            max_size = torch.Tensor(self.map_shape[::-1]).to(kp_get_.device)
            inner_mask = torch.min(max_size - 1 > kp_get_, dim=-1)[0] & torch.min(0 < kp_get_, dim=-1)[0]
            vis_get_ = vis_get_ & inner_mask
            kp_get_ = torch.min(kp_get_, max_size * torch.ones_like(kp_get_) - 1 - eps)
            kp_get_ = torch.max(kp_get_, torch.zeros_like(kp_get_) + eps)
            
        else: # 6 dof
            kp_get_ = self.kp_get
            vis_get_ = self.vis_get[None]

        # feature_map: (N, C, H, W), kp_scores: (B, K_all, H, W)
        kp_scores = torch.nn.functional.conv2d(feature_map, self.verts_features[..., None, None], )

        # out_score: (B, L, N, K)
        out_score = fast_collect_score(kp_locations=kp_get_, kp_score=kp_scores, clutter_score=clutter_score if self.cfg.get('clutter_in_init', False) else -torch.ones_like(clutter_score), kp_indexs=None)

        scores_ = torch.sum(out_score[:, 0] * vis_get_.float(), dim=2) / torch.sum(vis_get_.float(), dim=2)
        sel_idx = scores_.max(1)[1]

        init_C = torch.gather(self.cam_pos_samples, dim=0, index=sel_idx[:, None].expand(-1, 3))
        init_theta = torch.gather(self.theta_samples, dim=0, index=sel_idx)

        if dof <= 4:
            init_principal = principal
        else:
            init_principal = torch.gather(self.principal_samples, dim=0, index=sel_idx[:, None].expand(-1, 2))
            
        return init_C, init_theta, init_principal


def grid_sample_similarity(kp_locations, kp_features, feature_maps, clutter_score):
    # out_score1 = grid_sample_similarity(kp_locations=kp_get_, kp_features=self.verts_features, feature_maps=feature_map, clutter_score=clutter_score if self.cfg.get('clutter_in_init', False) else -torch.ones_like(clutter_score))
    kp_locations = kp_locations / torch.Tensor(list(feature_maps.shape[2:])).to(feature_maps.device) * 2 - 1
    get_features = torch.nn.functional.grid_sample(feature_maps, kp_locations)

    score_obj = torch.einsum("bnkc,kc->bnk", get_features.permute(0, 2, 3, 1), kp_features)

    score_clu = torch.nn.functional.grid_sample(clutter_score[:, None], kp_locations)[:, 0]

    return torch.max(score_obj, score_clu)
