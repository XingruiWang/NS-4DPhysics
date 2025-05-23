import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

from nemo.models.base_model import BaseModel
from nemo.models.feature_banks import MaskCreater, StaticLatentMananger
from nemo.models.mesh_interpolate_module import MeshInterpolateModule
from nemo.models.solve_pose import pre_compute_kp_coords
from nemo.models.supercleve_pose import pre_compute_kp_coords_clevr
from nemo.models.solve_pose import solve_pose
from nemo.models.batch_solve_pose import solve_pose as batch_solve_pose
from nemo.models.batch_solve_pose import StandardAnnoFilter
from nemo.models.project_kp import func_multi_select
from nemo.models.keypoint_representation_net import net_stride

from nemo.utils import center_crop_fun
from nemo.utils import construct_class_by_name, get_obj_by_name
from nemo.utils import get_param_samples
from nemo.utils import normalize_features
from nemo.utils import pose_error, position_error, iou, pre_process_mesh_pascal, load_off
from nemo.utils.pascal3d_utils import IMAGE_SIZES, CATEGORIES
from nemo.utils.meshloader import MeshLoader
from nemo.models.project_kp import PackedRaster

from PIL import Image


class NeMo(BaseModel):
    def __init__(
        self,
        cfg,
        cate,
        mode,
        backbone,
        memory_bank,
        num_noise,
        max_group,
        down_sample_rate,
        mesh_path,
        training,
        inference,
        proj_mode='runtime',
        checkpoint=None,
        transforms=[],
        device="cuda:0",
        **kwargs
    ):
        super().__init__(cfg, cate, mode, checkpoint, transforms, ['loss', 'loss_main', 'loss_reg'], device)
        self.net_params = backbone
        self.memory_bank_params = memory_bank
        self.num_noise = num_noise
        self.max_group = max_group
        self.down_sample_rate = down_sample_rate
        self.training_params = training
        self.inference_params = inference
        self.dataset_config = cfg.dataset
        self.accumulate_steps = 0
        self.batch_size = cfg.training.batch_size
        self.start_epoch = 0

        if cate == 'all':
            self.category = CATEGORIES
        else:
            self.category = [cate]
        
        proj_mode = self.training_params.proj_mode
        self.mesh_loader = kwargs.get('mesh_loader', None) if kwargs.get('mesh_loader', None) is not None else MeshLoader(self.category, mesh_path)
        self.all_vertices, self.all_faces = self.mesh_loader.get_mesh_para()
        
        self.num_verts = self.mesh_loader.get_max_vert()
        self.all_verts_num = self.mesh_loader.get_verts_num_list()

        if proj_mode != 'prepared':
            raster_conf = {
                        'image_size': self.dataset_config.image_sizes if isinstance(self.dataset_config.image_sizes, list) else self.dataset_config.image_sizes[self.category[0]],
                        **self.training_params.kp_projecter
            }
            if raster_conf['down_rate'] == -1:
                raster_conf['down_rate'] = net_stride[self.net_params['net_type']]

            self.projector = PackedRaster(raster_conf, [self.all_vertices, self.all_faces], device='cuda')
        
        else:
            self.projector = None
        
        self.build()
        self.net.module.kwargs['n_vert'] = self.num_verts  # Only used in VoGE training

    def build(self):
        if self.mode == "train":
            self._build_train()
        else:
            self._build_inference()

    def _build_train(self):
        self.n_gpus = torch.cuda.device_count()
        if self.training_params.separate_bank:
            self.ext_gpu = f"cuda:{self.n_gpus-1}"
        else:
            self.ext_gpu = "cuda"

        net = construct_class_by_name(**self.net_params)
        if self.training_params.separate_bank:
            self.net = nn.DataParallel(net, device_ids=[i for i in range(self.n_gpus - 1)]).cuda()
        else:
            self.net = nn.DataParallel(net).cuda()
        memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            num_pos=sum(self.all_verts_num),
            num_noise=self.num_noise,
            max_groups=self.max_group,
            mesh_n_list=self.all_verts_num)

        if self.training_params.separate_bank:
            self.memory_bank = memory_bank.cuda(self.ext_gpu)
        else:
            self.memory_bank = memory_bank.cuda()

        self.start_epoch = 0
        if self.checkpoint is not None:
            self.net.load_state_dict(self.checkpoint["state"])
            self.memory_bank.copy_memory(self.checkpoint["memory"], exact_match=True)
            self.start_epoch = self.checkpoint.get('epoch', 0)
        # Problem?
        # self.memory_bank.memory.copy_(
        #     self.checkpoint["memory"][0 : self.memory_bank.memory.shape[0]]
        # )

        self.optim = construct_class_by_name(
            **self.training_params.optimizer, params=self.net.parameters())
        self.scheduler = construct_class_by_name(
            **self.training_params.scheduler, optimizer=self.optim)

        kappas = {'pos':self.training_params.get('weight_pos', 0), 'near':self.training_params.get('weight_near', 1e5), 'class':-math.log(self.training_params.get('weight_class', 1e-20)), 'clutter': -math.log(self.training_params.weight_noise)}
        self.training_mask_creater = MaskCreater(self.training_params.distance_thr, kappas, self.num_noise * self.max_group, verts_ori=[t if torch.is_tensor(t) else torch.from_numpy(t) for t in self.all_vertices], device=self.ext_gpu)
        
    def step_scheduler(self):
        self.scheduler.step()
        self.projector.step()

    def kpts_vis(self, kpts, img, vis=None, idx = 0):
        # import ipdb; ipdb.set_trace()
        img = img[idx].cpu().numpy()
        kpts = kpts[idx]
        vis = vis[idx]
        if vis is None:
            vis = torch.ones(kpts.shape[0]).type(torch.bool)
        kpts = kpts[vis]
        kpts = kpts.cpu().numpy()
        for kpt in kpts:
            cv2.circle(img, (int(kpt[1]), int(kpt[0])), 3, (0, 0, 255), -1)
        cv2.imwrite('kpts_vis.jpg', img)
        import ipdb; ipdb.set_trace()
        return img

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()

        # indexs is useful only if func_of_mesh == func_reselect or func_of_mesh == func_multi_select
        kwargs_ = dict(indexs=sample['label'].cuda() if 'label' in sample.keys() else torch.zeros(img.shape[0]).cuda(), func_of_mesh=get_obj_by_name(name=self.training_params.get('func_of_mesh', 'nemo.models.project_kp.func_single')))
        kwargs_.update(dict(principal=sample['principal'].cuda()) if 'principal' in sample.keys() else dict())
        kwargs_.update(dict(R=sample["R"].cuda(), T=sample["T"].cuda()) if 'R' in sample.keys() and 'T' in sample.keys() else dict())

        if 'transforms' in sample.keys():
            transforms = sample['transforms']
            if torch.is_tensor(transforms):
                # transforms shape: [b, k, 4, 4]
                get_transforms = [[Transform3d(matrix=transforms[ii, jj].cuda(), device='cuda') for jj in range(sample['num_objects'][ii])] for ii in range(transforms.shape[0])]
            else:
                get_transforms = transforms
            kwargs_.update(dict(transforms=get_transforms))

        if self.training_params.proj_mode == 'prepared':
                kp = sample['kp'].cuda()
                kpvis = sample["kpvis"].cuda().type(torch.bool)
        else:
            with torch.no_grad():
                kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)             
        
        features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True,)
        import ipdb; ipdb.set_trace()
        if self.training_params.separate_bank:
            features = features.to(self.ext_gpu)
            kpvis, kp = kpvis.to(self.ext_gpu), kp.to(self.ext_gpu)
            labels = sample['label'].to(self.ext_gpu) if 'label' in sample.keys() else None
        else:
            labels = sample['label'].cuda() if 'label' in sample.keys() else None

        # self.kpts_vis(kp, sample['original_img'], kpvis)

        if self.projector is not None and 'voge' in self.projector.raster_type:
            kpvis_bool = kpvis > self.projector.kp_vis_thr
        else:
            kpvis_bool = kpvis

        feature_similarity, noise_similarity = self.memory_bank(features, kpvis, object_labels=labels, vis_mask=kpvis_bool)

        feature_similarity /= self.training_params.T
        mask_distance_legal, y_idx = self.training_mask_creater(sample_indexs=labels, 
                                                                vis_mask=kpvis_bool, 
                                                                kps=None if self.training_params.remove_near_mode == 'vert' else kp, 
                                                                dtype_template=kpvis_bool)
        # import ipdb; ipdb.set_trace()
        loss_main = nn.CrossEntropyLoss().to(self.ext_gpu)(feature_similarity - mask_distance_legal, y_idx)

        if self.num_noise > 0:
            loss_reg = torch.mean(noise_similarity) * self.training_params.loss_reg_weight
            loss = loss_main + loss_reg
        else:
            loss_reg = torch.zeros(1)
            loss = loss_main

        loss.backward()

        self.accumulate_steps += 1
        if self.accumulate_steps % self.training_params.train_accumulate == 0:
            self.optim.step()
            self.optim.zero_grad()

        self.loss_trackers['loss'].append(loss.item())
        self.loss_trackers['loss_main'].append(loss_main.item())
        self.loss_trackers['loss_reg'].append(loss_reg.item())

        return {'loss': loss.item(), 'loss_main': loss_main.item(), 'loss_reg': loss_reg.item()}

    def _build_inference(self):
        assert len(self.category) == 1, "During inference, category must be specificed."
        self.net = construct_class_by_name(**self.net_params)
        self.net = nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(self.checkpoint["state"])
        self.memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            num_pos=sum(self.all_verts_num),
            num_noise=0,
            max_groups=self.max_group,
            mesh_n_list=self.all_verts_num
        ).to(self.device)

        with torch.no_grad():
            self.memory_bank.copy_memory(self.checkpoint["memory"])
            # self.memory_bank.memory.copy_(
            #     self.checkpoint["memory"][0 : self.memory_bank.memory.shape[0]]
            # )
        self.feature_bank = self.checkpoint["memory"][0 : self.memory_bank.memory.shape[0]].detach()
        self.clutter_bank = self.checkpoint["memory"][self.memory_bank.memory.shape[0] :].detach().to(self.device)
        self.clutter_bank = normalize_features(
            torch.mean(self.clutter_bank, dim=0)
        ).unsqueeze(0)
        self.kp_features = self.checkpoint["memory"][
            0 : self.memory_bank.memory.shape[0]
        ].to(self.device)

        image_h, image_w = self.dataset_config.image_sizes[self.category[0]] if isinstance(self.dataset_config.image_sizes, dict) else self.dataset_config.image_sizes
        
        map_shape = (image_h // self.down_sample_rate, image_w // self.down_sample_rate)

        if self.inference_params.cameras.get('image_size', 0) == -1:
            self.inference_params.cameras['image_size'] = (map_shape, )
        if self.inference_params.cameras.get('principal_point', 0) == -1:
            self.inference_params.cameras['principal_point'] = ((map_shape[1] // 2, map_shape[0] // 2), )
        if self.inference_params.cameras.get('focal_length', None) is not None:
            self.inference_params.cameras['focal_length'] = self.inference_params.cameras['focal_length'] / self.down_sample_rate

        cameras = construct_class_by_name(**self.inference_params.cameras, device=self.device)
        raster_settings = construct_class_by_name(
            **self.inference_params.raster_settings, image_size=map_shape
        )
        if self.inference_params.rasterizer.class_name == 'VoGE.Renderer.GaussianRenderer':
            rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, render_settings=raster_settings
            )
        else:
            rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, raster_settings=raster_settings
            )
        self.inter_module = construct_class_by_name(
            class_name=self.inference_params.get('mesh_interpolate_module', 'nemo.models.mesh_interpolate_module.MeshInterpolateModule'),
            mesh_loader=self.mesh_loader, 
            memory_bank=self.feature_bank,
            rasterizer=rasterizer,
            post_process=center_crop_fun(map_shape, (render_image_size,) * 2) if self.inference_params.get('center_crop', False) else None,
            convert_percentage=self.inference_params.get('convert_percentage', 0.5)
        ).to(self.device)

        (
            translation_samples,
            azimuth_samples,
            elevation_samples,
            theta_samples,
            distance_samples,
            px_samples,
            py_samples,
        ) = get_param_samples(self.cfg)

        if self.cfg.task == '3d_pose_estimation_clevr':
            init_samples, kp_coords, kp_vis = pre_compute_kp_coords_clevr(
                self.cfg.model.mesh_path, 
                azimuth_samples,
                elevation_samples,
                theta_samples,
                distance_samples,
                translation_samples=translation_samples
                )# can add rotation for the 6d
        
        self.init_mode = self.cfg.inference.get('init_mode', '3d_batch')

        dof = int(self.init_mode.split('d_')[0])
        self.proposal_module = construct_class_by_name(
            **self.inference_params.proposal, 
            kp_projector=self.projector,
            kp_features=self.kp_features,
            inter_module=self.inter_module, 
            cfg=self.cfg, 
            init_samples=init_samples,
            kp_coords=kp_coords, 
            kp_vis=kp_vis,
            azum_samples=azimuth_samples, 
            elev_samples=elevation_samples, 
            theta_samples=theta_samples, 
            distance_samples=distance_samples, 
            px_samples=px_samples, 
            py_samples=py_samples, 
            dof=dof, 
            pre_render=self.cfg.inference.get('pre_render', True), 
            down_sample_rate=self.down_sample_rate,
            device=self.device
        )
        
        if self.inference_params.get('anno_filter', None) is not None:
            self.anno_filter = construct_class_by_name(**self.inference_params.anno_filter, dof=dof)
        else:
            self.anno_filter = StandardAnnoFilter(dof=dof, device=self.device)
        self.pose_solver = get_obj_by_name(self.inference_params.get('pose_solver', 'nemo.models.batch_solve_pose.solve_pose'))
        self.physics_simulator = construct_class_by_name(**self.inference_params.physics_simulator)

    def transform_physics_prediction(self, next_frame_prediction, device):
        # Initialize a new dictionary to match the format of prev_pred['super_clever_1000']
        transformed_pred = {
            'num_objects': torch.tensor(len(next_frame_prediction['object_name_idx'])),  # Based on the number of objects
            'obj_idxs': torch.tensor(next_frame_prediction['object_name_idx']),
            'loss': None,  # Placeholder, assuming not available in next_frame_prediction
            'distance_loss': None  # Placeholder, assuming not available in next_frame_prediction
        }

        # Transform translations and rotations to match the expected tensor shape and fill them
        translation_tensors = [t.unsqueeze(0) for t in next_frame_prediction['translations']]
        rotation_tensors = [r.unsqueeze(0) for r in next_frame_prediction['rotations']]
        scales_tensor = torch.stack(next_frame_prediction['sizes'])

        # Concatenate all object translations and rotations to a single tensor each, padding with zeros to match the shape in the example
        max_objects = 10  # Assuming a fixed max number of objects based on the example
        translation_tensor = torch.cat(translation_tensors + [torch.zeros(1, 3) for _ in range(max_objects - len(translation_tensors))])
        rotation_tensor = torch.cat(rotation_tensors + [torch.zeros(1, 3) for _ in range(max_objects - len(rotation_tensors))])
        scales_tensor = torch.cat([scales_tensor, torch.zeros(max_objects - len(scales_tensor), device=scales_tensor.device)])

        # Update the transformed prediction dictionary
        transformed_pred['translation'] = translation_tensor.to(device)
        transformed_pred['rotation'] = rotation_tensor.to(device)
        transformed_pred['scales'] = scales_tensor.to(device)

        return transformed_pred

    def evaluate(self, sample, prev_pred = None, debug=False, repeat= 1):
        start_time = time.time()

        self.net.eval()

        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        img_name = sample["this_name"]
        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        dof = int(self.init_mode.split('d_')[0])

        # set up the physics simulator
        # obly support batch size 1
        # scene_names = [img_n.split('/')[0] for img_n in img_name]
        # frame_names = [img_n.split('/')[1].split('.')[0] for img_n in img_name]
        
        preds = self.pose_solver(
            self.cfg,
            feature_map,
            self.inter_module,
            self.proposal_module, 
            clutter_bank=self.clutter_bank,
            device=self.device,
            dof=dof,
            img_name = img_name,
            prev_pred=prev_pred,
            **self.anno_filter(sample) # Avoid leak annotations to the inference process
        )
        
        scene_name, frame_name = preds[0]['final']['scene'].split('/')
        frame_name = frame_name.split('.')[0]

        start_time = time.time()
        with torch.no_grad():
            self.physics_simulator.update_pred(preds[0])

        if frame_name == 'rgba_00000':
            self.physics_simulator.setup_scene(scene_name)

        c_frame_id = int(frame_name.split('_')[-1])
        n_frame_id = c_frame_id + 1
        self.physics_simulator.calculate_velocity(scene_name, c_frame_id)

        next_frame_out, collisions = self.physics_simulator.predict_next_frame(scene_name, n_frame_id, repeat)
        physics_prediction = self.transform_physics_prediction(
                next_frame_out,
                device = self.device
            )
        if isinstance(preds, dict):
            preds = [preds]
        
        to_print = []
        classification_result = {}
        
        with torch.no_grad():
            for i, pred in enumerate(preds):
                L = int(preds[i]['final']['num_objects'])
                gt_3d_poses = []

                for l in range(L):
                    
                    azimuth_gt, elevation_gt, theta_gt, scale_gt, visibility_gt = (
                        float(sample["angles"][i, l, 1]),
                        float(sample["angles"][i, l, 0]),
                        float(sample["angles"][i, l, 2]),
                        float(sample["scales"][i, l, 0]),
                        int(sample["visibility"][i, l, 0]),
                    )
                    translation_gt = sample["translations"][i, l, 0].cpu().numpy()
                    gt_3d_poses.append({'azimuth': azimuth_gt, 'elevation': elevation_gt, 'theta': theta_gt, \
                                        'scale': scale_gt, 'translation': translation_gt, \
                                        'visibility': visibility_gt
                                        })
                
                # preds[0]['final']['rotation'].shape = torch.Size([10, 4])
                pred_3d_poses = []
                for l in range(L):
                    azimuth_pred, elevation_pred, theta_pred, scale_pred = (
                        float(preds[i]['final']['rotation'][l, 1]),
                        float(preds[i]['final']['rotation'][l, 0]),
                        float(preds[i]['final']['rotation'][l, 2]),
                        float(preds[i]['final']['scales'][l]),
                    )
                    translation_pred = preds[i]['final']['translation'][l].detach().cpu().numpy()
                    pred_3d_poses.append({'azimuth': azimuth_pred, 'elevation': elevation_pred, 'theta': theta_pred, 'scale': scale_pred, 'translation': translation_pred})

                pose_errors = [pose_error(gt_3d_poses[l], pred_3d_poses[l]) for l in range(L) if gt_3d_poses[l]['visibility']]
                position_errors = [position_error(gt_3d_poses[l], pred_3d_poses[l]) for l in range(L) if gt_3d_poses[l]['visibility']]
                
                pred["pose_errors"] = pose_errors
                pred["position_errors"] = position_errors
        time_4 = time.time()

            # import ipdb; ipdb.set_trace()
            # if "azimuth" in sample and "elevation" in sample and "theta" in sample:
            #     pose_error_ = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]}, pred["final"][0])
            #     pred["pose_error"] = pose_error_
            #     classification_result[sample['this_name'][i]] = (pred['final'][0]['score'], pose_error_)

        #         to_print.append(pose_error_)
            # else:
                # TODO: 
                # import ipdb; ipdb.set_trace()

        return (preds, physics_prediction, collisions), classification_result


    def demo(self, sample, prev_pred = None, debug=False, repeat= 1):
        start_time = time.time()

        self.net.eval()

        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        img_name = sample["this_name"]
        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        dof = int(self.init_mode.split('d_')[0])

        # set up the physics simulator
        # obly support batch size 1
        # scene_names = [img_n.split('/')[0] for img_n in img_name]
        # frame_names = [img_n.split('/')[1].split('.')[0] for img_n in img_name]
        
        preds = self.pose_solver(
            self.cfg,
            feature_map,
            self.inter_module,
            self.proposal_module, 
            clutter_bank=self.clutter_bank,
            device=self.device,
            dof=dof,
            img_name = img_name,
            prev_pred=prev_pred,
            **self.anno_filter(sample) # Avoid leak annotations to the inference process
        )
        
        scene_name, frame_name = preds[0]['final']['scene'].split('/')
        frame_name = frame_name.split('.')[0]

        start_time = time.time()
        with torch.no_grad():
            self.physics_simulator.update_pred(preds[0])

        if frame_name == 'rgba_00000':
            self.physics_simulator.setup_scene(scene_name)

        c_frame_id = int(frame_name.split('_')[-1])
        n_frame_id = c_frame_id + 1
        self.physics_simulator.calculate_velocity(scene_name, c_frame_id)

        next_frame_out, collisions = self.physics_simulator.predict_next_frame(scene_name, n_frame_id, repeat)
        physics_prediction = self.transform_physics_prediction(
                next_frame_out,
                device = self.device
            )
        if isinstance(preds, dict):
            preds = [preds]
        
        to_print = []
        classification_result = {}
        
        with torch.no_grad():
            for i, pred in enumerate(preds):
                L = int(preds[i]['final']['num_objects'])
                gt_3d_poses = []

                for l in range(L):
                    
                    azimuth_gt, elevation_gt, theta_gt, scale_gt, visibility_gt = (
                        float(sample["angles"][i, l, 1]),
                        float(sample["angles"][i, l, 0]),
                        float(sample["angles"][i, l, 2]),
                        float(sample["scales"][i, l, 0]),
                        int(sample["visibility"][i, l, 0]),
                    )
                    translation_gt = sample["translations"][i, l, 0].cpu().numpy()
                    gt_3d_poses.append({'azimuth': azimuth_gt, 'elevation': elevation_gt, 'theta': theta_gt, \
                                        'scale': scale_gt, 'translation': translation_gt, \
                                        'visibility': visibility_gt
                                        })
                
                # preds[0]['final']['rotation'].shape = torch.Size([10, 4])
                pred_3d_poses = []
                for l in range(L):
                    azimuth_pred, elevation_pred, theta_pred, scale_pred = (
                        float(preds[i]['final']['rotation'][l, 1]),
                        float(preds[i]['final']['rotation'][l, 0]),
                        float(preds[i]['final']['rotation'][l, 2]),
                        float(preds[i]['final']['scales'][l]),
                    )
                    translation_pred = preds[i]['final']['translation'][l].detach().cpu().numpy()
                    pred_3d_poses.append({'azimuth': azimuth_pred, 'elevation': elevation_pred, 'theta': theta_pred, 'scale': scale_pred, 'translation': translation_pred})

                pose_errors = [pose_error(gt_3d_poses[l], pred_3d_poses[l]) for l in range(L) if gt_3d_poses[l]['visibility']]
                position_errors = [position_error(gt_3d_poses[l], pred_3d_poses[l]) for l in range(L) if gt_3d_poses[l]['visibility']]
                
                pred["pose_errors"] = pose_errors
                pred["position_errors"] = position_errors
        time_4 = time.time()

            # import ipdb; ipdb.set_trace()
            # if "azimuth" in sample and "elevation" in sample and "theta" in sample:
            #     pose_error_ = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]}, pred["final"][0])
            #     pred["pose_error"] = pose_error_
            #     classification_result[sample['this_name'][i]] = (pred['final'][0]['score'], pose_error_)

        #         to_print.append(pose_error_)
            # else:
                # TODO: 
                # import ipdb; ipdb.set_trace()

        return (preds, physics_prediction, collisions), classification_result

    def visualize(self, sample, debug=False):
        self.net.eval()

        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        img_name = sample["this_name"]
        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        dof = int(self.init_mode.split('d_')[0])
        # Default: nemo.models.batch_solve_pose.solve_pose
        preds = self.pose_solver(
            self.cfg,
            feature_map,
            self.inter_module,
            self.proposal_module,
            clutter_bank=self.clutter_bank,
            device=self.device,
            dof=dof,
            img_name = img_name,
            **self.anno_filter(sample) # Avoid leak annotations to the inference process
        )
        
    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt

    def predict_inmodal(self, sample, visualize=False):
        self.net.eval()

        # sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        assert len(img) == 1, "The batch size during validation should be 1"

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        clutter_score = None
        if not isinstance(self.clutter_bank, list):
            clutter_bank = [self.clutter_bank]
        for cb in clutter_bank:
            _score = (
                torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3))
                .squeeze(0)
                .squeeze(0)
            )
            if clutter_score is None:
                clutter_score = _score
            else:
                clutter_score = torch.max(clutter_score, _score)

        nkpt, c = self.kp_features.shape
        feature_map_nkpt = feature_map.expand(nkpt, -1, -1, -1)
        kp_features = self.kp_features.view(nkpt, c, 1, 1)
        kp_score = torch.sum(feature_map_nkpt * kp_features, dim=1)
        kp_score, _ = torch.max(kp_score, dim=0)

        clutter_score = clutter_score.detach().cpu().numpy().astype(np.float32)
        kp_score = kp_score.detach().cpu().numpy().astype(np.float32)
        pred_mask = (kp_score > clutter_score).astype(np.uint8)
        pred_mask_up = cv2.resize(
            pred_mask, dsize=(pred_mask.shape[1]*self.down_sample_rate, pred_mask.shape[0]*self.down_sample_rate),
            interpolation=cv2.INTER_NEAREST)

        pred = {
            'clutter_score': clutter_score,
            'kp_score': kp_score,
            'pred_mask_orig': pred_mask,
            'pred_mask': pred_mask_up,
        }

        if 'inmodal_mask' in sample:
            gt_mask = sample['inmodal_mask'][0].detach().cpu().numpy()
            pred['gt_mask'] = gt_mask
            pred['iou'] = iou(gt_mask, pred_mask_up)

            obj_mask = sample['amodal_mask'][0].detach().cpu().numpy()
            pred['obj_mask'] = obj_mask

            # pred_mask_up[obj_mask == 0] = 0
            thr = 0.8
            new_mask = (kp_score > thr).astype(np.uint8)
            new_mask = cv2.resize(new_mask, dsize=(obj_mask.shape[1], obj_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            new_mask[obj_mask == 0] = 0
            pred['iou'] = iou(gt_mask, new_mask)
            pred['pred_mask'] = new_mask

        return pred

    def fast_inference(self, sample):
        self.net.eval()
        bbox = sample['bbox']
        cls_index = self.cate_index

        n_vertice = self.num_verts
        img = sample["img"].to(self.device)
        score = {}

        with torch.no_grad():
            all_predicted_map = self.net.module.forward_test(img)

        B, C, H, W = all_predicted_map.shape
        all_similarity = []

        for i in range(B):
            object_height = (int(bbox[i][0].item() // 8), int(bbox[i][1].item() // 8))
            object_width = (int(bbox[i][2].item() // 8), int(bbox[i][3].item() // 8))
            predicted_map = all_predicted_map[i][..., object_height[0] : object_height[1],object_width[0] : object_width[1],]
            C, H, W = predicted_map.shape
            predicted_map = predicted_map.contiguous().view(C, -1)  # [C, H, W] -> [C, HW]
            

            object_score_per_vertex = torch.matmul(self.feature_bank.to(self.device), predicted_map)  # [N, C] x [C, HW] -> [N, HW]
            clutter_score = torch.matmul(self.clutter_bank, predicted_map).view(H, W)  # [H, W]

            object_score, vertex_activation_indices = object_score_per_vertex.max(dim=0)
            similarity = torch.maximum(object_score, clutter_score.view(-1, ))          

            similarity = similarity.view(H, W)  # [HW] -> [H, W]
            similarity_score = torch.sum(similarity) / (H * W)
            all_similarity.append(similarity_score)

        return all_similarity


    def fix_init(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        if 'voge' in self.projector.raster_type:
            with torch.no_grad():
                frag_ = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)
  
            features, kpvis = self.net.forward(img, keypoint_positions=frag_, obj_mask=1 - obj_mask, do_normalize=True,)
        else:
            if self.training_params.proj_mode == 'prepared':
                kp = sample['kp'].cuda()
                kpvis = sample["kpvis"].cuda().type(torch.bool)
            else:
                with torch.no_grad():
                    kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)

            features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True,)
        return features.detach(), kpvis.detach()


class NeMoRuntimeOptimize(NeMo):
    def _build_train(self):
        super()._build_train()

        self.latent_iter_per = self.training_params.latent_iter_per
        self.latent_to_update = self.training_params.latent_to_update
        self.latent_lr = self.training_params.latent_lr

        self.n_shape_state = len(self.latent_to_update)
        self.latent_manager = StaticLatentMananger(n_latents=self.n_shape_state, to_device='cuda', store_device='cpu')
        self.latent_manager_optimizer = StaticLatentMananger(n_latents=self.n_shape_state * 2, to_device='cuda', store_device='cpu')

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        image_names = sample['this_name']
        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()

        feature_map_ = self.net.forward(img, do_normalize=True, mode=0)
        img_shape = img.shape[2::]

        # Dynamic implementation, for those in latent manager
        # params = self.latent_manager.get_latent(image_names, sample['params'].float())
        # Others
        # params = sample['params'].float()
        all_params = ['azimuth', 'elevation', 'distance', 'theta']

        get_values_updated = eval(
            'self.latent_manager.get_latent(image_names, ' +
            ', '.join(["sample['%s'].float()" % t for t in self.latent_to_update]) +
            ')'
            )
        azimuth, elevation, distance, theta = tuple([get_values_updated[self.latent_to_update.index(n_)] if n_ in self.latent_to_update else sample[n_].float() for n_ in all_params])

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        assert self.training_params.proj_mode != 'prepared'
        with torch.no_grad():
            kp, kpvis = self.projector(azim=azimuth.cuda(), elev=elevation.cuda(), dist=distance.cuda(), theta=theta.cuda(), **kwargs_)

        features = self.net.forward(feature_map_, keypoint_positions=kp, obj_mask=1 - obj_mask, img_shape=img_shape, mode=1)

        if self.training_params.separate_bank:
            get, y_idx, noise_sim = self.memory_bank(
                features.to(self.ext_gpu), index.to(self.ext_gpu), kpvis.to(self.ext_gpu)
            )
        else:
            get, y_idx, noise_sim = self.memory_bank(features, index, kpvis)
        
        if 'voge' in self.projector.raster_type:
            kpvis = kpvis > self.projector.kp_vis_thr

        get /= self.training_params.T

        kappas={'pos':self.training_params.get('weight_pos', 0), 'near':self.training_params.get('weight_near', 1e5), 'clutter': -math.log(self.training_params.weight_noise)}
        # The default manner in VoGE-NeMo
        if self.training_params.remove_near_mode == 'vert':
            vert_ = self.projector.get_verts_recent()  # (B, K, 3)
            vert_dis = (vert_.unsqueeze(1) - vert_.unsqueeze(2)).pow(2).sum(-1).pow(.5)

            mask_distance_legal = remove_near_vertices_dist(
                vert_dis,
                thr=self.training_params.distance_thr,
                num_neg=self.num_noise * self.max_group,
                kappas=kappas,
            )
            if mask_distance_legal.shape[0] != get.shape[0]:
                mask_distance_legal = mask_distance_legal.expand(get.shape[0], -1, -1).contiguous()
        # The default manner in original-NeMo
        else:
            mask_distance_legal = mask_remove_near(
                kp,
                thr=self.training_params.distance_thr
                * torch.ones((img.shape[0],), dtype=torch.float32).cuda(),
                num_neg=self.num_noise * self.max_group,
                dtype_template=get,
                kappas=kappas,
            )
        if self.training_params.get('training_loss_type', 'nemo') == 'nemo':
            loss_main = nn.CrossEntropyLoss(reduction="none").cuda()(
                (get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2]))[
                    kpvis.view(-1), :
                ],
                y_idx.view(-1)[kpvis.view(-1)],
            )
            loss_main = torch.mean(loss_main)
        elif self.training_params.get('training_loss_type', 'nemo') == 'kl_alan':
            loss_main = torch.mean((get.view(-1, get.shape[2]) * mask_distance_legal.view(-1, get.shape[2]))[kpvis.view(-1), :])

        self.optim.zero_grad()
        if self.num_noise > 0:
            loss_reg = torch.mean(noise_sim) * self.training_params.loss_reg_weight
            loss = loss_main + loss_reg
        else:
            loss_reg = torch.zeros(1)
            loss = loss_main

        loss.backward()
        self.optim.step()



        self.loss_trackers['loss'].append(loss.item())
        self.loss_trackers['loss_main'].append(loss_main.item())
        self.loss_trackers['loss_reg'].append(loss_reg.item())

        ############################ Optimize Pose During Training ##########################################
        feature_map_detached = feature_map_.detach()
        assert 'voge' in self.projector.raster_type, 'Current we only support VoGE as differentiable sampler.'

        params_dict = {'azimuth': azimuth, 'elevation': elevation, 'distance': distance, 'theta': theta}
        estep_params = [torch.nn.Parameter(params_dict[t]) for t in self.latent_to_update]
        optimizer_estep = torch.optim.Adam([{'params': t, } for t in estep_params], lr=self.latent_lr, betas=(0.8, 0.6))
        
        with torch.no_grad():
            states = self.latent_manager_optimizer.get_latent_without_default(image_names, )
            if states is not None:
                state_dicts_ = {'state': {i: {'step': 0, 'exp_avg': states[i * 2].clone(), 'exp_avg_sq': states[i * 2 + 1].clone()} for i in range(self.n_shape_state)}, 'param_groups': optimizer_estep.state_dict()['param_groups']}
                optimizer_estep.load_state_dict(state_dicts_)
            else:
                print('States not found in saved dict, this should only happens in epoch 0!')

        get_parameters = lambda param_name, params_dict=params_dict: estep_params[self.latent_to_update.index(param_name)] if param_name in self.latent_to_update else params_dict[param_name].cuda()
        all_loss_estep = []
        for latent_iter in range(self.latent_iter_per):
            optimizer_estep.zero_grad()
            kp, kpvis  = self.projector(azim=get_parameters('azimuth'), elev=get_parameters('elevation'), dist=get_parameters('distance'), theta=get_parameters('theta'), **kwargs_)
            features = self.net.forward(feature_map_detached, keypoint_positions=kp, obj_mask=1 - obj_mask, img_shape=img_shape, mode=1)

            loss_estep = self.memory_bank.compute_feature_dist(features, kpvis)
            loss_estep.backward()

            optimizer_estep.step()
            all_loss_estep.append(loss_estep.item())

        optimizer_estep.zero_grad()
        with torch.no_grad():
            state_dict_optimizer = optimizer_estep.state_dict()['state']
            self.latent_manager_optimizer.save_latent(image_names, *[state_dict_optimizer[i // 2]['exp_avg_sq' if i % 2 else 'exp_avg'] for i in range(self.n_shape_state * 2)])
            self.latent_manager.save_latent(image_names, *estep_params)

        return {'loss': loss.item(), 'loss_main': loss_main.item(), 'loss_reg': loss_reg.item(), 'loss_estep': sum(all_loss_estep)}

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        ckpt['update_latents'] = self.latent_to_update
        ckpt['latents'] = self.latent_manager.latent_set
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt
