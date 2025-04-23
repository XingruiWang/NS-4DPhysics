import sys
sys.path.append('./')

import numpy as np
import os
import torch
import torchvision
import json
from PIL import Image
from pytorch3d.renderer import (
    MeshRenderer, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
    PerspectiveCameras, TexturesVertex)
from pytorch3d.transforms import Transform3d, Translate, Scale, quaternion_to_matrix, quaternion_invert
from pytorch3d.structures import Meshes, join_meshes_as_scene
from nemo.utils import meshloader
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import copy

from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
from nemo.utils import load_off


class SuperClever(Dataset):
    def __init__(
        self,
        root_path,
        transforms,
        mesh_path,
        enable_cache=True,
        transforms_test=None,
        training=True,
        max_objects_in_scenes=10,
        **kwargs,
    ):  

        if transforms_test is None:
            transforms_test = transforms
        self.training = training
        self.root_path = get_abs_path(root_path)
        self.enable_cache = enable_cache
        self.mesh_path = mesh_path
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )
        self.transforms_test = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms_test]
        )
        self.kwargs = kwargs

        mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path)])))
        self.mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
        self.mesh_sub_cates = mesh_sub_cates
        # self.mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

        # self.file_list = ['superCLEVR_new_%06d' % i for i in range(120, 140)]
        scene_list = self.kwargs['scene_list']

        self.scene_list  = ['super_clever_%0d' % i for i in range(scene_list[0], scene_list[1])]
        self.image_path = os.path.join(self.root_path, "images")
        self.annotation_path = os.path.join(self.root_path, "scenes")
        self.mask_path = os.path.join(self.root_path, "masks")
        self.max_objects_in_scenes = max_objects_in_scenes

        # frame should be selected randomly
        # self.frame = 20
        self.detect_all_frames()

        self.cache = {}

    def detect_all_frames(self):
        '''
        Detect all frames in the dataset.
        This will return a dictionary with the scene name as the key and a list of frame ids as the value.
        '''
        self.all_frames = {}
        for name_scene in self.scene_list:
            self.all_frames[name_scene] = []
            scene_path = os.path.join(self.root_path, name_scene)
            for file in os.listdir(scene_path):
                if file.startswith('rgba_') and file.endswith('.png'):
                    frame_id = int(file.split('_')[-1].split('.')[0])
                    self.all_frames[name_scene].append(frame_id)
        return
    
    def random_frame(self, name_scene):
        frame_list = self.all_frames[name_scene]
        random_frame = np.random.choice(frame_list)
        return random_frame

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, item):

        name_scene = self.scene_list[item]
        self.frame = self.random_frame(name_scene)
        
        name_img = name_scene + '_rgba_%05d' % self.frame
        
        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
        else:
            scene_path = os.path.join(self.root_path, name_scene)
            img = Image.open(os.path.join(scene_path, 'rgba_%05d' % self.frame+ '.png'))
            if img.mode != "RGB":
                img = img.convert("RGB")
            anno = json.load(open(os.path.join(scene_path, 'metadata' + '.json')))

            mask_path = os.path.join(scene_path, 'masks')
            obj_mask = np.array(Image.open(os.path.join(mask_path, 'mask_%05d' % self.frame + '.png'))) / 255
            R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 
            
            C = torch.Tensor(anno['camera']['positions'][self.frame])[None] @ R__ 
            R = look_at_rotation(C, )
            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

            sub_cates_indexs = [self.mesh_sub_cates.index(anno['instances'][i]['asset_id']) for i in range(len(anno['instances']))] + [-1 for _ in range(self.max_objects_in_scenes - len(anno['instances']))]

            label = np.array(sub_cates_indexs)

            objects_transfroms = []

            for i in range(len(anno['instances'])):
                scale_ = anno['instances'][i]['size']
                trans_ = torch.Tensor(anno['instances'][i]['positions'][self.frame])[None] @ R__

                quaternion = Quaternion(anno['instances'][i]['quaternions'][self.frame])
                R_ = torch.Tensor(quaternion.rotation_matrix).T @ R__
                R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
                this_transforms = Transform3d(matrix=R_, ).compose(Scale(scale_, )).compose(Translate(trans_, ))
                objects_transfroms.append(this_transforms.get_matrix()[0])
            objects_transfroms += [torch.zeros((4, 4)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]

            objects_transfroms = torch.stack(objects_transfroms)

            sample = {
                "this_name": name_img,
                "cad_index": 0,
                "azimuth": 0,
                "elevation": 0,
                "theta": 0,
                "distance": 5,
                "R": R[0],
                "T": T[0],
                "obj_mask": obj_mask,
                "img": img,
                "original_img": np.array(img),
                "label": label,
                "num_objects": len(anno['instances']),
                "transforms": objects_transfroms
            }

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        if self.training:
            if self.transforms:
                sample = self.transforms(sample)
        
        else:
            if self.transforms_test:
                sample = self.transforms_test(sample)

        return sample

    def __get_img__(self, item):
        name_scene = self.scene_list[item]
        scene_path = os.path.join(self.root_path, name_scene)
        img = Image.open(os.path.join(scene_path, 'rgba_%05d' % self.frame+ '.png'))
        if img.mode != "RGB":
            img = img.convert("RGB")

        return np.array(img)

if __name__ == '__main__':
    data_idx_max = 62
    image_idx_max = 60
    device = 'cuda'

    home_path = '/home/xingrui/superclevr-physics/data/output/'
    mesh_path = home_path + 'convert/objs_downsample/'
    
    mesh_sub_cates = [t.split('.')[0] for t in os.listdir(mesh_path)]
    mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
    
    # mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

    import tqdm
    for data_idx in tqdm.trange(61, data_idx_max):
        mask_dir = home_path + 'super_clever_%0d/' % data_idx + 'masks/'
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        for image_idx in tqdm.trange(0, image_idx_max):

            anno_ = json.load(open(home_path + 'super_clever_%0d/' % data_idx + 'metadata.json'))
            # anno_['objects'] = anno_['objects'][1::]
            # import ipdb; ipdb.set_trace()
            img_ = np.array(Image.open(home_path + 'super_clever_%0d/' % data_idx + 'rgba_%05d.png' % image_idx))
            render_image_size = img_.shape[:2]

            R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to(device) 
            # cameras = PerspectiveCameras(focal_length=700, image_size=(img_.shape[:2], ), principal_point=((img_.shape[1] // 2, img_.shape[0] // 2), ), device=device, in_ndc=False)
            cameras = PerspectiveCameras(focal_length=700 * 2, image_size=(img_.shape[:2], ), principal_point=((img_.shape[1] // 2, img_.shape[0] // 2), ), device=device, in_ndc=False)

            blend_params = BlendParams(sigma=0, gamma=0)
            raster_settings = RasterizationSettings(
                image_size=render_image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0
            )
            # We can add a point light in front of the object.
            phong_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(device=device, cameras=cameras),
            )

            sub_cates_indexs = [mesh_sub_cates.index(anno_['instances'][i]['asset_id']) for i in range(len(anno_['instances']))]

            # collected_cads = [mesh_loader.all_vertices[i] for i in sub_cates_indexs]
            objects_transfroms = []

            for i in range(len(sub_cates_indexs)):
                scale_ = anno_['instances'][i]['size']
                # trans_ = (torch.Tensor(anno_['objects'][i]['3d_coords'])[None] @ R__.cpu()).squeeze()
                trans_ = torch.Tensor(anno_['instances'][i]['positions'][image_idx])[None] @ R__.cpu()

                # scale_matrix = torch.Tensor([[scale_, 0, 0, trans_[0]], [0, scale_, 0, trans_[1]], [0, 0, scale_, trans_[2]], [0, 0, 0, 1]])
                quaternion = Quaternion(anno_['instances'][i]['quaternions'][image_idx])
                R_ = torch.Tensor(quaternion.rotation_matrix).T @ R__.cpu()
                # R_ = look_at_view_transform(dist=1, azim=-anno_['instances'][i]['rotation'] + 180)[0][0]
                R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
                # K_ = scale_matrix # @ R_  # P' = P @ K_
                
                # objects_transfroms.append(Transform3d(matrix=K_.to(device), device=device))
                objects_transfroms.append(Transform3d(matrix=R_.to(device), device=device).compose(Scale(scale_, device=device)).compose(Translate(trans_.to(device), device=device)))
                
            # rr = Rotation.from_euler('XYZ', anno_['camera']["rotation_euler"], degrees=False).as_matrix()
            # import ipdb; ipdb.set_trace()
            C = torch.Tensor(anno_['camera']['positions'][image_idx]).to(device)[None] @ R__ 
            R = look_at_rotation(C, device=device)
            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

            # verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].to(device)) for k, i in enumerate(sub_cates_indexs)]
            verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].to(device)) for k, i in enumerate(sub_cates_indexs)]
            
            faces_ = [mesh_loader.all_faces[i].to(device) for i in sub_cates_indexs]
            textures = TexturesVertex([torch.ones_like(vv) for vv in verts_])
            mesh_ = join_meshes_as_scene(Meshes(verts_, faces_, textures=textures))
            # import ipdb
            # ipdb.set_trace()

            img = phong_renderer(mesh_, R=R, T=T).cpu().numpy()[0]

            rate_ = 0.8
            img_out = img[..., :3] * img[..., 3:] * rate_ * 255 + img_[..., :3] * (1 - img[..., 3:] * rate_)
            # Image.fromarray(img_out.astype(np.uint8)).save(home_path + 'debug/super_clever_cc2.png')

            # Image.fromarray((img[..., 3] * 255).astype(np.uint8)).save(home_path + 'super_clever_%0d/' % data_idx + 'masks/mask_%05d.png' % image_idx)

            import ipdb
            ipdb.set_trace()






