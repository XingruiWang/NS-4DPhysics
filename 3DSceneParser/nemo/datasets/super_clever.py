import sys
sys.path.append('./')

import numpy as np
import os
import torch
import torchvision
import json
from PIL import Image
import cv2
from tqdm import tqdm
from pytorch3d.renderer import (
    MeshRenderer, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
    PerspectiveCameras, TexturesVertex)
from pytorch3d.transforms import Transform3d, Translate, Scale, matrix_to_axis_angle

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

        # load scene files
        # self.file_list = ['superCLEVR_new_%06d' % i for i in range(120, 140)]
        if training:
            scene_list = self.kwargs['train_scene_list']
        else:
            scene_list = self.kwargs['val_scene_list']
        print(scene_list)
        scene_list = [1000, 1001]
        self.scene_list  = ['super_clever_%0d' % i for i in range(scene_list[0], scene_list[1])]
        # load all frames
        if training:
            frame_lst_path = "all_frames.txt"
        else:
            frame_lst_path = "val_frames.txt"
        if os.path.exists(os.path.join(self.root_path, frame_lst_path)):
            with open(os.path.join(self.root_path, frame_lst_path), "r") as f:
                all_frames_path = f.readlines()
            all_frames_path = [p.strip('\n') for p in all_frames_path]
        else:
            frame_path = []
            for name_scene in tqdm(['super_clever_%0d' % i for i in range(scene_list[0], scene_list[1])]):
                scene_path = os.path.join(self.root_path, name_scene)
                
                metadata = json.load(open(os.path.join(scene_path, 'metadata.json')))
                for f in range(metadata['metadata']['num_frames']):
                    file_name = os.path.join(name_scene, "rgba_%05d.png" % f)
                    instance_visible = [metadata['instances'][i]['visibility'][f] > 100 for i in range(metadata['metadata']['num_instances'])]
                    if np.any(instance_visible):
                        frame_path.append(file_name)
                    else:
                        continue


            frame_path = sorted(frame_path)

            write_content = '\n'.join(frame_path)
            with open(os.path.join(self.root_path, frame_lst_path), "w") as f:
                f.write(write_content)
            all_frames_path = frame_path

        if training:
            self.all_frames_path = [i for i in all_frames_path if i.split('/')[0] in self.scene_list and \
                                                                int(i.split('/')[1].split('_')[1].split('.')[0]) % 2 == 0 and \
                                                                int(i.split('/')[1].split('_')[1].split('.')[0]) < 90]
        else:
            self.all_frames_path = [i for i in all_frames_path if i.split('/')[0] in self.scene_list]
        # import ipdb; ipdb.set_trace()
        # print(self.all_frames_path)
        # self.image_path = os.path.join(self.root_path, "images")
        # self.annotation_path = os.path.join(self.root_path, "scenes")
        # self.mask_path = os.path.join(self.root_path, "masks")
        self.max_objects_in_scenes = max_objects_in_scenes

        # frame should be selected randomly
        # self.frame = 20
        self.anno = {}
    
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
        # return len(self.scene_list)
        return len(self.all_frames_path)

    def __getitem__(self, item):

        name_img = self.all_frames_path[item]
        frame_id = int(name_img.split('_')[-1].split('.')[0])
        name_scene = name_img.split('/')[0]
        # self.frame = self.random_frame(name_scene)
        
        # name_img = name_scene + '_rgba_%05d' % self.frame
        
        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
            scene_path = os.path.join(self.root_path, name_scene)
            mask_path = os.path.join(scene_path, 'masks')
            
            # obj_mask = cv2.imread(os.path.join(mask_path, 'mask_%05d' % frame_id + '.png'), cv2.IMREAD_GRAYSCALE) / 255
            img = Image.open(os.path.join(self.root_path, name_img))
            
            if img.mode != "RGB":
                img = img.convert("RGB")

            sample['obj_mask'] = obj_mask
            sample['img'] = img
            sample['original_img'] = np.array(img)
            obj_mask=np.ones((img.size[1], img.size[0]))
            

        else:
            scene_path = os.path.join(self.root_path, name_scene)
            img = Image.open(os.path.join(self.root_path, name_img))
            if img.mode != "RGB":
                img = img.convert("RGB")
            if scene_path not in self.anno.keys():
                anno = json.load(open(os.path.join(scene_path, 'metadata' + '.json')))
                # self.anno[scene_path] = anno
            else:
                anno = self.anno[scene_path]
            # img = img.resize((960, 1280))
            
            img = img.resize(( 1280, 960))

            mask_path = os.path.join(scene_path, 'masks')
            # obj_mask = np.array(Image.open(os.path.join(mask_path, 'mask_%05d' % frame_id + '.png'))) / 255
            # obj_mask = cv2.imread(os.path.join(mask_path, 'mask_%05d' % frame_id + '.png'), cv2.IMREAD_GRAYSCALE) / 255
            obj_mask = np.ones((img.size[1], img.size[0]))
            R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 
            
            C = torch.Tensor(anno['camera']['positions'][frame_id])[None] @ R__ 
            # R = look_at_rotation(C, )
            p1, p2, p3 = anno['camera']['look_at']
            R = look_at_rotation(C, at=([p1, p3, -p2],))
            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

            sub_cates_indexs = [self.mesh_sub_cates.index(anno['instances'][i]['asset_id']) for i in range(len(anno['instances']))] + [-1 for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            # print(len(self.mesh_sub_cates)) 18
            label = np.array(sub_cates_indexs)
            # print(label)

            objects_transfroms = []
            objects_angles = []
            objects_translations = []
            objects_scales = []
            objects_visibility = []

            objects_velocities = []
            objects_colors = []


            for i in range(len(anno['instances'])):
                scale_ = anno['instances'][i]['size']
                objects_scales.append(torch.tensor([scale_]))

                trans_ = torch.Tensor(anno['instances'][i]['positions'][frame_id])[None] @ R__
                objects_translations.append(trans_)

                velocities_ = torch.Tensor(anno['instances'][i]['velocities'][frame_id])[None] @ R__
                objects_velocities.append(velocities_)
                

                quaternion = Quaternion(anno['instances'][i]['quaternions'][frame_id])
                R_ = torch.Tensor(quaternion.rotation_matrix).T @ R__
                # objects_angles.append(matrix_to_euler_angles(R_, "XYZ"))
                objects_angles.append(matrix_to_axis_angle(R_))


                objects_colors.append(anno['instances'][i]['color'])

                
                '''
                The first rotation is applied around the X-axis, which corresponds to Elevation (or pitch in aviation terms).
                The second rotation is around the Y-axis, corresponding to Azimuth (or yaw).
                The third rotation is around the Z-axis, corresponding to Theta (or roll).
                '''

                R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
                this_transforms = Transform3d(matrix=R_, ).compose(Scale(scale_, )).compose(Translate(trans_, ))
                # print(R_, torch.norm(torch.tensor(anno['instances'][i]['quaternions'][frame_id])))
                objects_transfroms.append(this_transforms.get_matrix()[0])

                visibility = int(anno['instances'][i]['visibility'][frame_id] > 0)
                objects_visibility.append(torch.tensor([visibility]))
                
                
            objects_transfroms += [torch.zeros((4, 4)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_angles += [torch.zeros((3)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_translations += [torch.zeros((1, 3)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_velocities += [torch.zeros((1, 3)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_scales += [torch.zeros((1)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_visibility += [torch.zeros((1)) for _ in range(self.max_objects_in_scenes - len(anno['instances']))]
            objects_colors += ['None' for _ in range(self.max_objects_in_scenes - len(anno['instances']))]

            objects_transfroms = torch.stack(objects_transfroms)
            objects_angles = torch.stack(objects_angles)
            objects_translations = torch.stack(objects_translations)
            objects_velocities = torch.stack(objects_velocities)
            objects_scales = torch.stack(objects_scales)
            objects_visibility = torch.stack(objects_visibility)

            sample = {
                "this_name": name_img,
                "cad_index": 0,
                "azimuth": 0,
                "elevation": 0,
                "theta": 0,
                "distance": 5,
                "R": R[0],
                "T": T[0],
                "camera": anno['camera'],
                "obj_mask": obj_mask,
                "img": img,
                "original_img": np.array(img),
                "num_objects": len(anno['instances']),
                # gt transforms
                "transforms": objects_transfroms,
                # gt labels
                "angles": objects_angles,
                "scales":objects_scales,
                "translations":objects_translations,
                "velocities":objects_velocities,
                "label": label,
                "visibility": objects_visibility,
                "colors": objects_colors,
            }

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)
                # clean the cache
                self.cache[name_img]['img'] = None
                self.cache[name_img]['original_img'] = None
                self.cache[name_img]['obj_mask'] = None

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
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    process_id = 2

    iterval = 200
    start_idx = process_id * iterval
    data_idx_max =1101
    data_idx_max = min(start_idx + iterval, data_idx_max)
    image_idx_max = 120

    device = 'cuda'

    # home_path = '/home/xingrui/projects/superclevr-physics/data/output/'
    # home_path = '/home/xingrui/projects/superclevr-physics/data/output_1k/'
    # home_path = '/home/xingrui/projects/superclevr-physics/CogAI_nemo_data/'
    home_path = '/ccvl/net/ccvl15/xingrui/output_v3_1k/'

    
    mesh_path = '/home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/data/objs_downsample/'
    
    mesh_sub_cates = [t.split('.')[0] for t in os.listdir(mesh_path)]
    mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
    
    # mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

    import tqdm
    for data_idx in tqdm.trange(start_idx, data_idx_max):
        mask_dir = home_path + 'super_clever_%0d/' % data_idx + 'masks/'
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        else:
            print("Replace the masks", data_idx)
        anno_ = json.load(open(home_path + 'super_clever_%0d/' % data_idx + 'metadata.json'))
        for image_idx in tqdm.trange(0, image_idx_max):    
            
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
                # print(anno_['instances'][i]['positions'][image_idx])
                # print(anno_['instances'][i]['asset_id'])
                # anno_['instances'][i]['positions'][image_idx] = [0, 0, 0]
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
            p1, p2, p3 = anno_['camera']['look_at']
            R = look_at_rotation(C, at=([p1, p3, -p2],), device=device)

            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

            # verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].to(device)) for k, i in enumerate(sub_cates_indexs)]
            verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].to(device)) for k, i in enumerate(sub_cates_indexs)]
            
            faces_ = [mesh_loader.all_faces[i].to(device) for i in sub_cates_indexs]
            textures = TexturesVertex([torch.ones_like(vv) for vv in verts_])
            mesh_ = join_meshes_as_scene(Meshes(verts_, faces_, textures=textures))
            # import ipdb
            # ipdb.set_trace()

            img = phong_renderer(mesh_, R= R, T=T).cpu().numpy()[0]

            rate_ = 0.8
            # img_out = img[..., :3] * img[..., 3:] * rate_ * 255 + img_[..., :3] * (1 - img[..., 3:] * rate_)

            # Image.fromarray(img_out.astype(np.uint8)).save('debug/super_clever_cc2.png')

            # Image.fromarray((img[..., 3] * 255).astype(np.uint8)).save(home_path + 'super_clever_%0d/' % data_idx + 'masks/mask_%05d.png' % image_idx)
            # resave img_out with opencvimport ipdb; ipdb.set_trace()
            cv2.imwrite(home_path + 'super_clever_%0d/' % data_idx + 'masks/mask_%05d.png' % image_idx, cv2.cvtColor((img[..., 3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            
            # import ipdb
            # ipdb.set_trace()






