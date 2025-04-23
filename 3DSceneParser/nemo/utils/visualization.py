import sys
sys.path.append('./')

import numpy as np
import os
import torch
import torchvision
import json
from PIL import Image, ImageDraw
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

import cv2
COLORMAP = {
    "gray": [153, 153, 153],
    "red": [255, 51, 51],
    "brown": [183, 84, 84],
    "yellow": [255, 255, 51],
    "green": [51, 153, 51],
    "cyan": [51, 255, 255],
    "blue": [51, 51, 255],
    "purple": [153, 51, 153]
}
# from scipy.spatial.transform import Rotation
# from torch.utils.data import Dataset

# from nemo.utils import construct_class_by_name
# from nemo.utils import get_abs_path
# from nemo.utils import load_off

def visual_clevr(cfg, translation, rotation, init_obj_idxs, R, T, save_dir, distances = None, **kwargs):
    device = 'cuda'
    # home_path = '/home/xingrui/projects/superclevr-physics/data/output_1k/'
    # mesh_path = home_path + 'convert/objs_downsample/'
    mesh_path = cfg.dataset.mesh_path
    mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path) if t.endswith('.obj')])))
    mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
    
    # mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

    if 'original_img' not in kwargs:
        img_ = np.zeros((960, 1280, 3), dtype=np.uint8)
    elif isinstance(kwargs['original_img'], torch.Tensor):
        img_ = kwargs['original_img'].squeeze().numpy()
    else:
        img_ = kwargs['original_img']

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

    sub_cates = [mesh_sub_cates[i] for i in init_obj_idxs.squeeze() if i > -1]
    sub_cates_indexs = [mesh_sub_cates.index(i) for i in sub_cates]
    # collected_cads = [mesh_loader.all_vertices[i] for i in sub_cates_indexs]
    objects_transfroms = []
    colors = []

    for i in range(len(sub_cates_indexs)):
        if distances is not None:
            scale_ = distances.squeeze()[i].item()
        else:
            scale_ = 1
        trans_ = torch.Tensor(translation.squeeze()[i])[None].cpu() # @ R__.cpu()
        R_ = torch.Tensor(rotation.squeeze()[i]).cpu() # @ R__.cpu()
        R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)

        objects_transfroms.append(Transform3d(matrix=R_.to(device), device=device).compose(Scale(scale_, device=device)).compose(Translate(trans_.to(device), device=device)))
        colors.append(torch.tensor(COLORMAP[kwargs['colors'][i][0]])/255)
    # rr = Rotation.from_euler('XYZ', anno_['camera']["rotation_euler"], degrees=False).as_matrix()
    # import ipdb; ipdb.set_trace()
    # C = torch.Tensor([7.509376525878906, -6.309525012969971, 5.79176139831543]).to(device)[None]# @ R__ 
    # R = look_at_rotation(C, device=device)
    # T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

    verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].float().to(device)) for k, i in enumerate(sub_cates_indexs)]
    faces_ = [mesh_loader.all_faces[i].to(device) for i in sub_cates_indexs]
    
    textures = TexturesVertex([torch.ones_like(vv)*colors[idx].cuda() for idx, vv in enumerate(verts_)])
    # import ipdb; ipdb.set_trace()
    mesh_ = join_meshes_as_scene(Meshes(verts_, faces_, textures=textures))

    img = phong_renderer(mesh_, R=R, T=T).cpu().detach().numpy()[0]

    # import ipdb; ipdb.set_trace()
    rate_ = 0.8
    img_out = img[..., :3] * img[..., 3:] * rate_ * 255 + img_[..., :3] * (1 - img[..., 3:] * rate_)
    # Image.fromarray(img_out.astype(np.uint8)).save('debug.png')
    # import ipdb; ipdb.set_trace()
    return img_out[:, :, ::-1]

    # Image.fromarray((img[..., 3] * 255).astype(np.uint8)).save(save_dir + 'debug.png')
    
    # import ipdb; ipdb.set_trace()


def visual_extrema(num, boxes, save_dir, **kwargs):

    img_ = kwargs['original_img'].squeeze().numpy()
    h, w, _= img_.shape
    img_ = Image.fromarray(img_.astype(np.uint8))

    step_x, step_y = w // num, h // num

    draw = ImageDraw.Draw(img_)
    for i in range(len(boxes)):

        rectangle_coords = (boxes[i][0] / num * w,
                            boxes[i][1] / num * h,
                            boxes[i][0] / num * w + step_x,
                            boxes[i][1] / num * h + step_y)
        print(rectangle_coords)
        draw.rectangle(rectangle_coords, outline="red", width=5)
    img_out = np.array(img_)
    Image.fromarray(img_out.astype(np.uint8)).save(save_dir + 'box_debug.png')

def visual_feature(feature_map, kpt_):
    '''To visualize the feature map
    '''
    return

def visual_3D(x, y, z, camera, base_image):
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
        return image_coords
    x, y = transform_3d_to_2d(x, y, z, camera)
    img_collision = cv2.circle(base_image, (x, y), 5, (0, 0, 255), -1)
