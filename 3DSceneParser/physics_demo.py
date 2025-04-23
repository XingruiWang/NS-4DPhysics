import pybullet as p
from nemo.utils.visualization import visual_clevr
import os
from pytorch3d.transforms import Transform3d, Translate, Rotate, Scale, euler_angles_to_matrix, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_quaternion
import torch
from pytorch3d.renderer import look_at_rotation
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def asset_id(name):
    home_path = '/home/xingrui/projects/superclevr-physics/data/output/'
    mesh_path = home_path + 'convert/objs_downsample/'
    mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path)])))
    return mesh_sub_cates.index(name)

class physics_module():
    def __init__(self, urdf_dir, root_dir, super_clevr_instance):
        self.physicsClient = p.connect(p.DIRECT)

        self.urdf_dir = urdf_dir
        self.anno_dir = os.path.join(root_dir, super_clevr_instance, 'metadata.json')
        self.image_dir = os.path.join(root_dir, super_clevr_instance)
    
    def run_all_frames(self, data):
        pred_frames = {}
        for frame_id in tqdm(sorted(data.keys())[1:]):  # Exc
            pred_frames[frame_id] = self.run_next_frame(data, frame_id)
        return pred_frames
    
    def set_scenes(self, object_data):
        # Reset simulation to avoid accumulating objects in the world
        p.resetSimulation()

        p.setGravity(0, 0, -10)

        self.objects_info = {}
        # Load objects for the current frame
        object_ids = []
        object_name_idx = []
        object_scales = []
        for obj in object_data:
            obj_id = p.loadURDF(obj["urdf"], basePosition=obj["position"], baseOrientation=obj["quaternion"])
            object_ids.append(obj_id)
            object_name_idx.append(asset_id(obj["name"]))
            object_scales.append(obj["size"])
            
        self.objects_info["object_ids"] = object_ids
        self.objects_info["object_name_idx"] = object_name_idx
        self.objects_info["object_scales"] = object_scales

    def run_next_frame(self, data, frame_id=0):
        # the velocity will be replaced by the predicted velocity
        R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 
        pred_frame = {}
        
        if frame_id <= 0:
            raise ValueError("frame_id should be greater than 0")
        
        for obj, obj_id in zip(data[frame_id-1], self.objects_info["object_ids"]):
            # obj_id = p.loadURDF(obj["urdf"], basePosition=obj["position"], baseOrientation=obj["quaternion"])
            p.resetBasePositionAndOrientation(obj_id, obj["position"],obj["quaternion"])
            p.resetBaseVelocity(obj_id, linearVelocity=obj["velocity"])
       
        for _ in range(10):
            p.stepSimulation()

        # Get predicted positions
        predicted_positions = []
        predicted_orientations = []

        for obj_id in self.objects_info["object_ids"]:
            pos, orientation = p.getBasePositionAndOrientation(obj_id)
            pos = torch.Tensor(pos) @ R__

            R_ = quaternion_to_matrix(torch.tensor(orientation)).T @ R__

            predicted_positions.append(pos)
            predicted_orientations.append(R_)

        pred_frame["translations"] = predicted_positions
        pred_frame["rotations"] = predicted_orientations
        pred_frame["sizes"] = self.objects_info["object_scales"]
        pred_frame["object_name_idx"] = self.objects_info["object_name_idx"]

        return pred_frame

    def process_gt_anno(self):
        '''
        This function is only for this demo and shouldn't be used in nemo
        '''

        with open(self.anno_dir, "r") as f:
            import json
            self.anno = json.load(f)
            instances = self.anno["instances"]

        data = {}
        for instance in instances:
            name = instance["name"]
            urdf = f"{self.urdf_dir}/{name}.urdf"
            frame_len = len(instance["positions"])
            for f_id in range(frame_len):
                position = instance["positions"][f_id]
                quaternion = instance["quaternions"][f_id]
                velocity = instance["velocities"][f_id]
                size = instance["size"]

                if f_id not in data:
                    data[f_id] = []

                data[f_id].append({
                    "name": name,
                    "urdf": urdf,
                    "position": position,
                    "quaternion": quaternion,
                    "velocity": velocity,
                    "size": size
                })
        return data
    
    def visualize(self):
        '''
        This function is only for this demo and shouldn't be used in nemo

        '''
        images = []
        for frame_id in tqdm(self.pred_frames):
            translation= torch.stack(self.pred_frames[frame_id]['translations'])
            rotation= torch.stack(self.pred_frames[frame_id]['rotations'])
            init_obj_idxs = torch.tensor(self.pred_frames[frame_id]["object_name_idx"])

            R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 

            C = torch.Tensor(self.anno['camera']['positions'][frame_id])[None] @ R__ 
            p1, p2, p3 = self.anno['camera']['look_at']
            R = look_at_rotation(C, at=([p1, p3, -p2],))
            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
            R = R.cuda()
            T = T.cuda()

            distance = torch.tensor(self.objects_info["object_scales"])

            ori_image = cv2.imread(os.path.join(self.image_dir, f"rgba_{frame_id:05d}.png"))
            img = visual_clevr(translation, rotation, init_obj_idxs, R, T, 'save_dir', distances = distance, original_img = ori_image)
            images.append(img)
            cv2.imwrite(f"vis/physics_output/phy_{frame_id:05d}.png", img[:, :, ::-1])
        return images
            
if __name__ == "__main__":
    # home_path = '/home/xingrui/projects/superclevr-physics/data/output/'
    # mesh_path = home_path + 'convert/objs_downsample/'
    # mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path)])))

    super_clevr_instance = 'super_clever_20'
    urdf_dir = "/home/xingrui/projects/superclevr-physics/data/output/convert/urdf"
    root_dir = "/home/xingrui/projects/superclevr-physics/data/output/"


    phy = physics_module(urdf_dir, root_dir, super_clevr_instance)
    data = phy.process_gt_anno()
    phy.set_scenes(data[0])

    phy.pred_frames = phy.run_all_frames(data)
    img = phy.visualize()
