import os
import pybullet as p
from nemo.utils.visualization import visual_clevr

from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle, quaternion_to_matrix, axis_angle_to_matrix
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

def asset_name(idx):
    home_path = '/home/xingrui/projects/superclevr-physics/data/output/'
    mesh_path = home_path + 'convert/objs_downsample/'
    mesh_sub_cates = sorted(list(set([t.split('.')[0] for t in os.listdir(mesh_path)])))
    return mesh_sub_cates[idx]

class NeMoPhysicsSimulator():    
    def __init__(self, urdf_dir, frame_rate=60, step_size=4):
        self._data = {}
        '''
        _data['super_clever_1000']['rgba_00000.png'].keys()
dict_keys(['scene', 'translation', 'rotation', 'scales', 'num_objects', 'loss', 'distance_loss'])
        
        '''
        self.physicsClient = p.connect(p.DIRECT)
        self.urdf_dir = urdf_dir
        self.frame_rate = frame_rate
        self.step_size = step_size

    def transform_pytorch3d_to_pybullet(self, translation, rotation):
        R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to(translation.device)
        translation = translation @ R__.inverse()
        rotation = matrix_to_quaternion((axis_angle_to_matrix(rotation) @ R__.inverse()).T)
        return translation, rotation
    
    def update_pred(self, pred):
        '''
        This function is only for this demo and shouldn't be used in nemo
        '''
        '''
        
        pred['final']['scene'] = super_clever_1000/rgba_00000.png
        pred['final'].keys() = dict_keys(['scene', 'translation', 'rotation', 'scales', 'num_objects', 'loss', 'distance_loss'])

        '''
        video_id, frame_id = pred['final']['scene'].split('.')[0].split('/')
        frame_id = int(frame_id.split('_')[-1])
        if video_id not in self._data:
            self._data[video_id] = {}
        if frame_id not in self._data[video_id]:
            self._data[video_id][frame_id] = []
        num_objects = int(pred['final']['num_objects'])

        for i in range(num_objects):

            object_id = i
            object_name_idx = pred['final']["obj_idxs"][i]
            name = asset_name(object_name_idx)
            urdf = f"{self.urdf_dir}/{name}.urdf"
            
            size = pred['final']["scales"][i]

            position, quaternion = self.transform_pytorch3d_to_pybullet(pred['final']["translation"][i], pred['final']["rotation"][i])

            self._data[video_id][frame_id].append({
                "object_id": object_id,
                "name": name,
                "urdf": urdf,
                "name_idx": object_name_idx,
                "size": size,
                "position": position,
                "quaternion": quaternion,
                "simulated": False
            })

    def setup_scene(self, scene_name):
        object_data = self._data[scene_name][0]
        # Reset simulation to avoid accumulating objects in the world
        p.resetSimulation()

        p.setGravity(0, 0, -10)
        # p.setGravity(0, 0, -0)

        self.objects_info = {}
        # Load objects for the current frame
        object_ids = []
        object_name_idx = []
        object_scales = []
        for obj in object_data:
            # obj_id = p.loadURDF(obj["urdf"], basePosition=obj["position"], baseOrientation=obj["quaternion"])
            obj_id = p.loadURDF(obj["urdf"])
            p.changeDynamics(obj_id, -1, contactProcessingThreshold=0.01)

            object_ids.append(obj_id)
            object_name_idx.append(asset_id(obj["name"]))
            object_scales.append(obj["size"])
            
        self.objects_info["object_ids"] = object_ids
        self.objects_info["object_name_idx"] = object_name_idx
        self.objects_info["object_scales"] = object_scales

        

        # Add the floor to the simulation environment
        plane_collision = p.createCollisionShape(p.GEOM_PLANE)
        self.floor_id = p.createMultiBody(
            baseMass=0,                # Zero mass makes it static
            baseCollisionShapeIndex=plane_collision,
            basePosition=[0, 0, 0]     # Position the plane at origin
        )

    def calculate_velocity(self, scene_name, frame_id):
        M = 5
        object_data = self._data[scene_name]
        # import ipdb; ipdb.set_trace()
        for obj, obj_id in zip(object_data[frame_id], self.objects_info["object_ids"]):
            if frame_id == 0:
                obj["velocity"] = torch.zeros_like(obj["position"])
            elif frame_id <= M:
                obj["velocity"] = (obj["position"] - object_data[0][obj_id]["position"]) * self.frame_rate / frame_id
            else:
                obj["velocity"] = (obj["position"] - object_data[frame_id-M][obj_id]["position"]) * self.frame_rate / M
            
                # obj["velocity"] = (obj["position"] - object_data[frame_id-1][obj_id]["position"]) * self.frame_rate / self.step_size

    
    def predict_next_frame(self, scene_name, frame_id, repeat = 1):
        object_data = self._data[scene_name]
        next_frame_prediction, collisions = self.run_next_frame(object_data, frame_id, repeat)
        return next_frame_prediction, collisions


    def run_all_frames(self, data):
        pred_frames = {}
        for frame_id in tqdm(sorted(data.keys())[1:]):  # Exc
            pred_frames[frame_id] = self.run_next_frame(data, frame_id)
        return pred_frames
    

    def run_next_frame(self, data, frame_id=0, repeat=1):
        # the velocity will be replaced by the predicted velocity
        R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 
        pred_frame = {}
        
        if frame_id <= 0:
            raise ValueError("frame_id should be greater than 0")
        
        for obj, obj_id in zip(data[frame_id-1], self.objects_info["object_ids"]):
            p.resetBasePositionAndOrientation(obj_id, obj["position"],obj["quaternion"])
            
            # while self.check_foreground_overlap(obj_id):
            #     obj["position"] += torch.Tensor([0, 0, 0.01]).to(obj["position"].device)
            #     p.resetBasePositionAndOrientation(obj_id, obj["position"],obj["quaternion"])
            
            p.resetBaseVelocity(obj_id, linearVelocity=obj["velocity"])
       

        for _ in range(self.step_size * repeat):
            p.stepSimulation()

        # Get predicted positions
        predicted_positions = []
        predicted_orientations = []

        for obj_id in self.objects_info["object_ids"]:
            pos, orientation = p.getBasePositionAndOrientation(obj_id)
            pos = torch.Tensor(pos) @ R__

            R_ = quaternion_to_matrix(torch.tensor(orientation)).T @ R__

            predicted_positions.append(pos)
            predicted_orientations.append(matrix_to_axis_angle(R_))

        pred_frame["translations"] = predicted_positions
        pred_frame["rotations"] = predicted_orientations
        pred_frame["sizes"] = self.objects_info["object_scales"]
        pred_frame["object_name_idx"] = self.objects_info["object_name_idx"]

        # pred collison
        all_contacts = []
        # for r in range(self.step_size):
        #     contacts = p.getContactPoints()

        #     if len(contacts) > 0:
        #         all_contacts.extend(self.process_collisions(contacts, frame_id + min(10, int(r / self.step_size))))
        #     p.stepSimulation()
        return pred_frame, all_contacts
    
    def check_foreground_overlap(self, obj_idx):

        body_ids = [
            p.getBodyUniqueId(i)
            for i in range(p.getNumBodies()) if i != self.floor_id
        ]

        for body_id in body_ids:
            if body_id == obj_idx:
                continue
            overlap_points = p.getClosestPoints(
                obj_idx, body_id, distance=0.1)
            if overlap_points:
                return True
        return False
    def check_background_overlap(self, obj_idx):

        body_ids = [
            p.getBodyUniqueId(i)
            for i in range(p.getNumBodies()) if i == self.floor_id
        ]

        for body_id in body_ids:
            if body_id == obj_idx:
                continue
            overlap_points = p.getClosestPoints(
                obj_idx, body_id, distance=0.0)
            if overlap_points:
                return True
        return False
    def process_collisions(self, contact_points, frame_id):
        # import ipdb; ipdb.set_trace()

        # print(p.getContactPoints())
        # return
        # contact_points = self.physicsClient.getContactPoints()
        collisions = []
        for collision in contact_points:
            (contact_flag,
            body_a, body_b,
            link_a, link_b,
            position_a, position_b, contact_normal_b,
            contact_distance, normal_force,
            lateral_friction1, lateral_friction_dir1,
            lateral_friction2, lateral_friction_dir2) = collision
            del link_a, link_b  # < unused
            del contact_flag, contact_distance, position_a  # < unused
            del lateral_friction1, lateral_friction2  # < unused
            del lateral_friction_dir1, lateral_friction_dir2  # < unused
            if normal_force > 1e-6:
                obj_A, obj_B = min(body_a, body_b), max(body_a, body_b)
                if obj_B < self.floor_id:
                    collisions.append({
                        "instances": (obj_A, obj_B),
                        "position": position_b,
                        "contact_normal": contact_normal_b,
                        "frame": frame_id,
                        "force": normal_force
                    })
                # else:
                #     collisions.append({
                #         "instances": (-1, obj_A),
                #         "position": position_b,
                #         "contact_normal": contact_normal_b,
                #         "frame": frame_id,
                #         "force": normal_force
                #     })
        # collision_obj_1, collision_obj_2 = min(body_a, body_b), max(body_a, body_b)
        return collisions
 
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

    super_clevr_instance = 'super_clever_1003'
    urdf_dir = "/home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/data/urdf"
    root_dir = "/home/xingrui/projects/superclevr-physics/data/output_v3_1k"


    phy = NeMoPhysicsSimulator(urdf_dir, root_dir, super_clevr_instance)
    data = phy.process_gt_anno()
    phy.set_scenes(data[0])

    phy.pred_frames = phy.run_all_frames(data)
    img = phy.visualize()
