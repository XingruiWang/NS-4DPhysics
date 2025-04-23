"""
Process the generated scene annotation into the same format as the CLEVRER dataset
"""

import os
import json
import numpy as np

import argparse

from tqdm import tqdm

def load_collisions(event_file, frame_len):
    last_crash_frame = {}
    with open(event_file) as f:
        events = json.load(f)
    collisions = events["collisions"]
    filtered_collisions = []
    for c in collisions:
        instances = tuple(sorted(c['instances']))
        # frame = int(c['frame'])+1
        frame = int(c['frame']+0.5)
        if frame >= frame_len:
            continue
        if instances[0] >= 0:
            # if instances not in last_crash_frame:
            #     last_crash_frame[instances] = frame
            #     c['frame'] = frame
            #     filtered_collisions.append(c)
            if instances not in last_crash_frame or \
                frame - last_crash_frame[instances] > 5:
                last_crash_frame[instances] = frame
                c['frame'] = frame
                c['instances'] = sorted(list(instances))
                filtered_collisions.append(c)
    return filtered_collisions


def get_velocity_label(velocities):
    speed_norm = np.linalg.norm(velocities)
    if speed_norm < 0.1:
        return "stationary"
    elif speed_norm < 5.0:
        return "slow"
    elif speed_norm > 5.5:
        return "fast"
    else:
        return "None"

def rotate_vectors(base_vec, target_vec):
    # Extract components of the base vector
    x, y = base_vec[0], base_vec[1]

    # Calculate the angle of rotation needed to align base vector with (1, 0, 0)
    theta = -np.arctan2(y, x)

    # Rotation matrix around the Z-axis
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Rotate the base and target vectors
    base_rotated = R @ np.array(base_vec)
    target_rotated = R @ np.array(target_vec)

    return target_rotated


def get_velocity_direction(velocities, camera):
    # R__ = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 

    c0 = np.array(camera['location'])
    look_at = np.array(camera['look_at'])
    v1 =  c0 - look_at
    v1[2] = 0
    V = np.array(velocities)
    angle = rotate_vectors(v1, V)
    # R = np.array(camera['R'])[:3, :3]
    # T = np.array(camera['location'])
    # V = np.array(velocities)
    # # # Calculate the translated vector

    # v_c = R.T @ (V - T)
    # c_0 = R.T @ (- T)
    # import ipdb; ipdb.set_trace()

    return find_direction(angle)

def normalize(v):
    return v / np.linalg.norm(v)

def find_direction(vector, threshold=0.8):
    directions = {
        'front': np.array([1, 0, 0]),
        'back': np.array([-1, 0, 0]),
        'right': np.array([0, 1, 0]),
        'left': np.array([0, -1, 0]),
        'up': np.array([0, 0, 1]),
        'down': np.array([0, 0, -1])
    }
    
    # Normalize the input vector
    norm_vector = normalize(np.array(vector))
    
    # Calculate dot products and check against the threshold
    for direction, dir_vector in directions.items():
        dot_product = np.dot(norm_vector, dir_vector)
        if dot_product > threshold:
            return direction
    
    return 'None'  # Return undefined if no direction matches the threshold


def load_metadata(meta_file):
    with open(meta_file) as f:
        metadata = json.load(f)
    meta = metadata['metadata']
    camera = metadata["camera"]
    camera.pop("positions")
    camera.pop("quaternions")

    instances = metadata["instances"]

    # 1. Load objects 
    objects = []
    for instance in instances:
        # 'angular_velocities', 'asset_id', 'bboxes_3d', 'color', 'engine_on', 'floated', 'friction', 'id', 'image_positions', 'init_position', 'init_speed', 'mass', 'name', 'positions', 'quaternions', 'restitution', 'size', 'velocities', 'visibility'])

        obj = {
            "id": instance["id"],
            "name": instance["name"],
            "shape": instance["name"].split("_")[1],
            "color": instance["color"],
            "size": instance["size"],
            
            "mass": instance["mass"],
            "friction": instance["friction"],
            "restitution": instance["restitution"],
            
            "engine_on": instance["engine_on"],
            "floated": instance["floated"]
        }
        objects.append(obj)

    # 2. Load motions
    motion = {}
    frame_len = len(instances[0]["positions"])
    for instance in instances:
        for f in range(frame_len):
            if f not in motion:
                motion[f] = {}
                motion[f]['frame'] = f
                motion[f]['objects'] = []
            if "visibility" not in instance:
                visibility = 1000
            else:
                visibility = instance["visibility"][f]

            motion[f]['objects'].append({   
                "id": instance["id"],
                "location": instance["positions"][f],
                "quaternions": instance["quaternions"][f],
                # "_bboxes_3d": instance["bboxes_3d"][f],
                # "_image_positions": instance["image_positions"][f],
                "velocities": instance["velocities"][f],
                "angular_velocities": instance["angular_velocities"][f],

                "visibility": float(visibility),
                # "velocities_label": get_velocity_label(instance["velocities"][f]),
                # "velocities_direction": get_velocity_direction(instance["velocities"][f], camera),
            })
        # print(meta_file)
        # print(instance['asset_id'], get_velocity_direction(instance["velocities"][0], camera))
        # import ipdb; ipdb.set_trace()
    
    motion = list(motion.values())
    motion = sorted(motion, key=lambda x: x['frame'])

    for m in motion:
        visibility = [not o['visibility'] for o in m['objects']]
        if all(visibility):
            frame_len = m['frame']
            break

    motion = [m for m in motion if m['frame'] < frame_len]

    coming_in = {o['id']: -1 for o in objects}
    coming_out = {o['id']: -1 for o in objects}
    
    for m in motion:
        for o in m['objects']:
            if o['visibility'] and coming_in[o['id']] == -1:
                coming_in[o['id']] = m['frame']
            if not o['visibility'] and coming_in[o['id']] != -1 and coming_out[o['id']] == -1:
                coming_out[o['id']] = m['frame']

    if not len(coming_in) == len(objects):
        import ipdb; ipdb.set_trace()

    return objects, motion, coming_in, coming_out, frame_len

def run_scene(root, scene_id):
    event_file = os.path.join(root, scene_id, "events.json")
    meta_file = os.path.join(root, scene_id, "metadata.json")
    # event_file = f"/home/xingrui/physics_questions_generation/data/output/{scene_id}/events.json"
    # meta_file = f"/home/xingrui/physics_questions_generation/data/output/{scene_id}/metadata.json"

    
    objects, motions, coming_in, coming_out, frame_len = load_metadata(meta_file)
    collisions = load_collisions(event_file, frame_len)
    objects = {o['id']: o for o in objects}
    scene = {}
    scene['index'] = int(scene_id.split("_")[-1])
    scene['scene_filename'] = scene_id
    scene['frame_len'] = frame_len
    scene['objects'] = objects
    scene['motions'] = motions
    scene['collisions'] = collisions
    scene['coming_in'] = coming_in
    scene['coming_out'] = coming_out

    assert len(motions) == frame_len
    return scene


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='The root dir of generated video data')
    parser.add_argument('--output_train', help='The output dir of the training scene')
    parser.add_argument('--output_val', help='The output dir of the validation scene')

    parser.add_argument('--train_scene_length', type=int, default=1000)
    parser.add_argument('--val_scene_length', type=int, default=100)
    args = parser.parse_args()
    root = args.data_dir

    # if os.path.exists(args.output_train):
    #     print(f"{args.output_train} already exists, do you want to overwrite it? (y/n)")
    #     user_input = input()
    #     if user_input.lower() != 'y':
    #         print("Abort")
    #         exit(0)

    if args.output_train:

        all_scenes = {}
        all_scenes['info'] = {
            "name": "SuperCLEVR_physics",
            "version": "1.0",
            "date": "2024-04-15",
            "description": "SuperCLEVR_physics dataset",
            "contributor": "Xingrui Wang",
            "split": "train"
        }
        all_scenes['scenes'] = []
        for i in tqdm(range(args.train_scene_length)):
            scene_id = f"super_clever_{i}"
            scene = run_scene(root, scene_id)
            all_scenes['scenes'].append(scene)

        with open(args.output_train, "w") as f:
            json.dump(all_scenes, f)
        print("Done")

    if args.output_val:
        all_scenes = []

        for i in tqdm(range(args.train_scene_length, args.train_scene_length+args.val_scene_length)):
            scene_id = f"super_clever_{i}"
            scene = run_scene(root, scene_id)
            all_scenes.append(scene)
        with open(args.output_val , "w") as f:
            json.dump(all_scenes, f)

