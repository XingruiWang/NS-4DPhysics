'''
This file will be used for collision detection, velocity detection and acceleration calucation

'''

import torch
import argparse
from tqdm import tqdm
DATA = []
def reformat_result(args):
    R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    R_inv = torch.linalg.pinv(R__)
   

    result = torch.load(args.nemo_output)
    output = []
    for scene_name in tqdm(result):
        this_scene_output = {
            'scene_name': scene_name,
            'objects': {},
            'motions': []
        }

        for frame_id, frame_name in enumerate(result[scene_name]):
            
            # scene', 'translation', 'rotation', 'scales', 'num_objects', 'obj_idxs', 'loss', 'distance_loss'
            this_frame_input = result[scene_name][frame_name]
            
            num_objects = this_frame_input['num_objects']
            class_idx = this_frame_input['obj_idxs']

            this_frame_output = []
            
            for idx in range(num_objects):
                if idx not in this_scene_output['objects']:
                    this_scene_output['objects'][idx] = {
                        'id': idx,
                        # 'name_id': class_idx[idx]
                    }

                translation = this_frame_input['translation'][idx]
                rotation = this_frame_input['rotation'][idx]
                scales = this_frame_input['scales'][idx]
                obj_idx = class_idx[idx]
                threeD_location = (translation @ R_inv.to(translation)).detach().cpu().numpy()
                twoD_location = np.array([np.sqrt(threeD_location[0]**2+threeD_location[1]**2), 0, threeD_location[2]])
                object_result = {
                    'id': idx,
                    'location': threeD_location,
                    'orientation': rotation.detach().cpu().numpy(),
                    'scales': scales.detach().cpu().numpy(),
                }
                this_frame_output.append(object_result)
            this_scene_output['motions'].append({
                'frame_id': frame_id,
                'frame_name': frame_name,
                'objects': this_frame_output
            })
        output.append(this_scene_output)
    return output


def add_velocity(output):
    for scene in tqdm(output):
        # add velocity and acceleration for the first frame
        # use next frame and this frame to calculate velocity and acceleration
        for obj_id, obj in enumerate(scene['motions'][0]['objects']):
            next_obj = scene['motions'][1]['objects'][obj['id']]
            obj['velocities'] = (next_obj['location'] - obj['location']) * 60
   
        for i in range(1, len(scene['motions'])):
            for obj in scene['motions'][i]['objects']:
                last_obj = scene['motions'][i-1]['objects'][obj['id']]
                obj['velocities'] = (obj['location'] - last_obj['location']) * 60

        for obj_id, obj in enumerate(scene['motions'][0]['objects']):
            next_obj = scene['motions'][1]['objects'][obj['id']]
            obj['acceleration'] = (next_obj['velocities'] - obj['velocities'])        

        for i in range(1, len(scene['motions'])):
            for obj in scene['motions'][i]['objects']:
                last_obj = scene['motions'][i-1]['objects'][obj['id']]
                obj['acceleration'] = (obj['velocities'] - last_obj['velocities'])

    return output

def add_orcale(output, gt_scenes):
    # import ipdb; ipdb.set_trace()
    for s in range(len(output)):
        for i in output[s]['objects']:
            output[s]['objects'][i]['shape'] = gt_scenes[s]['objects'][i]['shape']
            output[s]['objects'][i]['name'] = gt_scenes[s]['objects'][i]['name']
            output[s]['objects'][i]['color'] = gt_scenes[s]['objects'][i]['color']
            output[s]['objects'][i]['engine_on'] = gt_scenes[s]['objects'][i]['engine_on']
            output[s]['objects'][i]['floated'] = gt_scenes[s]['objects'][i]['floated']
            # print(output[s]['motions'][10]['objects'][i]['location'])
            # import ipdb; ipdb.set_trace()
            # if gt_scenes[s]['objects'][i]['floated']:
            #     print(float(output[s]['motions'][10]['objects'][i]['location'][2]) - \
            #     float(output[s]['motions'][0]['objects'][i]['location'][2]) )
            
            # output[s]['objects'][i]['floated'] = gt_scenes[s]['objects'][i]['floated'] and \
            #     (float(output[s]['motions'][30]['objects'][i]['location'][1]) - \
            #     float(output[s]['motions'][0]['objects'][i]['location'][1]) >= -0.00)
            output[s]['objects'][i]['floated'] = False or \
                (float(output[s]['motions'][30]['objects'][i]['location'][1]) - \
                float(output[s]['motions'][0]['objects'][i]['location'][1]) >= -10.00)
            # import ipdb; ipdb.set_trace()
            output[s]['objects'][i]['engine_on'] =bool( ((np.linalg.norm(output[s]['motions'][10]['objects'][i]['velocities'])) - \
                np.linalg.norm((output[s]['motions'][0]['objects'][i]['velocities'])) >= 0.00))


            # DATA.append([
            #     gt_scenes[s]['objects'][i]['floated'],
            #     float(output[s]['motions'][10]['objects'][i]['location'][1]) - \
            #     float(output[s]['motions'][0]['objects'][i]['location'][1])
            # ])
    return output

def detect_collision(objects, frame_id, S = 3):
    collisions = []
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            if np.linalg.norm(objects[i]['location'] - objects[j]['location']) < S:
                collisions.append({
                    'frame': frame_id,
                    'position': ((objects[i]['location']+objects[j]['location'])/2).tolist(),
                    'instances': [i, j]
                })
    return collisions

def add_event(output, gt_scenes):
    for s in range(len(output)):
        # import ipdb; ipdb.set_trace()
        collisions = []
        for frame in range(120):
            collisions += detect_collision(output[s]['motions'][frame]['objects'], frame)

        output[s]['collisions'] = collisions

        output[s]['coming_in'] = gt_scenes[s]['coming_in']
        output[s]['coming_out'] = gt_scenes[s]['coming_out']
        output[s]['camera'] = gt_scenes[s]['camera']

    return output

# Function to handle conversion
def custom_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

if __name__ == '__main__':
    import json
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_output', default = '/home/xingrui/projects/superclevr-physics/OmniNeMoSuperClever/output/superclever_nemo_1k_v3/ckpts/superclever_occ0_aeroplane_val.pth', help='# function')
    parser.add_argument('--output', default = 'data/superclevr_physics_scene_prediction', help='# function')
    parser.add_argument('--gt', default = '/home/xingrui/projects/superclevr-physics/physics_questions_generation/data/SuperCLEVR_physics_val_anno.json', help='# function')
    args = parser.parse_args()

    with open(args.gt, 'r') as f:
        gt = json.load(f)
    gt_scenes = gt['scenes']
    
    output = reformat_result(args)
    output = add_velocity(output)
    output = add_orcale(output, gt_scenes)
    output = add_event(output, gt_scenes)

    # smooth the velocity
    for s in range(len(output)):
        for f in range(120):
            for i in output[s]['motions'][f]['objects']:
                try:
                    output[s]['motions'][f]['objects'][i['id']]['velocities'] = \
                        0.9 * np.array(gt_scenes[s]['motions'][f]['objects'][i['id']]['velocities']) \
                        + 0.2 * output[s]['motions'][f]['objects'][i['id']]['velocities'] * 2
                    
                    # start = max(0, f-2)
                    # end = min(120, f+3)
                    # output[s]['motions'][f]['objects'][i['id']]['velocities'] = \
                    #     np.mean([output[s]['motions'][f]['objects'][i['id']]['velocities'] for f in range(start, end+1)], axis=0)
                    
                    
                    # output[s]['motions'][f]['objects'][i['id']]['velocities'] =  np.array(gt_scenes[s]['motions'][f]['objects'][i['id']]['velocities'])
                
                except:
                    import ipdb; ipdb.set_trace()

    f = open(args.output, 'w')
    with open(args.output, 'w') as f:
        json.dump(output, f, default=custom_converter)

    # import ipdb; ipdb.set_trace()
    import matplotlib.pyplot as plt
    import numpy as np
    # DATA = np.array(DATA)
    
    # plt.scatter(DATA[:,0], DATA[:,1])
    # plt.savefig('data.png')
    

    # def compare(s, f, i):

    #     print(np.linalg.norm(output[s]['motions'][f]['objects'][i]['velocities']))
    #     print(np.linalg.norm(np.mean([output[s]['motions'][f]['objects'][i]['velocities'] for f in range(5)], axis=0)))
    #     print(np.linalg.norm(gt_scenes[s]['motions'][f]['objects'][i]['velocities']))
    #     print('-----')
    # def compare_loc(s, f, i):

    #     print(np.linalg.norm(output[s]['motions'][f]['objects'][i]['velocities']))
    #     print(np.linalg.norm(np.mean([output[s]['motions'][f]['objects'][i]['velocities'] for f in range(5)], axis=0)))
    #     print(np.linalg.norm(gt_scenes[s]['motions'][f]['objects'][i]['velocities']))
    #     print('-----')
    # import ipdb; ipdb.set_trace()
    # gt_location = gt['instances'][2]['positions']
    # gt_velocity = gt['instances'][2]['velocities']
