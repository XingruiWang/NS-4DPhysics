import argparse

import torch
from pytorch3d.transforms import quaternion_to_axis_angle
import numpy as np

def quaternion_to_axis_angle22(quaternion):
    # Ensure the quaternion is a numpy array
    quaternion = np.array(quaternion)
    
    # Normalize the quaternion to avoid numerical instability
    quaternion = quaternion / np.linalg.norm(quaternion)

    # Extract components
    a, b, c, d = quaternion

    # Calculate the angle (theta)
    theta = 2 * np.arccos(a)

    # Compute the sin(theta / 2) to avoid division by zero when a is ±1
    sin_theta_over_2 = np.sqrt(1 - a * a)

    # If sin_theta_over_2 is close to zero, then angle is 0 or 360 degrees, axis does not matter
    if sin_theta_over_2 < 1e-8:
        # Any normalized vector is fine when theta is zero (no rotation)
        axis = np.array([1, 0, 0])
    else:
        # Calculate the axis [x, y, z]
        axis = np.array([b, c, d]) / sin_theta_over_2

    return axis, theta



def eval(simulation, gt):
    error_matrix = np.zeros((3, 120, len(gt['instances'])))
    gt['instances'] = sorted(gt['instances'], key=lambda x: x['id'])
    simulation['instances'] = sorted(simulation['instances'], key=lambda x: x['id'])
    for i in range(len(gt['instances'])):
        for frame_id in range( 120):
            for k, key in enumerate(['positions','quaternions', 'velocities']):
                assert gt['instances'][i]['asset_id'] == simulation['instances'][i]['asset_id'] 
                obj_gt = np.array(gt['instances'][i][key][frame_id])
                obj_pred = np.array(simulation['instances'][i][key][frame_id])

                if key == 'quaternions':
                    # import ipdb; ipdb.set_trace()
                    quaternions_pred = torch.tensor([simulation['instances'][i][key][frame_id-1]], dtype=torch.float32)
                    obj_pred = quaternion_to_axis_angle(quaternions_pred)
                    
                    quaternions_gt = torch.tensor([gt['instances'][i][key][frame_id]], dtype=torch.float32)
                    obj_gt = quaternion_to_axis_angle(quaternions_gt)

                # import ipdb; ipdb.set_trace()
                error = ((obj_gt - obj_pred)**2).mean()
                error_matrix[k, frame_id, i] = error
                # import ipdb; ipdb.set_trace()
    return (error_matrix.mean(axis=2))


if __name__ == '__main__':
    import os
    import json
    parser = argparse.ArgumentParser()
    # parser.add_argument('--simulation', default='/home/xingrui/projects/superclevr-physics/physical_symbolic/data/resimulation_final/resimulate_counterfactual_all')

    
    parser.add_argument('--simulation', default='/ccvl/net/ccvl15/xingrui/baseline_time/resimulate_counterfactual_all')

    # parser.add_argument('--gt', default='/home/xingrui/projects/superclevr-physics/data/output_v3_1k')
    parser.add_argument('--gt', default='/ccvl/net/ccvl15/xingrui/output_v3_counterfactual')

    args = parser.parse_args()
    # with open(args.simulation) as f:
    #     simulation = json.load(f)

    # with open(args.gt) as f:
    #     gt = json.load(f)
    all_scene = [f'super_clever_{i}' for i in range(1000,1100)]
    error = []
    from tqdm import tqdm
    for scene in tqdm(all_scene):
        with open(os.path.join(args.simulation, scene, 'metadata.json')) as f:
            simulation = json.load(f)
        with open(os.path.join(args.gt, scene, 'metadata.json')) as f:
            gt = json.load(f)        

        error_ = eval(simulation, gt)
        error.append(error_)
    error = np.array(error)
    mean_error = error.mean(axis=0) 
    for k, key in enumerate(['positions','quaternions', 'velocities']):
        print(key)
        print("\t".join([str(mean_error[k, frame]) for frame in range(0,120,30)]))
        print(mean_error[k, -1])

    