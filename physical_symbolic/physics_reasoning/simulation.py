import argparse
import os

if __name__ == '__main__':
    import json
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt_folder', default='/ccvl/net/ccvl15/xingrui/output_v3_1k')
    # parser.add_argument('--nemo_output', default='/home/xingrui/projects/superclevr-physics/physical_symbolic/data/scene/superclevr_physics_scene_prediction.json')
    
    # parser.add_argument('--output_folder', default='/home/xingrui/projects/superclevr-physics/physical_symbolic/data/resimulate_pred')
    parser.add_argument('--gt_folder', default='/ccvl/net/ccvl15/xingrui/output_v3_counterfactual/')
    parser.add_argument('--nemo_output', default='/home/xingrui/projects/superclevr-physics/physical_symbolic/data/scene/superclevr_physics_scene_prediction.json')
    
    parser.add_argument('--output_folder', default='/home/xingrui/projects/superclevr-physics/physical_symbolic/data/resimulate_counterfactual/')
    
    args = parser.parse_args()

    with open(args.nemo_output, 'r') as f:
        nemo_output = json.load(f)

    for nemo_scene in tqdm(nemo_output):
        scene_name = nemo_scene['scene_name']
        gt_scene_path = os.path.join(args.gt_folder, scene_name, 'metadata.json')

        with open(gt_scene_path, 'r') as f:
            gt_scene_data = json.load(f)   
        for i in range(len(gt_scene_data['instances'])):
            # gt_scene_data['instances'][i]['engine_on'] = nemo_scene['objects'][str(i)]['engine_on']
            # gt_scene_data['instances'][i]['floated'] = nemo_scene['objects'][str(i)]['floated']
            gt_scene_data['instances'][i]['init_position'] = nemo_scene['motions'][0]['objects'][i]['location']
            gt_scene_data['instances'][i]['pred_orientation'] = nemo_scene['motions'][0]['objects'][i]['orientation']
            gt_scene_data['instances'][i]['pred_velocities'] = nemo_scene['motions'][0]['objects'][i]['velocities']
            
            
            # import ipdb; ipdb.set_trace()
        os.makedirs(os.path.join(args.output_folder, scene_name), exist_ok=True)
        with open(os.path.join(args.output_folder, scene_name, 'metadata.json'), 'w') as f:
            json.dump(gt_scene_data, f)
    # for scene in sorted(os.listdir(args.gt_folder)):
    #     if scene.startswith('super'):

    #         import ipdb; ipdb.set_trace()
         
    #         scene_path = os.path.join(args.gt_folder, scene, 'metadata.json')
    #         with open(scene_path, 'r') as f:
    #             scene_data = json.load(f)
        
