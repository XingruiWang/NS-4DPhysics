import sys
sys.path.append('./')

import logging
import time
import numpy as np
from tqdm import tqdm

from nemo.utils import pose_error


def inference_3d_pose_estimation(
    cfg,
    cate,
    model,
    dataloader,
    cached_pred=None
):
    save_pred = {}
    save_classification = {}
    pose_errors = []
    running = []
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cate}")):
        if cached_pred is None or True:
            #
            preds, classification_result = model.evaluate(sample)
            if classification_result:
                for img_name in classification_result.keys():
                    save_classification[img_name] = classification_result[img_name]
            
            for pred, name_ in zip(preds, sample['this_name']):
                save_pred[str(name_)] = pred
        else:
            for name_ in sample['this_name']:
                save_pred[str(name_)] = cached_pred[str(name_)]
        for pred in preds:
            # _err = pose_error(sample, pred["final"][0])
            if 'pose_error' in pred.keys():
                _err = pred['pose_error']
                pose_errors.append(_err)
                running.append((cate, _err))
    pose_errors = np.array(pose_errors)

    results = {}
    results["running"] = running
    results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    results["med_err"] = np.median(pose_errors) / np.pi * 180.0
    results["save_pred"] = save_pred
    results["save_classification"] = save_classification

    return results


def inference_3d_pose_estimation_clevr_physics(
    cfg,
    cate,
    model,
    physics_simulator,
    dataloader,
    cached_pred=None
):
    save_pred = {}
    save_classification = {}
    pose_errors = []
    loss = []
    distance_loss = []
    running = []
    with tqdm(dataloader, desc=f"{cfg.task}") as tqdm_iterator:
        for i, sample in enumerate(tqdm_iterator):
            # print(f"sample: {sample['this_name']}")
            tqdm_iterator.set_description(f"{cfg.task} | Sample: {sample['this_name'][0]}")

            if cached_pred is None or True:
                # add physics simulation
                physics_simulator.update_data(save_pred)   
                preds, classification_result = model.evaluate(sample, prev_pred = save_pred, repeat = 1)
                if classification_result:
                    for img_name in classification_result.keys():
                        save_classification[img_name] = classification_result[img_name]
                
                # import ipdb; ipdb.set_trace()
                for pred, name_ in zip(preds, sample['this_name']):
                    # import ipdb; ipdb.set_trace()import ipdb; ipdb.set_trace()
                    assert pred['final']  ['scene'] == name_
                    video_name, frame_name = name_.split('/')
                    if video_name not in save_pred:
                        save_pred[video_name] = {}
                    save_pred[video_name][frame_name] = pred['final']
                        # save_pred[str(name_)] = pred
            else:
                for name_ in sample['this_name']:
                    save_pred[str(name_)] = cached_pred[str(name_)]
            for pred in preds:
                # # _err = pose_error(sample, pred["final"][0])
                if 'pose_errors' in pred.keys():
                    _err = pred['pose_errors']
                    pose_errors.extend(_err)
                #     running.append((cate, _err))
                _loss = pred['final']['loss']
                # _distance_loss = pred['final']['distance_loss']
                _distance_loss = 0.0
                loss.append(_loss)
                distance_loss.append(_distance_loss)
            
                running.append((pred['final']['scene'], _loss, _distance_loss))
    
    results = {}
    results["running"] = running
    results["loss"] = np.mean(_loss)
    results["distance_loss"] = np.mean(_distance_loss)
    results["pi6_acc"] = np.mean(np.array(pose_errors) < np.pi / 6)
    results["pi18_acc"] = np.mean(np.array(pose_errors) < np.pi / 18)
    results["med_err"] = np.median(np.array(pose_errors)) / np.pi * 180.0
    results["save_pred"] = save_pred
    results["save_classification"] = save_classification
    logging.info(f"pi6_acc: {results['pi6_acc']}, pi18_acc: {results['pi18_acc']}, med_err: {results['med_err']}")

    return results


def inference_3d_pose_estimation_clevr(
    cfg,
    cate,
    model,
    dataloader,
    cached_pred=None
):
    save_pred = {}
    save_classification = {}
    pose_errors = []
    position_errors = []
    loss = []
    distance_loss = []
    running = []
    start_time, end_time = 0, 0
    iter_num = 0
    time_lst = []
    with tqdm(dataloader, desc=f"{cfg.task}") as tqdm_iterator:
        
        for i, sample in enumerate(tqdm_iterator):
            

            # print(f"sample: {sample['this_name']}")
            # tqdm_iterator.set_description(f"{cfg.task} | Sample: {sample['this_name'][0]}")

            # add physics simulation
            # import ipdb; ipdb.set_trace()
            if cached_pred is None or True:
                start_time = time.time()
                (preds, physics_prediction, collisions), classification_result = model.evaluate(sample, prev_pred = save_pred)
                end_time = time.time()
                
                if classification_result:
                    for img_name in classification_result.keys():
                        save_classification[img_name] = classification_result[img_name]
                
                # import ipdb; ipdb.set_trace()
                for pred, name_ in zip(preds, sample['this_name']):
                    # import ipdb; ipdb.set_trace()import ipdb; ipdb.set_trace()
                    assert pred['final']['scene'] == name_
                    video_name, frame_name = name_.split('/')
                    frame_id = int(frame_name.split('_')[-1].split('.')[0])
                    next_frame_name = f"rgba_{frame_id+1:05d}.png"
                    if video_name not in save_pred:
                        save_pred[video_name] = {}

                    save_pred[video_name][frame_name] = pred['final']
                    save_pred[video_name][next_frame_name] = physics_prediction
                    save_pred[video_name][next_frame_name]['scene'] = f'{video_name}/{next_frame_name}.png',  # Assuming fixed scene path

                    save_pred[video_name][frame_name]['collisions'] = collisions
                    save_pred[video_name][next_frame_name]['collisions'] = collisions
            else:
                for name_ in sample['this_name']:
                    
                    video_name, frame_name = name_.split('/')
                    # import ipdb; ipdb.set_trace()
                    save_pred[video_name] = cached_pred[video_name]    
            for pred in preds:
                # # _err = pose_error(sample, pred["final"][0])
                if 'pose_errors' in pred.keys():
                    _err = pred['pose_errors']
                    pose_errors.extend(_err)
                if 'position_errors' in pred.keys():
                    _err = pred['position_errors']
                    position_errors.extend(_err)
                #     running.append((cate, _err))
                _loss = pred['final']['loss']
                # _distance_loss = pred['final']['distance_loss']
                _distance_loss = 0.0
                loss.append(_loss)
                distance_loss.append(_distance_loss)
            
                running.append((pred['final']['scene'], _loss, _distance_loss))
            
            pi18_acc = np.mean(np.array(pose_errors) < np.pi / 18)
            position_mse = np.mean(np.array(position_errors))
            
            time_lst.append(end_time - start_time)
            if i > 0:
                tqdm_iterator.set_description(f"Sample: {sample['this_name'][0]} | Pi18_acc: {pi18_acc:.3f} | Loc MSE: {position_mse:.4f} | Time: {np.mean(time_lst[1:])*1000:.2f}")

    results = {}
    results["running"] = running
    results["loss"] = np.mean(_loss)
    results["distance_loss"] = np.mean(_distance_loss)
    results["pi6_acc"] = np.mean(np.array(pose_errors) < np.pi / 6)
    results["pi18_acc"] = np.mean(np.array(pose_errors) < np.pi / 18)
    results["location_mse"] = np.mean(np.array(position_errors))
    results["med_err"] = np.median(np.array(pose_errors)) / np.pi * 180.0
    results["save_pred"] = save_pred
    results["save_classification"] = save_classification
    logging.info(f"pi6_acc: {results['pi6_acc']}, pi18_acc: {results['pi18_acc']}, med_err: {results['med_err']}, location_mse: {results['location_mse']}")

    return results



def inference_3d_pose_estimation_video_clevr(
    cfg,
    cate,
    model,
    dataloader,
    cached_pred=None
):
    '''
    save_pred = {
        "video_name": {
            "frame_name": {
                "pose": np.array([x, y, z, qw, qx, qy, qz]),
                "score": float
            }
        }
    }
    
    '''
    save_pred = {}
    save_classification = {}
    loss = []
    distance_loss = []
    running = []
    start_time, end_time = 0, 0
    with tqdm(dataloader, desc=f"{cfg.task}") as tqdm_iterator:
        
        for i, sample in enumerate(tqdm_iterator):
            # print(f"sample: {sample['this_name']}")
            tqdm_iterator.set_description(f"{cfg.task} | sample: {sample['this_name']} | time: {end_time - start_time}")
            
            start_time = time.time()
            preds, classification_result = model.evaluate(sample, None)

            # skip
            if classification_result:
                for img_name in classification_result.keys():
                    save_classification[img_name] = classification_result[img_name]
            
            for pred, name_ in zip(preds, sample['this_name']):
                video_id, frame_id = name_.split('_')
                if video_id not in save_pred:
                    save_pred[video_id] = {}
                save_pred[video_id][frame_id] = pred
                # save_pred[str(name_)] = pred

            # TODO: log the loss, accuracy,m but just copy from the old code, not sure if it's correct
            for pred in preds:
                # # _err = pose_error(sample, pred["final"][0])
                # if 'pose_error' in pred.keys():
                #     _err = pred['pose_error']
                #     pose_errors.append(_err)
                #     running.append((cate, _err))
                _loss = pred['final']['loss']
                _distance_loss = pred['final']['distance_loss']
                loss.append(_loss)
                distance_loss.append(_distance_loss)
                running.append((pred['final']['scene'], _loss, _distance_loss))
            end_time = time.time()

    results = {}
    results["running"] = running
    results["loss"] = np.mean(_loss)
    results["distance_loss"] = np.mean(_distance_loss)
    # results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    # results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    # results["med_err"] = np.median(pose_errors) / np.pi * 180.0
    results["save_pred"] = save_pred
    results["save_classification"] = save_classification

    return results



def print_3d_pose_estimation(
    cfg,
    all_categories,
    running_results
):
    logging.info(f"\n3D Pose Estimation Results:")
    logging.info(f"Dataset:     {cfg.dataset.name} (root={cfg.dataset.root_path})")
    logging.info(f"Category:    {all_categories}")
    logging.info(f"# samples:   {len(running_results)}")
    logging.info(f"Model:       {cfg.model.name} (ckpt={cfg.args.checkpoint})")

    cate_line = f'            '
    pi_6_acc = f'pi/6 acc:   '
    pi_18_acc = f'pi/18 acc:  '
    med_err = f'Median err: '
    for cate in all_categories:
        pose_errors_cate = np.array([x[1] for x in running_results if x[0] == cate])
        cate_line += f'{cate[:6]:>8s}'
        pi_6_acc += f'  {np.mean(pose_errors_cate < np.pi / 6)*100:5.1f}%'
        pi_18_acc += f'  {np.mean(pose_errors_cate < np.pi / 18)*100:5.1f}%'
        med_err += f'  {np.median(pose_errors_cate)/np.pi*180.0:6.2f}'
    cate_line += f'    mean'
    pose_errors_cate = np.array([x[1] for x in running_results])
    pi_6_acc += f'  {np.mean(pose_errors_cate < np.pi / 6)*100:5.1f}%'
    pi_18_acc += f'  {np.mean(pose_errors_cate < np.pi / 18)*100:5.1f}%'
    med_err += f'  {np.median(pose_errors_cate)/np.pi*180.0:6.2f}'

    logging.info('\n'+cate_line+'\n'+pi_6_acc+'\n'+pi_18_acc+'\n'+med_err)


def print_3d_pose_estimation_clevr(
    cfg,
    all_categories,
    running_results
):
    logging.info(f"\n3D Pose Estimation Results:")
    logging.info(f"Dataset:     {cfg.dataset.name} (root={cfg.dataset.root_path})")
    logging.info(f"Category:    all_categories")
    # logging.info(f"Scenes:      {cfg.dataset.scene_list}")
    logging.info(f"# samples:   {len(running_results)}")
    logging.info(f"Model:       {cfg.model.name} (ckpt={cfg.args.checkpoint})")
    # import ipdb; ipdb.set_trace()
    cate_line = f'            '
    loss = f'loss :      '
    distance_loss = f'dis_loss:   '
    for scene in running_results:
        scene_name, _loss, _distance_loss = scene
        parts = scene_name.split('_')
        cate_line += f'{"_".join(parts[2:3]):>8s}'
        loss += f'   {_loss:5.3f}'
        distance_loss += f'   {_distance_loss:5.3f}'
    cate_line += f'    mean'
    loss_mean = np.mean([running_result[1] for running_result in running_results])
    distance_loss_mean = np.mean([running_result[2] for running_result in running_results])
    loss += f'   {loss_mean:5.3f}'
    distance_loss += f'   {distance_loss_mean:5.3f}'



helper_func_by_task = {"3d_pose_estimation": inference_3d_pose_estimation,
                        "4d_pose_estimation": inference_3d_pose_estimation, 
                        "6d_pose_estimation": inference_3d_pose_estimation,
                        "3d_pose_estimation_clevr": inference_3d_pose_estimation_clevr,
                        "3d_pose_estimation_print": print_3d_pose_estimation, 
                        "4d_pose_estimation_print": print_3d_pose_estimation, 
                        "6d_pose_estimation_print": print_3d_pose_estimation,
                        "3d_pose_estimation_clevr_print": print_3d_pose_estimation_clevr}
