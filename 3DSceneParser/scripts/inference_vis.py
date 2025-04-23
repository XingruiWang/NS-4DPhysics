import sys
sys.path.append('./')

import logging

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
            # model.evaluate(sample)
            model.visualize(sample)
       
    return

def inference_3d_pose_estimation_clevr(
    cfg,
    cate,
    model,
    dataloader,
    cached_pred=None
):
    save_pred = {}
    save_classification = {}
    loss = []
    distance_loss = []
    running = []
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}")):
        if cached_pred is None or True:
            # preds, classification_result = model.evaluate(sample)
            model.visualize(sample)
            # if classification_result:
            #     for img_name in classification_result.keys():
            #         save_classification[img_name] = classification_result[img_name]
            
            # for pred, name_ in zip(preds, sample['this_name']):
            #     save_pred[str(name_)] = pred
        # for pred in preds:
            # # _err = pose_error(sample, pred["final"][0])
            # if 'pose_error' in pred.keys():
            #     _err = pred['pose_error']
            #     pose_errors.append(_err)
            #     running.append((cate, _err))
            # _loss = pred['final']['loss']
            # _distance_loss = pred['final']['distance_loss']
            # loss.append(_loss)
            # distance_loss.append(_distance_loss)
            # running.append((pred['final']['scene'], _loss, _distance_loss))
            
    # results = {}
    # results["running"] = running
    # results["loss"] = np.mean(_loss)
    # results["distance_loss"] = np.mean(_distance_loss)
    # # results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    # # results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    # # results["med_err"] = np.median(pose_errors) / np.pi * 180.0
    # results["save_pred"] = save_pred
    # results["save_classification"] = save_classification

    # return results
    return


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
    logging.info(f"Scenes:      {cfg.dataset.scene_list}")
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

    logging.info('\n'+cate_line+'\n'+loss+'\n'+distance_loss+'\n')


helper_func_by_task = {"3d_pose_estimation": inference_3d_pose_estimation,
                        "4d_pose_estimation": inference_3d_pose_estimation, 
                        "6d_pose_estimation": inference_3d_pose_estimation,
                        "3d_pose_estimation_clevr": inference_3d_pose_estimation_clevr,
                        "3d_pose_estimation_print": print_3d_pose_estimation, 
                        "4d_pose_estimation_print": print_3d_pose_estimation, 
                        "6d_pose_estimation_print": print_3d_pose_estimation,
                        "3d_pose_estimation_clevr_print": print_3d_pose_estimation_clevr}
