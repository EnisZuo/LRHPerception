import os
import numpy as np
import torch

def print_info(epoch, traj_pred, optimizer, loss_dict, logger):
    loss_dict['kld_weight'] = traj_pred.param_scheduler.kld_weight.item()
    loss_dict['z_logit_clip'] = traj_pred.param_scheduler.z_logit_clip.item()

    info = "Epoch:{},\t lr:{:6},\t loss_goal:{:.4f},\t loss_traj:{:.4f},\t loss_kld:{:.4f},\t \
            kld_w:{:.4f},\t z_clip:{:.4f} ".format( 
            epoch, optimizer.param_groups[0]['lr'], loss_dict['loss_goal'], loss_dict['loss_traj'], 
            loss_dict['loss_kld'], loss_dict['kld_weight'], loss_dict['z_logit_clip']) 
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)

def viz_results(viz, 
                X_global, 
                y_global, 
                pred_traj, 
                img_path, 
                bbox_type='cxcywh',
                normalized=True,
                logger=None, 
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    id_to_show = np.random.randint(pred_traj.shape[0])

    # 1.1 initialize visualizer
    viz.initialize(img_path[id_to_show])

    # NOTE 1.2 Find the best trajectory per MSE
    obj_pred_traj = pred_traj[id_to_show]
    gt_expanded = np.repeat(y_global[id_to_show][:, np.newaxis, :], obj_pred_traj.shape[1], axis=1)
    mse = np.mean((gt_expanded - obj_pred_traj)**2, axis=(0, 2))
    best_pred_index = np.argmin(mse)
    best_pred = obj_pred_traj[:, best_pred_index, :]

    # 2. visualize point trajectory or box trajectory
    if y_global.shape[-1] == 2:
        viz.visualize(best_pred, color=(0, 1, 0), label='pred future', viz_type='point')
        viz.visualize(X_global[id_to_show], color=(0, 0, 1), label='past', viz_type='point')
        viz.visualize(y_global[id_to_show], color=(1, 0, 0), label='gt future', viz_type='point')
    elif y_global.shape[-1] == 4:
        T = X_global.shape[1]
        viz.visualize(best_pred, color=(0, 255., 0), label='pred future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])
        viz.visualize(X_global[id_to_show], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(y_global[id_to_show], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])        
    
    # 4. get image. 
    if y_global.shape[-1] == 2:
        viz_img = viz.plot_to_image(clear=True)
    else:
        viz_img = viz.img

    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)

def post_process(cfg, X_global, y_global, pred_traj=None, pred_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else:
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
    if pred_goal is not None:
        pred_goal = pred_goal.detach().to('cpu').numpy()
    if pred_traj is not None:
        pred_traj = pred_traj.detach().to('cpu').numpy()
    
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.DATASET.MIN_BBOX)[None, None, :] # B, T, dim
        _max = np.array(cfg.DATASET.MAX_BBOX)[None, None, :]
        # print(_min, _max)
        if cfg.DATASET.NORMALIZE:
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            if pred_traj is not None:
                pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        else:
            raise ValueError()
    return X_global, y_global, pred_goal, pred_traj