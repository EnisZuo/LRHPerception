import torch, os, cv2, sys, time
from tqdm import tqdm
import torch.distributed as dist
import numpy as np

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def online_eval(cfg, model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    num_images = 0
    total_time = 0
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue
            
            num_images = num_images + 1
            start_time = time.time()
            pred_depth = model(image, specific = 'depth')
            end_time = time.time()
            total_time += (end_time - start_time)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
        # print(pred_depth.shape, gt_depth.shape)
        
        height, width = gt_depth.shape
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
        pred_depth = pred_depth_uncropped

        min_depth_eval = 1e-3
        max_depth_eval = 80
        pred_depth[pred_depth < min_depth_eval] = min_depth_eval
        pred_depth[pred_depth > max_depth_eval] = max_depth_eval
        pred_depth[np.isinf(pred_depth)] = max_depth_eval
        pred_depth[np.isnan(pred_depth)] = min_depth_eval

        valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)

        # if args.garg_crop or args.eigen_crop:
        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)
        # if args.garg_crop:
        eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            # elif args.eigen_crop:
            #     if args.dataset == 'kitti':
            #         eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            #     elif args.dataset == 'nyu':
            #         eval_mask[45:471, 41:601] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1
        
    fps = num_images / total_time
    print(f"Model speed: {fps:.2f} FPS")

    if cfg.MULTIPROCESSING_DISTRIBUTED:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not cfg.MULTIPROCESSING_DISTRIBUTED or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None

def calculate_mIOU(segmentation, gt_masks):
    # Binarize predictions for single-class segmentation
    preds_binary = (segmentation > 0.5).float()
    gt = gt_masks.unsqueeze(1)
    assert preds_binary.shape == gt.shape, "Predictions and ground truth tensors must have the same shape"

    intersection = (preds_binary * gt).sum(dim=(2, 3))
    union = (preds_binary + gt).clamp(0, 1).sum(dim=(2, 3))

    # Calculate IoU for each image in the batch
    iou = intersection / (union + 1e-6)

    # Calculate mean IoU across the batch
    return iou.mean()

def eval_seg(model, dataloader, cfgs):
    model.eval()
    if cfgs.EVAL.SPEED_TEST:
        dummy_data = torch.randn(1, 3, 512, 1024).cuda(cfgs.GPU)
        with torch.no_grad():
            print("\033[31m" + "== model warming up for speed test ==" + "\033[0m")
            for _ in range(200):
                _ = model(dummy_data, None)
            print("\033[32m" + "== model running for speed test ==" + "\033[0m")
            start_time = time.time()
            for _ in range(500):
                _ = model(dummy_data, None)
            end_time = time.time()
            print(f'Speed for processing one data {(end_time - start_time) / 500}s')
    with torch.no_grad():
        total_mIOU = 0
        for i, data in enumerate(tqdm(dataloader)):
            det_imgs, det_labels, seg_labels = data[0], data[1], data[2]

            det_imgs = det_imgs.cuda(cfgs.GPU)
            det_labels = det_labels.cuda(cfgs.GPU)
            seg_labels = seg_labels.cuda(cfgs.GPU)
            _, seg_out = model(det_imgs, det_labels=det_labels)
            mIOU_batch = calculate_mIOU(seg_out, seg_labels)
            total_mIOU += mIOU_batch
            # print(mIOU_batch)
        print('mIOU: {:.4f}'.format(total_mIOU / len(dataloader)))
    model.train()

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def init_weights(model, cfg):
    if cfg.ORI_CKPT != '':
        try:
            ori_dict = torch.load(cfg.ORI_CKPT, map_location='cpu')
            model.load_state_dict(ori_dict['model_state_dict'], strict=False)
            print("== Loaded ORI checkpoint '{}'".format(cfg.ORI_CKPT))
            return
        except:
            try:
                ori_dict = torch.load(cfg.ORI_CKPT, map_location='cpu')
                filtered_dict = {'.'.join(k.split('.')[1:]): v for k, v in ori_dict.items()}
                model.load_state_dict(filtered_dict, strict=True)
                print("== Loaded ORI checkpoint '{}'".format(cfg.ORI_CKPT), "go ahead load traj prediction weights.")
            except:
                print("== No ORI checkpoint found, Load individual checkpoint.")    
                
    if cfg.TRAJ_PRED_CKPT!= '':
        if os.path.isfile(cfg.TRAJ_PRED_CKPT):
            print("== Loading TRAJ_PRED checkpoint '{}'".format(cfg.TRAJ_PRED_CKPT))
            pretrained_dict = torch.load(cfg.TRAJ_PRED_CKPT, map_location='cpu')
            try:
                model.traj_predictor.load_state_dict(pretrained_dict)
                print("== Loaded traj_pred checkpoint '{}'".format(cfg.TRAJ_PRED_CKPT))
                return
            except:
                try:
                    current_dict = model.traj_predictor.state_dict()
                    updated_dict = {k: v for k, v in pretrained_dict.items() if k in current_dict}
                    assert updated_dict.keys() == current_dict.keys(), f"{pretrained_dict.keys()}, {current_dict.keys()}"
                    model.traj_predictor.load_state_dict(updated_dict)
                    print("== Loaded traj_pred checkpoint '{}'".format(cfg.TRAJ_PRED_CKPT))
                    return
                except Exception as e:
                    print(e)
                    print("== Failed to load traj_pred checkpoint '{}'".format(cfg.TRAJ_PRED_CKPT))

        else:
            print("== No depth checkpoint found at '{}'".format(cfg.TRAJ_PRED_CKPT))

    # if cfg.DEPTH_CKPT != '':
    #     if os.path.isfile(cfg.DEPTH_CKPT):
    #         print("== Loading depth checkpoint '{}'".format(cfg.DEPTH_CKPT))
    #         depth_dict = model.depth.state_dict()
    #         backbone_dict = model.backbone.state_dict()
    #         pretrained_dict = torch.load(cfg.DEPTH_CKPT, map_location='cpu')
    #         pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict['model'].items()}
    #         pretrained_backbone_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if 'backbone' in k}
    #         pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if 'depth' in k}

    #         filtered_dict = {k: v for k, v in pretrained_dict.items() if k in depth_dict and 'backbone' not in k}
    #         fil_backbone_dict = {k: v for k, v in pretrained_backbone_dict.items() if k in backbone_dict}
    #         assert filtered_dict.keys() == depth_dict.keys(), f"filtered_dict keys{pretrained_dict.keys()}, depth_dict keys{depth_dict.keys()}"
    #         assert fil_backbone_dict.keys() == backbone_dict.keys(), "keys for backbone cannot be fully loaded"
    #         depth_dict.update(filtered_dict)
    #         backbone_dict.update(fil_backbone_dict)
    #         try:
    #             # print(depth_dict.keys())
    #             model.depth.load_state_dict(depth_dict, strict= True)
    #             model.backbone.load_state_dict(backbone_dict, strict= True)
    #             print("== Loaded depth checkpoint '{}'".format(cfg.DEPTH_CKPT))
    #         except:
    #             print("== Failed to load depth checkpoint '{}'".format(cfg.DEPTH_CKPT))
    #             pass
    #         del depth_dict, pretrained_dict, filtered_dict
    #     else:
    #         print("== No depth checkpoint found at '{}'".format(cfg.DEPTH_CKPT))
            
    # if cfg.SEG_HEAD_CKPT!= '':
    #     if os.path.isfile(cfg.SEG_HEAD_CKPT):
    #         print("== Loading SEG_HEAD checkpoint '{}'".format(cfg.SEG_HEAD_CKPT))
    #         seg_head_dict = model.seg_head.state_dict()
    #         pretrained_dict = torch.load(cfg.SEG_HEAD_CKPT, map_location='cpu')['model_state_dict']
    #         selected_dict = {k.replace('seg_head.', ''): v for k, v in pretrained_dict.items() if 'seg_head.' in k}
    #         assert seg_head_dict.keys() == selected_dict.keys(), f"seg_head_dict{seg_head_dict.keys()}!= selected_dict{selected_dict.keys()}"
    #         try:
    #             model.seg_head.load_state_dict(selected_dict)
    #             print("== Loaded seg_head checkpoint '{}'".format(cfg.SEG_HEAD_CKPT))
    #         except:
    #             print("== Failed to load seg_head checkpoint '{}'".format(cfg.SEG_HEAD_CKPT))
    #             pass

    # if cfg.DETECT_CKPT != '':
    #     if os.path.isfile(cfg.DETECT_CKPT):
    #         print("== Loading detection checkpoint '{}'".format(cfg.DETECT_CKPT))
    #         pretrained_dict = torch.load(cfg.DETECT_CKPT, map_location='cpu')
    #         pretrained_dict = pretrained_dict['model']
    #         # print(pretrained_dict.keys())
    #         try:
    #             for part_name, prefix in [('neck', 'backbone'), ('head', 'head')]:
    #                 part = getattr(model, part_name)  # get the model part
    #                 part_dict = part.state_dict()  # get the part's current weights

    #                 # Filter pretrained_dict for keys related to this part, update part_dict
    #                 if part_name == 'neck':
    #                     filtered_dict = {k.replace(prefix + '.', ''): v for k, v in pretrained_dict.items() if k.startswith(prefix) and not k.startswith('backbone.backbone')}
    #                 else:
    #                     filtered_dict = {k.replace(prefix + '.', ''): v for k, v in pretrained_dict.items() if k.startswith(prefix) and not 'cls_preds' in k}
                        
    #                 part_dict.update(filtered_dict)

    #                 # Load the updated weights into the part
    #                 part.load_state_dict(part_dict)
    #             print("== Loaded detection checkpoint '{}'".format(cfg.DETECT_CKPT))
    #         except Exception as e:
    #             print("== Failed to load detection checkpoint '{}'".format(cfg.DETECT_CKPT))
    #             print(e)
    #             pass
    #     else:
    #         print("== No detection checkpoint found at '{}'".format(cfg.DETECT_CKPT))
            
def init_seg_head_weights(model, cfgs):
    ori_dict = torch.load(cfgs.ORI_CKPT, map_location='cpu')
    filtered_dict = {'.'.join(k.split('.')[1:]): v for k, v in ori_dict.items()}
    filtered_dict = {k: v for k, v in filtered_dict.items() if k.startswith('seg_head')}
    model.load_state_dict(filtered_dict, strict=False)
    print("==segmentation head Weights Loaded==")
def init_det_head_weights(model, cfgs):
    ori_dict = torch.load(cfgs.ORI_CKPT, map_location='cpu')
    filtered_dict = {'.'.join(k.split('.')[1:]): v for k, v in ori_dict.items()}
    filtered_dict = {k: v for k, v in filtered_dict.items() if k.startswith('head')}
    model.load_state_dict(filtered_dict, strict=False)
    print("==detection head Weights Loaded==")
def init_backbone_weights(model, cfgs):
    ori_dict = torch.load(cfgs.ORI_CKPT, map_location='cpu')
    filtered_dict = {'.'.join(k.split('.')[1:]): v for k, v in ori_dict.items()}
    filtered_dict = {k: v for k, v in filtered_dict.items() if k.startswith('backbone.')}
    model.load_state_dict(filtered_dict, strict=False)
    print("==backbone Weights Loaded==")
    
def init_backbone_neck_weights(model, cfgs):
    ori_dict = torch.load(cfgs.ORI_CKPT, map_location='cpu')
    filtered_dict = {'.'.join(k.split('.')[1:]): v for k, v in ori_dict.items()}
    filtered_dict = {k: v for k, v in filtered_dict.items() if k.startswith('backbone.') or k.startswith('neck.')}
    model.load_state_dict(filtered_dict, strict=False)
    print("==backbone and neck Weights Loaded==")

def print_time_cost(model, num_iter):
    print("time costs: ")
    print("yolo_precess_time: ", model.yolo_process_time / num_iter)
    print("tracker_process_time: ", model.tracker_process_time / num_iter)
    print("traj_pred_process_time: ", model.traj_pred_process_time / num_iter)
    print("depth_process_time: ", model.depth_process_time / num_iter)
    
def video_creater(frame_path):
    output_dir = os.path.join(frame_path, 'video')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = 'video.mp4'
    frame_rate = 30
    frame_size = (512, 256)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, frame_rate, frame_size)

    for frame_file in sorted(os.listdir(frame_path)):
        # Ensure file is an image
        if frame_file.endswith('.png'):
            # Read image
            img = cv2.imread(os.path.join(frame_path, frame_file))
            # Add image to video
            video.write(img)
    video.release()

def mIOU(segmentation, gt_masks):
    # Binarize predictions for single-class segmentation
    preds_binary = (segmentation > 0.5).float()
    gt = gt_masks.unsqueeze(1)
    assert preds_binary.shape == gt.shape, "Predictions and ground truth tensors must have the same shape"

    intersection = (preds_binary * gt).sum(dim=(2, 3))
    false_positives = (preds_binary * (1 - gt)).sum(dim=(2, 3))
    false_negatives = ((1 - preds_binary) * gt).sum(dim=(2, 3))

    union = intersection + false_positives + false_negatives

    # Calculate IoU for each image in the batch
    iou = intersection / (union + 1e-6)

    # Calculate mean IoU across the batch
    return iou.mean()
