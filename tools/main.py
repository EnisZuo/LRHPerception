import argparse
import gc
import torch, os, time
import logging
import itertools
from loguru import logger
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from configs import cfg
from utils.data_utils import *
# from utils.data_utils.depth_dataloader import NewDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from packet.structure import DTPDNet
from utils.visualization import Visualizer, viz_results
from utils.logger import Logger
from datetime import datetime
from packet.structure.losses import *
from utils.utils import *
from termcolor import cprint

@logger.catch
def eval(model, cfg):
    print("== Model Initialized to eval mode ==")
    init_weights(model, cfg.CKPT)
    # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # initiate visualizer
    if cfg.VISUALIZE:
        save_frame_path = os.path.join(cfg.EVAL.OUTPUT_DIR, 'saved_frames')
        if not os.path.exists(save_frame_path):
            os.makedirs(save_frame_path)
        viz = Visualizer(cfg, mode='image', save_frame_dir=save_frame_path)
    
    logger = None
    if cfg.USE_WANDB:
        logger = Logger("MPED_RNN",
                cfg,
                project = cfg.PROJECT,
                viz_backend="wandb"
                )        
    # else:
    #     logger = logging.Logger("MPED_RNN")
    
    if cfg.EVAL.TASK == 'det':
        print("Evaluating detection on KITTI detection 2D dataset...")
        det_dataloader = NewDataLoader(cfg, mode='train', dataset='kitti_det')
        with torch.no_grad():
            for i, det_data in enumerate(det_dataloader.data):
                det_imgs, img_path, det_labels = det_data
                det_imgs = det_imgs.cuda(cfg.GPU)
                img_path = img_path[0]
                print(det_imgs.shape)
                det_labels = det_labels.cuda(cfg.GPU)
                dets, seg, hist_traj, pred_goal, pred_traj, depth_out = model(det_imgs, img_path)
                print(dets.shape)
            
            
    elif cfg.EVAL.DATASET_NAME == 'mot':
        is_distributed = cfg.NUM_GPUS > 1
        eval_dataloader = MOT_Dataloaders(cfg).get_eval_loader(is_distributed=is_distributed)
        with torch.no_grad():
            for iter, (imgs, target, info_imgs, img_id) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Inference"):
                imgs = imgs.to(cfg.DEVICE)
                # print(info_imgs[-1][0].split('/')[0])
                img_path = os.path.join(cfg.DATASET.ROOT, cfg.EVAL.DATASET_NAME, cfg.EVAL.DATASPLIT_NAME, info_imgs[-1][0])
                dets, seg, hist_traj, pred_goal, pred_traj, depth_out = model(imgs, img_path)
                if cfg.VISUALIZE and iter % max(int(len(eval_dataloader)/20), 1) == 0:
                    viz_results(viz, dets, seg, hist_traj, None, pred_goal, depth_out, img_path,
                                bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')
                    
    elif cfg.EVAL.DATASET_NAME == 'kitti_raw':
        print('Evaluating on KITTI raw dataset...')
        if cfg.EVAL.SPEED_TEST:
            k = 0
            with torch.no_grad():
                print("\033[31m " + "== Model Warming Up for Speed Test ==" + "\033[0m")
                test_loader = get_kitti_raw_dataloader(cfg, split='test', batch_size=1)
                image, path = None, None
                for iter, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference Progress"):
                    img = data['img'].to(cfg.DEVICE)
                    img_path = data['img_path'][0]
                    _ = model(img, img_path)
                    k+=1
                    if k == 150:
                        image, path = img, img_path
                        break
                test_loader = get_kitti_raw_dataloader(cfg, split='test', batch_size=1)
                print("\033[32m" + "== Speed Testing in Progress ==" + "\033[0m")
                start = time.time()
                for _ in range(300):
                    _ = model(image, path)
                end = time.time()
                print("Speed for One Image:", (end-start)/300)

        test_loader = get_kitti_raw_dataloader(cfg, split='test', batch_size=1)
        empty_list = []
        for i in range(14):
            empty_list.append(torch.zeros(0, 45, 4))
        with torch.no_grad():
            global_step = 0
            for iter, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference Progress"):
                img = data['img'].to(cfg.DEVICE)
                img_path = data['img_path'][0]
                dets, seg, hist_traj, pred_goal, pred_traj, depth_out = model(img, img_path)
                empty_list.append(pred_traj)
                if cfg.VISUALIZE:
                    viz_results(viz, dets, seg, hist_traj, None, pred_traj, depth_out, img_path,
                                bbox_type=cfg.DATASET.BBOX_TYPE, normalized=False, logger=logger, name='pred_test')

                global_step += 1
                if global_step == 1200:
                    quit()

def train(cfgs, model, mode):
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    writer = SummaryWriter(log_dir= cfgs.TRAIN.OUTPUT_DIR + '/logs/')
    # current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_save_path = os.path.join(cfgs.TRAIN.OUTPUT_DIR, cfgs.TRAIN.TASKS, 'checkpoints')
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, exist_ok=True)
        
    log_save_path = os.path.join(cfgs.TRAIN.OUTPUT_DIR, cfgs.TRAIN.TASKS, 'logs')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path, exist_ok=True)
    log_file = open(log_save_path + 'training_log.txt', 'w')
        
    if mode == 'all':
        model.train()
        det_dataloader = NewDataLoader(cfgs, mode='train', dataset='kitti_det')
        cycled_det = itertools.cycle(det_dataloader.data)
        seg_dataloader = get_dataloader(cfgs, cfgs.DATASET.CITY_ROOT, 'train', cfgs.TRAIN.BATCH_SIZE)
        seg_val_loader = get_dataloader(cfgs, cfgs.DATASET.CITY_ROOT, 'val', 2 * cfgs.TRAIN.BATCH_SIZE)
        cycled_seg = itertools.cycle(seg_dataloader)
        depth_loader = NewDataLoader(cfgs, mode='train', dataset='depth')
        depth_val_loader = NewDataLoader(cfgs, mode='online_eval', dataset='depth')

        seg_loss_func = CustomSegmentationLoss()
        params_group = [{'params': model.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.neck.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.seg_head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.depth.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
        optimizer = torch.optim.Adam(params_group)
        print(len(det_dataloader.data), len(seg_dataloader), len(depth_loader.data))

        global_step = 0
        for epoch in range(cfgs.TRAIN.ALL_EPOCHS):
            # depth_loader.train_sampler.set_epoch(epoch)
            for i, (det_data, seg_data, depth_data) in enumerate(zip(cycled_det, cycled_seg, tqdm(depth_loader.data))):
                optimizer.zero_grad()
                det_imgs, _, det_labels = det_data
                det_imgs = det_imgs.cuda(cfgs.GPU)
                det_labels = det_labels.cuda(cfgs.GPU)
                det_loss = model(det_imgs, det_labels=det_labels, specific = 'det')[0]
                det_loss.backward()

                seg_imgs = seg_data[0].cuda(cfgs.GPU)
                seg_gt = seg_data[2].float().unsqueeze(1).cuda(cfgs.GPU)
                seg_out= model(seg_imgs, specific = 'seg')
                seg_loss = 10 * seg_loss_func(seg_out, seg_gt)
                seg_loss.backward()

                depth_img = depth_data['image'].cuda(cfgs.GPU)
                depth_gt = depth_data['depth'].cuda(cfgs.GPU)
                depth_loss = model(depth_img, depth_gt = depth_gt, specific = 'depth')
                depth_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                optimizer.step()
                global_step += 1

                # print("det_loss: ", det_loss.item(), "seg_loss: ", seg_loss.item(), "depth_loss: ", depth_loss.item())
                combined_loss = det_loss.item() + seg_loss.item() + depth_loss.item()
                if global_step % cfgs.LOG_FREQ == (cfgs.LOG_FREQ - 1):
                    writer.add_scalar("entire model training loss", combined_loss, global_step)
                    print(f"combined_loss for step {global_step} is {combined_loss}")
                    torch.save(model.state_dict(), os.path.join(ckpt_save_path, f'ORI_distributed_step{global_step}_loss{combined_loss}.pth'))
                    
                    '''evaluate depth and segmentation'''
                    time.sleep(0.1)
                    model.eval()
                    with torch.no_grad():
                        total_mIOU = 0.0
                        for i, data in enumerate(tqdm(seg_val_loader)):
                            gt_mask = data[2].float().cuda(cfgs.GPU)
                            pred_mask = model(data[0].float().cuda(cfgs.GPU), specific='seg')
                            total_mIOU += mIOU(pred_mask, gt_mask)
                        avg_mIOU = total_mIOU / len(seg_val_loader)
                        cprint(f"Average mIOU is {avg_mIOU}","red", "on_green")
                    with torch.no_grad():
                        eval_measures = online_eval(cfgs, model, depth_val_loader, cfgs.GPU, cfgs.NGPU_PER_NODE, post_process=False)
                    block_print()
                    enable_print()
                    global_step += 1
                    model.module.train()
                

        writer.close()


    # train seg
    elif mode == 'seg':
        print('==training segmentation==')
        train_loader = get_kitti_road_dataloader(cfgs, split='train', batch_size=cfgs.TRAIN.BATCH_SIZE, save_mask=False)
        seg_loss = CustomSegmentationLoss()
        seg_params_group = [{'params': model.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.neck.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                    {'params': model.seg_head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
        seg_optimier = torch.optim.Adam(seg_params_group)
        for epoch in range(cfgs.TRAIN.SEG_EPOCHS):
            for batch_idx, batched_data in enumerate(train_loader):
                # for data in batched_data:
                seg_optimier.zero_grad()
                img = batched_data['img'].to(cfg.DEVICE)
                gt_img = batched_data['gt_img']
                gt_img = gt_img.float().to(cfg.DEVICE)
                seg_out = model(img, specific='seg')
                loss = seg_loss(seg_out, gt_img)
                loss.backward()
                seg_optimier.step()
                
                writer.add_scalar('loss/seg', loss.item(), epoch * len(train_loader) + batch_idx)
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch} | Batch: {batch_idx} | Seg Loss: {loss.item()}')
                    log_file.write(f'Epoch: {epoch} | Batch: {batch_idx} | Seg Loss: {loss.item()}\n')
                    
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': seg_optimier.state_dict(),
                'loss': loss.item()
            }, ckpt_save_path + f'epoch_{epoch}.pth')
            
    elif mode == 'det':
        cudnn.benchmark = True
        if cfgs.DATASET.DETECT.NAME == 'kitti':
            print('==training det==')
            det_dataloader = get_kitti_det_dataloader(cfgs, split='train', batch_size=cfgs.TRAIN.BATCH_SIZE)
            det_params_group = [{'params': model.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.neck.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
            
            optimizer = torch.optim.Adam(det_params_group)
            for epoch in range(cfgs.TRAIN.DET_EPOCHS):
                if cfgs.DISTRIBUTED:
                    det_dataloader.train_sampler.set_epoch(epoch)
                for batch_idx, det_data in enumerate(det_dataloader.data):
                    optimizer.zero_grad()
                    det_imgs, det_img_paths, det_labels = det_data
                    det_imgs = det_imgs.cuda(cfgs.GPU)
                    det_labels = det_labels.cuda(cfgs.GPU)
                    det_loss = model(det_imgs, det_labels=det_labels, specific='det')[0]

                    det_loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        print(f'Epoch: {epoch} | Batch: {batch_idx} | Det Loss: {det_loss}')
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': det_loss.item()
                }, ckpt_save_path + f'epoch_{epoch}.pth')
                
        else: #cityscapes
            model.train()
            model.module.depth.eval()
            transform = transforms.Compose([
                transforms.Resize((512, 1024)),
                transforms.ToTensor()
            ])
            train_loader = get_dataloader(cfgs, '/home/azuo/DSPNet/data', split='train', batch_size = cfgs.TRAIN.BATCH_SIZE, shuffle= True, transform=transform)
            val_loader = get_dataloader(cfgs, '/home/azuo/DSPNet/data', split='val', batch_size = cfgs.TRAIN.BATCH_SIZE, shuffle= True, transform=transform)
            print('==training det==')

            det_params_group = [{'params': model.module.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.module.head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
            
            optimizer = torch.optim.Adam(det_params_group)
            for epoch in range(cfgs.TRAIN.DET_EPOCHS):
                # if cfgs.DISTRIBUTED:
                #     det_dataloader.train_sampler.set_epoch(epoch)
                for batch_idx, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    det_imgs, det_labels, _ = data[0], data[1], data[2]

                    det_imgs = det_imgs.cuda(cfgs.GPU)
                    det_labels = det_labels.cuda(cfgs.GPU)
                    det_out, _ = model(det_imgs, det_labels=det_labels, specific='det')
                    det_loss = det_out[0]
                    det_loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        print(f'Epoch: {epoch} | Batch: {batch_idx} | Det Loss: {det_loss}')
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': det_loss.item()
                }, ckpt_save_path + f'epoch_{epoch}.pth')
    
    elif mode == 'det-seg':
        cudnn.benchmark = True
        if cfgs.DATASET.DETECT.NAME == 'kitti':
            print('==training det-seg==')
            det_dataloader = get_kitti_det_dataloader(cfgs, split='train', batch_size=cfgs.TRAIN.BATCH_SIZE)
            seg_dataloader = get_kitti_seg_dataloader(cfgs, split='train', batch_size=cfgs.TRAIN.BATCH_SIZE)
            seg_dataloader = itertools.cycle(seg_dataloader)
            seg_loss_func = CustomSegmentationLoss()
            det_params_group = [{'params': model.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.neck.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.seg_head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
            
            optimizer = torch.optim.Adam(det_params_group)
            for epoch in range(cfgs.TRAIN.DET_SEG_EPOCHS):
                if cfgs.DISTRIBUTED:
                    det_dataloader.train_sampler.set_epoch(epoch)
                for batch_idx, (det_data, seg_data) in enumerate(zip(det_dataloader.data, seg_dataloader)):
                    optimizer.zero_grad()
                    det_imgs, det_img_paths, det_labels = det_data

                    det_imgs = det_imgs.cuda(cfgs.GPU)
                    det_labels = det_labels.cuda(cfgs.GPU)
                    det_out, _ = model(det_imgs, det_labels=det_labels)
                    det_loss = det_out[0]

                    seg_imgs = seg_data['img'].cuda(cfgs.GPU)
                    seg_gt = seg_data['gt_img'].float().cuda(cfgs.GPU)
                    _, seg_out = model(seg_imgs, det_labels=None)
                    seg_loss = seg_loss_func(seg_out, seg_gt)
                    combined_loss = det_loss + seg_loss
                    combined_loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        print(f'Epoch: {epoch} | Batch: {batch_idx} | Combined Loss: {combined_loss}')
                    # writer.add_scalar('combined_loss/seg-det', combined_loss.item(), epoch * len(det_dataloader) + batch_idx)
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': combined_loss.item()
                }, ckpt_save_path + f'epoch_{epoch}.pth')
        else: #cityscapes
            model.train()
            model.module.depth.eval()
            seg_loss_func = CustomSegmentationLoss()
            transform = transforms.Compose([
                transforms.Resize((512, 1024)),
                transforms.ToTensor()
            ])
            train_loader = get_dataloader(cfgs, '/home/azuo/DSPNet/data', split='train', batch_size = cfgs.TRAIN.BATCH_SIZE, shuffle= True, transform=transform)
            val_loader = get_dataloader(cfgs, '/home/azuo/DSPNet/data', split='val', batch_size = cfgs.TRAIN.BATCH_SIZE, shuffle= True, transform=transform)
            print('==training det-seg==')

            det_params_group = [{'params': model.module.backbone.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.module.neck.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.module.head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE},
                                {'params': model.module.seg_head.parameters(), 'lr': cfgs.TRAIN.DET_SET_LEARNING_RATE}]
            
            optimizer = torch.optim.Adam(det_params_group)
            for epoch in range(cfgs.TRAIN.DET_SEG_EPOCHS):
                # if cfgs.DISTRIBUTED:
                #     det_dataloader.train_sampler.set_epoch(epoch)
                for batch_idx, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    det_imgs, det_labels, seg_labels = data[0], data[1], data[2]

                    det_imgs = det_imgs.cuda(cfgs.GPU)
                    det_labels = det_labels.cuda(cfgs.GPU)
                    seg_labels = seg_labels.cuda(cfgs.GPU)
                    det_out, seg_out = model(det_imgs, det_labels=det_labels)
                    seg_out = seg_out.squeeze(1)

                    det_loss = det_out[0]
                    seg_loss = seg_loss_func(seg_out, seg_labels)
                    combined_loss = det_loss + seg_loss
                    combined_loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        print(f'Epoch: {epoch} | Batch: {batch_idx} | Combined Loss: {combined_loss}')
                    # writer.add_scalar('combined_loss/seg-det', combined_loss.item(), epoch * len(det_dataloader) + batch_idx)
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': combined_loss.item()
                }, ckpt_save_path + f'epoch_{epoch}.pth')
                if epoch % 10 == 0:
                    eval_seg(model, val_loader, cfgs)
                
    # train depth
    if mode == 'depth':
        print('==training depth==')
        eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        train_loader = NewDataLoader(cfgs, mode='train')
        eval_loader = NewDataLoader(cfgs, mode='online_eval')
        depth_optimizer = torch.optim.Adam(list(model.depth.parameters())+ list(model.backbone.parameters()),\
                                            lr = cfgs.TRAIN.DEPTH_LEARNING_RATE)
        online_eval(cfgs, model, eval_loader, cfgs.GPU, 1)
        if cfg.DISTRIBUTED is False:
            for epoch in range(cfgs.TRAIN.DEPTH_NUM_EPOCHS):
                model.train()
                for batch_idx, sample_batched in enumerate(tqdm(train_loader.data)):
                    depth_optimizer.zero_grad()
                    image = torch.autograd.Variable(sample_batched['image'].cuda(cfgs.GPU, non_blocking=True))
                    depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(cfgs.GPU, non_blocking=True))
                    loss = model(x=image, depth_gt=depth_gt, specific = 'depth')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.depth.parameters(), 1.0)
                    depth_optimizer.step()
                    
                    # adjust learning rate
                    end_learning_rate = cfgs.TRAIN.DEPTH_LEARNING_RATE * 0.1
                    current_learning_rate = (cfgs.TRAIN.DEPTH_LEARNING_RATE - end_learning_rate) * (1 - epoch / cfgs.TRAIN.DEPTH_NUM_EPOCHS) ** 0.9 + end_learning_rate
                    for param_group in depth_optimizer.param_groups:
                        param_group['lr'] = current_learning_rate
                        
                    if batch_idx % 100 == 0:
                        print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}')

                if epoch % 2 == 0:
                    model.eval()
                    online_eval(cfgs, model, eval_loader, 0, 1)

                torch.save({
                    'epoch': epoch,
                    'model_depth_state_dict': model.depth.state_dict(),
                    'optimizer_state_dict': depth_optimizer.state_dict()}, ckpt_save_path + f'epoch_{epoch}.pth')
        else:
            cudnn.benchmark = True
            global_step = 0
            best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
            best_eval_measures_higher_better = torch.zeros(3).cpu()
            best_eval_steps = np.zeros(9, dtype=np.int32)
            if not cfgs.MULTIPROCESSING_DISTRIBUTED or (cfgs.MULTIPROCESSING_DISTRIBUTED and cfgs.RANK % cfgs.NGPU_PER_NODE == 0):
                writer = SummaryWriter(cfgs.CKPT.LOG_DIR + '/' + cfgs.MODEL_NAME + '/summaries', flush_secs=30)
                eval_summary_path = os.path.join(cfgs.CKPT.LOG_DIR, cfgs.MODEL_NAME, 'eval')
                eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

            duration = 0
            end_learning_rate = 0.1 * cfgs.TRAIN.DEPTH_LEARNING_RATE

            var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
            var_cnt = len(var_sum)
            var_sum = np.sum(var_sum)

            print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

            steps_per_epoch = len(train_loader.data)
            num_total_steps = cfgs.TRAIN.DEPTH_NUM_EPOCHS * steps_per_epoch
            epoch = global_step // steps_per_epoch

            for epoch in range(cfgs.TRAIN.DEPTH_NUM_EPOCHS):
                model.train()
                train_loader.train_sampler.set_epoch(epoch)
                for step, sample_batched in enumerate(tqdm(train_loader.data)):
                    depth_optimizer.zero_grad()
                    before_op_time = time.time()

                    image = torch.autograd.Variable(sample_batched['image'].cuda(cfgs.GPU, non_blocking=True))
                    depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(cfgs.GPU, non_blocking=True))
                    print('image.device', image.device)
                    loss = model(x=image, depth_gt=depth_gt, specific="depth")

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    depth_optimizer.step()

                    for param_group in depth_optimizer.param_groups:
                        current_lr = (cfgs.TRAIN.DEPTH_LEARNING_RATE - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                        param_group['lr'] = current_lr

                    #optimizer.step()

                    duration += time.time() - before_op_time
                    if global_step and global_step % cfgs.LOG_FREQ == (cfg.LOG_FREQ - 1) and cfgs.DEVICE == 0:

                        print('epoch:', epoch, 'global_step:', global_step, 'loss:', loss.item(), flush=True)

                    if global_step and global_step % cfgs.EVAL.EVAL_FREQ == (cfgs.EVAL.EVAL_FREQ - cfgs.EVAL.EVAL_FREQ + 1):
                        time.sleep(0.1)
                        print(f"global_step: {global_step}")
                        model.eval()
                        with torch.no_grad():
                            eval_measures = online_eval(cfgs, model, eval_loader, cfgs.DEVICE, cfgs.NGPU_PER_NODE, post_process=False)
                        if eval_measures is not None:
                            for i in range(9):
                                eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                                measure = eval_measures[i]
                                is_best = False
                                if i < 6 and measure < best_eval_measures_lower_better[i]:
                                    old_best = best_eval_measures_lower_better[i].item()
                                    best_eval_measures_lower_better[i] = measure.item()
                                    is_best = True
                                elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                                    old_best = best_eval_measures_higher_better[i-6].item()
                                    best_eval_measures_higher_better[i-6] = measure.item()
                                    is_best = True
                                if is_best:
                                    old_best_step = best_eval_steps[i]
                                    old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                                    model_path = cfgs.CKPT.LOG_DIR + '/' + cfgs.MODEL_NAME + old_best_name
                                    if os.path.exists(model_path):
                                        command = 'rm {}'.format(model_path)
                                        os.system(command)
                                    best_eval_steps[i] = global_step
                                    model_save_name = '/num_layer-{}_model-{}-best_{}_{:.5f}'.format(cfgs.MODEL.NUM_DUO_CONV_DPETH, global_step, eval_metrics[i], measure)
                                    print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                                    checkpoint = {'global_step': global_step,
                                                'model': model.state_dict(),
                                                'optimizer': depth_optimizer.state_dict(),
                                                'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                                'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                                'best_eval_steps': best_eval_steps
                                                }
                                    torch.save(checkpoint, cfgs.CKPT.LOG_DIR + '/' + cfgs.MODEL_NAME + model_save_name)
                            eval_summary_writer.flush()
                        model.module.depth.train()
                        block_print()
                        enable_print()
                    global_step += 1

                epoch += 1
    writer.close()

@logger.catch
def run(gpu, ngpus_per_node, cfgs):
    if cfgs.DISTRIBUTED:
        # if args.dist_url == "env://" and args.rank == -1:
        #     args.rank = int(os.environ["RANK"])
        # if cfg.multiprocessing_distributed:
        cfgs.RANK = cfgs.RANK * ngpus_per_node + gpu
        dist.init_process_group(backend=cfgs.DIST_BACKEND, init_method=cfgs.DIST_URL, world_size=3, rank=cfgs.RANK)
    print("init group done")
    model = DTPDNet(cfgs, train_task=cfgs.TRAIN.TASKS)
    model.train()
    mode = cfg.TRAIN.TASKS
    if not cfgs.TRAIN.START_NEW and mode == 'all':
        init_weights(model, cfgs.CKPT)
    elif mode == 'depth' and not cfgs.TRAIN.START_NEW:
        print("== Init depth backbone weights for training heads ==")
        init_backbone_weights(model, cfgs.CKPT)
    elif (mode == 'seg' or mode == 'det') and not cfgs.TRAIN.START_NEW:
        init_backbone_neck_weights(model, cfgs.CKPT)
        if mode == 'det':
            init_det_head_weights(model, cfgs.CKPT)
        if mode == 'seg':
            init_seg_head_weights(model, cfgs.CKPT)

    if cfgs.DISTRIBUTED:
        if gpu is not None:
            torch.cuda.set_device(cfgs.GPU)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(cfgs.GPU)
            cfg.batch_size = int(cfg.TRAIN.BATCH_SIZE / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfgs.GPU], find_unused_parameters=True)
            model.no_sync()
    else:
        model.cuda(cfgs.GPU)
    print("== Model Initialized on GPU: {}".format(cfgs.GPU))

    train(cfgs, model, mode)
        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # # parser.add_argument("--predictor-config-file", default="", metavar="FILE", help="path to config file")
    # # parser.add_argument("-f", "--exp_file", default='/home/azuo/LRHPerception/packages/yolox/exp/yolox_x_mix_det.py', type=str, help="pls input your tracking expriment description file")
    # # parser.add_argument("-t", "--train_task", default='none', type=str, help="")
    # args = parser.parse_args()

    
    if cfg.TRAIN.TASKS == 'pred':
        model = DTPDNet(cfg, train_task=cfg.TRAIN.TASKS)
        model.eval()
        model = model.to(cfg.DEVICE)
        eval(model, cfg)
    
    else:
        ngpus_per_node = torch.cuda.device_count()
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            cfg.WORLD_SIZE = ngpus_per_node * 1
            mp.spawn(run, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
        else:
            run(cfg.GPU, 1, cfg) # none: eval, trains: depth:depth, det-seg:det-seg, seg:road