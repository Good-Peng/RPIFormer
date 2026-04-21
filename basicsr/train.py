import argparse
import datetime
import logging
import math
import os

import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.misc import mkdir_and_rename2
from basicsr.utils.options import dict2str, parse

import numpy as np

from pdb import set_trace as stx

# 参数解析
def parse_options(is_train=True):
    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', type=str, default='options/train/rpiformer_lolv1.yml', help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # 2. 解析命令行参数
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)    # 解析YAML配置文件

    # 3. 分布式训练设置
    if args.launcher == 'none':
        opt['dist'] = False    # 单机训练
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True      # 分布式训练
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)    # 初始化分布式环境
            print('init dist .. ', args.launcher)

    # 4. 获取分布式信息
    opt['rank'], opt['world_size'] = get_dist_info()    # 当前进程rank和总进程数

    # 5. 设置随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])     # 每个进程使用不同的种子

    return opt

# 日志初始化
def init_loggers(opt):
    # 1. 创建训练日志文件
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # 2. 创建指标记录CSV文件
    log_file = osp.join(opt['path']['log'],
                        f"metric.csv")
    logger_metric = get_root_logger(logger_name='metric',
                                    log_level=logging.INFO, log_file=log_file)
    # 3. 写入CSV表头
    metric_str = f'iter ({get_time_str()})'
    for k, v in opt['val']['metrics'].items():
        metric_str += f',{k}'
    logger_metric.info(metric_str)

    # 4. 记录环境信息和配置
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    # if (opt['logger'].get('wandb')
    #         is not None) and (opt['logger']['wandb'].get('project')
    #                           is not None) and ('debug' not in opt['name']):
    #     assert opt['logger'].get('use_tb_logger') is True, (
    #         'should turn on tensorboard when using wandb')
    #     init_wandb_logger(opt)

    # 5. 初始化TensorBoard日志（可选）
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


# 数据加载器创建
def create_train_val_dataloader(opt, logger):  #train loader 和 val loader 一起构建
    # create train and val dataloaders
    train_loader, val_loader = None, None

    # 遍历配置中的数据集设置
    for phase, dataset_opt in opt['datasets'].items():
        # stx()
        if phase == 'train':
            # 1. 创建训练数据集
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)                                 #将option中的dataset参数传入create_dataset中构建train_set
            # stx()
            # 2. 创建数据采样器（用于分布式训练）
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)     #缺少关键字 world_size 和 rank，train_sampler是做什么？从get_dist_info得到
            # stx()
            # 3. 创建训练数据加载器
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            # stx()

            # 4. 计算训练统计信息
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))  #一个epoch遍历一次数据
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) #一个iteration就是一次 inference + backward，总的iteration是不变的
            # 5. 记录训练统计信息
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        # 创建验证数据集和加载器
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            # stx()
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters

# 主训练流程
def main():
    # parse options, set distributed setting, set ramdom seed
    # 1. 解析配置参数
    opt = parse_options(is_train=True)

    # 2. 设置CUDA优化
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 3. 自动恢复训练状态
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name']) #状态路径
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    # 找到最新的训练状态文件
    resume_state = None
    if len(states) > 0: #如果路径已存在
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary，resume_state是重新训练的时候接上的吗？
    # 4. 加载训练状态（如果存在）
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    # 5. 创建实验目录
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename2(
                osp.join('tb_logger', opt['name']), opt['rename_flag'])

    # 6. 初始化日志记录器
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    # 7. 创建数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # 8. 创建模型
    if resume_state:  # 恢复训练
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        best_metric = resume_state['best_metric']
        best_psnr = best_metric['psnr']
        best_iter = best_metric['iter']
        logger.info(f'best psnr: {best_psnr} from iteration {best_iter}')
    else:   # 新训练
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
        best_metric = {'iter': 0}
        for k, v in opt['val']['metrics'].items():
            best_metric[k] = 0
        # stx()

    # create message logger (formatted outputs)
    # 9. 创建消息记录器
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 10. 创建数据预取器
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch

    # 训练循环的核心部分
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)      # 设置epoch，确保分布式训练的数据分布
        prefetcher.reset()                  # 重置数据预取器
        train_data = prefetcher.next()      # 获取下一批数据
        
        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # 1. 更新学习率
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # ------Progressive learning ---------------------
            # 2. 渐进式学习（Progressive Learning）
            j = ((current_iter > groups) != True).nonzero()[
                0]  # 根据当前的iter次数判断在哪个阶段
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]          # 当前阶段的图像尺寸
            mini_batch_size = mini_batch_sizes[bs_j]    # 当前阶段的批次大小

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(
                    mini_gt_size, mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            # 3. 数据预处理
            lq = train_data['lq']   # 低质量图像
            gt = train_data['gt']   # 高质量图像（ground truth）

            # 4. 批次大小调整
            if mini_batch_size < batch_size:  # 默认生成batch_size对图片，小于就要抽样
                indices = random.sample(
                    range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            # 5. 图像尺寸调整（随机裁剪）
            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            # -------------------------------------------
            # print(lq.shape)
            # 6. 模型训练
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # 7. 日志记录
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            # 8. 保存检查点
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter, best_metric=best_metric)

            # 9. 验证
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                current_metric = model.validation(val_loader, current_iter, tb_logger,
                                                  opt['val']['save_img'], rgb2bgr, use_image)
                # log cur metric to csv file
                # 记录到CSV文件
                logger_metric = get_root_logger(logger_name='metric')
                metric_str = f'{current_iter},{current_metric}'
                logger_metric.info(metric_str)

                # log best metric
                # 更新最佳指标
                if best_metric['psnr'] < current_metric:
                    best_metric['psnr'] = current_metric
                    # save best model
                    best_metric['iter'] = current_iter
                    model.save_best(best_metric)
                if tb_logger:
                    tb_logger.add_scalar(  # best iter
                        f'metrics/best_iter', best_metric['iter'], current_iter)
                    for k, v in opt['val']['metrics'].items():  # best_psnr
                        tb_logger.add_scalar(
                            f'metrics/best_{k}', best_metric[k], current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    # end of epoch
    # 训练结束处理
    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    # 保存最终模型
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1表示保存最新模型
    # 最终验证
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    # 关闭TensorBoard日志
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
