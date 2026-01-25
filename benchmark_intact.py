#!/usr/bin/env python3
"""
Standalone script to benchmark InTAct on ImageNet-R
Prints: total parameters, trainable parameters, GPU memory, batch time
"""
import os
import sys
import torch
import numpy as np
import yaml
import argparse
from torch.utils.data import DataLoader

import dataloaders
from dataloaders.utils import get_transform
import learners
from learners.default import accumulate_acc
import models
from regularization.interval_regularization import IntervalPenalization
from utils.metric import AverageMeter, Timer, accuracy


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return memory_allocated, memory_reserved
    return 0.0, 0.0


def benchmark_intact(config_path='configs/imnet-r_prompt_5_tasks.yaml', 
                     num_batches=50, 
                     gpuid=0):
    """
    Run InTAct benchmark on ImageNet-R
    
    Args:
        config_path: Path to config file
        num_batches: Number of batches to benchmark
        gpuid: GPU device ID
    """
    
    # Load config
    print("="*80)
    print("InTAct Benchmark on ImageNet-R")
    print("="*80)
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # Override some settings for benchmarking
    config['gpuid'] = [gpuid]
    config['learner_type'] = 'prompt'
    config['learner_name'] = 'Prompt'
    config['prompt_param'] = [100, 8, 0.0]  # pool_size, prompt_length, ortho_mu
    config['use_interval_activation'] = True
    config['model_type'] = 'zoo'
    config['model_name'] = 'vit_pt_imnet'
    config['dataset'] = 'ImageNet_R'
    config['dataroot'] = config.get('dataroot', 'data')
    config['workers'] = config.get('workers', 1)
    config['batch_size'] = config.get('batch_size', 64)
    config['first_split_size'] = config.get('first_split_size', 40)
    config['other_split_size'] = config.get('other_split_size', 40)
    config['rand_split'] = config.get('rand_split', True)
    config['validation'] = config.get('validation', False)
    config['train_aug'] = config.get('train_aug', True)
    config['schedule'] = config.get('schedule', [50])
    config['schedule_type'] = config.get('schedule_type', 'cosine')
    config['optimizer'] = config.get('optimizer', 'Adam')
    config['lr'] = config.get('lr', 0.001)
    config['momentum'] = config.get('momentum', 0.9)
    config['weight_decay'] = config.get('weight_decay', 0.0)
    config['temp'] = config.get('temp', 2.0)
    config['memory'] = config.get('memory', 0)
    config['DW'] = config.get('DW', False)
    config['upper_bound_flag'] = False
    config['debug_mode'] = False
    config['overwrite'] = True
    config['out_dim'] = 200
    config['top_k'] = 1
    config['dil'] = False
    
    # Set random seed
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup dataset
    print("\n[1/5] Setting up ImageNet-R dataset...")
    Dataset = dataloaders.iIMAGENET_R
    num_classes = 200
    
    # Create task splits
    class_order = np.arange(num_classes)
    if config['rand_split']:
        np.random.shuffle(class_order)
    
    tasks = []
    p = 0
    while p < num_classes:
        inc = config['other_split_size'] if p > 0 else config['first_split_size']
        tasks.append(class_order[p:p+inc])
        p += inc
    
    num_tasks = len(tasks)
    print(f"   Number of tasks: {num_tasks}")
    print(f"   Classes per task: {[len(t) for t in tasks]}")
    
    # Create dataloader for first task
    resize_imnet = True
    train_transform = get_transform(dataset='ImageNet_R', phase='train', 
                                   aug=config['train_aug'], resize_imnet=resize_imnet)
    
    train_dataset = Dataset(
        config['dataroot'], 
        train=True, 
        lab=True, 
        tasks=tasks,
        download_flag=False, 
        transform=train_transform, 
        seed=seed, 
        rand_split=config['rand_split'], 
        validation=config['validation'],
        data_root_dir=config.get('data_root_dir', '/shared/sets/datasets/')
    )
    
    # Load first task
    train_dataset.load_dataset(0, train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True, 
        num_workers=config['workers']
    )
    
    # Create learner config
    print("\n[2/5] Creating InTAct model...")
    learner_config = {
        'num_classes': num_classes,
        'lr': config['lr'],
        'debug_mode': config['debug_mode'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'schedule': config['schedule'],
        'schedule_type': config['schedule_type'],
        'model_type': config['model_type'],
        'model_name': config['model_name'],
        'optimizer': config['optimizer'],
        'gpuid': config['gpuid'],
        'memory': config['memory'],
        'temp': config['temp'],
        'out_dim': num_classes,
        'overwrite': config['overwrite'],
        'DW': config['DW'],
        'batch_size': config['batch_size'],
        'upper_bound_flag': config['upper_bound_flag'],
        'tasks': tasks,
        'top_k': config['top_k'],
        'prompt_param': [num_tasks, config['prompt_param']],
        'use_interval_activation': config['use_interval_activation'],
        'dil': config['dil']
    }
    
    # Create learner (InTAct)
    learner = learners.prompt.Prompt(learner_config)
    learner.add_valid_output_dim(len(tasks[0]))
    
    # Set task id
    if hasattr(learner.model, 'module'):
        learner.model.module.task_id = 0
    else:
        learner.model.task_id = 0
    
    # Setup interval penalization (InTAct's key component)
    interval_penalization = IntervalPenalization(
        var_loss_scale=config.get('var_loss_scale', 0.01),
        internal_repr_drift_loss_scale=config.get('internal_repr_drift_loss_scale', 1.0),
        feature_loss_scale=config.get('feature_loss_scale', 1.0),
        use_align_loss=config.get('use_align_loss', True)
    )
    
    interval_penalization.setup_task(
        task_id=0,
        curr_classifier_head=learner.model.module.classifier if hasattr(learner.model, 'module') 
            else learner.model.classifier,
        feature_extractor=learner.model.module.feat if hasattr(learner.model, 'module') 
            else learner.model.feat,
        prompt=learner.model.prompt if hasattr(learner.model, 'prompt') else None
    )
    
    # Count parameters
    print("\n[3/5] Counting parameters...")
    total_params, trainable_params = count_parameters(learner.model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Warm up GPU
    print("\n[4/5] Warming up GPU...")
    learner.model.train()
    for i, (x, y, task) in enumerate(train_loader):
        if i >= 3:  # Just 3 warmup iterations
            break
        if learner.gpu:
            x = x.cuda()
            y = y.cuda()
        
        # Model update (following trainer pattern)
        loss, output = learner.update_model(x, y, interval_penalization=interval_penalization)
    
    # Clear cache and measure GPU memory
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Benchmark training - following trainer.py learn_batch pattern
    print("\n[5/5] Benchmarking training...")
    print(f"   Running {num_batches} batches...")
    
    # Setup data weighting (following trainer pattern)
    learner.data_weighting(train_dataset)
    
    # Initialize meters like in trainer
    losses = AverageMeter()
    acc = AverageMeter()
    batch_time = AverageMeter()
    batch_timer = Timer()
    
    learner.model.train()
    batch_count = 0
    
    for epoch in range(100):  # Large number to ensure we get enough batches
        batch_timer.tic()
        for i, (x, y, task) in enumerate(train_loader):
            if batch_count >= num_batches:
                break
            
            # verify in train mode
            learner.model.train()
            
            # send data to gpu
            if learner.gpu:
                x = x.cuda()
                y = y.cuda()
            
            # model update (exactly as in trainer)
            loss, output = learner.update_model(x, y, interval_penalization=interval_penalization)
            
            # measure elapsed time (exactly as in trainer)
            batch_time.update(batch_timer.toc())
            batch_timer.tic()
            
            # measure accuracy and record loss
            y = y.detach()
            accumulate_acc(output, y, task, acc, topk=(learner.top_k,))
            losses.update(loss, y.size(0))
            batch_timer.tic()
            
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"   Batch {batch_count}/{num_batches}: {batch_time.avg:.4f}s avg, {batch_time.val:.4f}s current")
        
        if batch_count >= num_batches:
            break
        
        # Reset meters at end of epoch (like trainer)
        if batch_count < num_batches:
            losses = AverageMeter()
            acc = AverageMeter()
    
    # Get GPU memory usage
    memory_allocated, memory_reserved = get_gpu_memory_usage()
    
    # Calculate statistics
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    min_batch_time = np.min(batch_times)
    max_batch_time = np.max(batch_times)
    
    # Get final average batch time (as trainer returns batch_time.avg)
    avg_batch_time = batch_time.avg
    
    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\n📊 Model Statistics:")
    print(f"   Total Parameters:        {total_params:,}")
    print(f"   Trainable Parameters:    {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    
    print(f"\n💾 GPU Memory Utilization:")
    print(f"   Allocated:  {memory_allocated:.2f} GB")
    print(f"   Reserved:   {memory_reserved:.2f} GB")
    
    print(f"\n⏱️  Batch Time (averaged over {batch_count} batches):")
    print(f"   Average:    {avg_batch_time:.4f} s")
    print(f"   Last batch: {batch_time.val:.4f} s")
    
    print(f"\n📈 Training Metrics:")
    print(f"   Final Loss:     {losses.avg:.3f}")
    print(f"   Final Accuracy: {acc.avg:.2f}%")
    
    print("\n" + "="*80)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gpu_memory_allocated_gb': memory_allocated,
        'gpu_memory_reserved_gb': memory_reserved,
        'avg_batch_time_s': avg_batch_time,
        'final_loss': losses.avg,
        'final_accuracy': acc.avgtr, default='configs/imnet-r_prompt_5_tasks.yaml',
                       help='Path to config file')
    parser.add_argument('--num_batches', type=int, default=50,
                       help='Number of batches to benchmark')
    parser.add_argument('--gpuid', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (GPU memory will be 0)")
    
    results = benchmark_intact(
        config_path=args.config,
        num_batches=args.num_batches,
        gpuid=args.gpuid
    )
