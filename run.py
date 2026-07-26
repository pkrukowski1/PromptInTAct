from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys

# ==============================================================================
# Intercept gpuid and set CUDA_VISIBLE_DEVICES BEFORE importing torch
# ==============================================================================
if '--gpuid' in sys.argv:
    try:
        gpuid_idx = sys.argv.index('--gpuid') + 1
        gpu_ids = []
        while gpuid_idx < len(sys.argv) and not sys.argv[gpuid_idx].startswith('--'):
            gpu_ids.append(sys.argv[gpuid_idx])
            gpuid_idx += 1
        
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu_ids)
    except Exception as e:
        print(f"Warning: Failed to parse --gpuid manually. Error: {e}")
# ==============================================================================

import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer
def create_args():
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid")
    parser.add_argument('--log_dir', type=str, default="outputs/out", help="Save experiments results in dir")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N', help="activate learner specific settings")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')
    
    # NEW: Explicit Random Seed
    parser.add_argument('--seed', type=int, default=0, help="Random seed for the experiment")
    # Kept for compatibility with trainer, but defaulted to 1
    parser.add_argument('--repeat', type=int, default=1, help="Keep at 1 for single seed execution")

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1], help="prompt params")
    parser.add_argument('--use_interval_activation', default=False, action='store_true', help="Use interval activations")
    parser.add_argument('--var_loss_scale', type=float, default=0.1, help="variance balancing")
    parser.add_argument('--internal_repr_drift_loss_scale', type=float, default=0.1, help="feature extractor output regularization")
    parser.add_argument('--feature_loss_scale', type=float, default=0.1, help="interval drift regularization")
    parser.add_argument('--use_align_loss', default=False, action='store_true', help="hypercube distance loss")

    # HINT model args
    parser.add_argument('--use_hint', default=False, action='store_true', help="Use HINT protection")
    parser.add_argument('--hnet_embedding_size', type=int, default=24, help="Dimension of the hypernetwork embedding")
    parser.add_argument('--perturbated_epsilon', type=float, default=1.0, help="Hypercube radius")
    parser.add_argument('--hnet_loss_reg', type=float, default=0.01, help="Hypernetwork regularization strength")
    parser.add_argument('--hnet_hidden_neurons', nargs="+", type=int, default=[100, 100], help="Neurons in hnet")

    # Data Args
    parser.add_argument('--data_root_dir', type=str, default="/shared/sets/datasets/", help="Root directory")
    parser.add_argument('--config', type=str, default="configs/config.yaml", help="yaml experiment config input")

    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)

class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # deterministic backend
    torch.backends.cudnn.deterministic = True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc','time']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    
    print('************************************')
    print(f'* STARTING TRIAL WITH SEED {args.seed}')
    print('************************************')

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set up a trainer
    trainer = Trainer(args, args.seed, metric_keys, save_keys)

    # init total run metrics storage
    max_task = trainer.max_task
    avg_metrics = {}
    
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: 
            avg_metrics[mkey][skey] = []
            
        # Keeping the 2D shape (max_task, 1) for compatibility with Trainer
        avg_metrics[mkey]['global'] = np.zeros((max_task, 1))
        if (not (mkey in global_only)):
            avg_metrics[mkey]['pt'] = np.zeros((max_task, max_task, 1))
            avg_metrics[mkey]['pt-local'] = np.zeros((max_task, max_task, 1))

    # train model
    avg_metrics = trainer.train(avg_metrics)  

    # evaluate model
    avg_metrics = trainer.evaluate(avg_metrics)    

    # save results
    for mkey in metric_keys: 
        m_dir = args.log_dir+'/results-'+mkey+'/'
        if not os.path.exists(m_dir): os.makedirs(m_dir)
        for skey in save_keys:
            if (not (mkey in global_only)) or (skey == 'global'):
                save_file = m_dir+skey+'.yaml'
                result = avg_metrics[mkey][skey]
                yaml_results = {}
                
                # Simplified YAML saving for a single run
                if len(result.shape) > 2:
                    yaml_results['mean'] = result[:,:,0].tolist()
                    yaml_results['history'] = result[:,:,0].tolist()
                else:
                    yaml_results['mean'] = result[:,0].tolist()
                    yaml_results['history'] = result[:,0].tolist()
                    
                with open(save_file, 'w') as yaml_file:
                    yaml.dump(yaml_results, yaml_file, default_flow_style=False)

    # Print the summary
    print('=== Summary of experiment ===')
    for mkey in metric_keys: 
        print(mkey, ' | final value:', avg_metrics[mkey]['global'][-1, 0])