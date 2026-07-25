from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from models.hint.interval_modules import parse_logits

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets, interval_penalization=None, hnet_reg=None):

        out, prompt_loss = self.model(inputs, train=True)
        
        if isinstance(out, tuple) and len(out) == 3:
            lower_logits, middle_logits, upper_logits = parse_logits(out)
        else:
            middle_logits = out
            lower_logits = upper_logits = None

        # Masking for Valid Out Dim
        middle_logits = middle_logits[:,:self.valid_out_dim]
        if lower_logits is not None:
            lower_logits = lower_logits[:,:self.valid_out_dim]
            upper_logits = upper_logits[:,:self.valid_out_dim]

        # CE with heuristic (Apply to all bounds if they exist)
        if not self.dil:
            middle_logits[:,:self.last_valid_out_dim] = -float('inf')
            if lower_logits is not None:
                lower_logits[:,:self.last_valid_out_dim] = -float('inf')
                upper_logits[:,:self.last_valid_out_dim] = -float('inf')
                
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        # 3. Calculate Worst-Case IBP Loss & Kappa Schedule
        loss_fit = self.criterion(middle_logits, targets.long(), dw_cls)

        if lower_logits is not None and upper_logits is not None:
            # Kappa Scheduling
            iterations_to_adjust = self.iterations_to_adjust
            target_kappa = 0.5
            
            if hasattr(self, 'current_iter') and self.current_iter < iterations_to_adjust:
                kappa = max(1.0 - 0.00005 * self.current_iter, target_kappa)
            else:
                kappa = target_kappa
                
            # Worst-case loss component
            tmp = F.one_hot(targets.long(), middle_logits.size(-1))
            z = torch.where(tmp.bool(), lower_logits, upper_logits)
            
            loss_spec = self.criterion(z, targets.long(), dw_cls)
            
            # Combine Standard + Worst-Case Loss
            total_loss = kappa * loss_fit + (1 - kappa) * loss_spec
            
            # Store worst case error for metrics/logging if needed later
            self.worst_case_error = (z.argmax(dim=1) != targets).float().sum().item()
        else:
            total_loss = loss_fit

        # 4. Interval Penalization
        if interval_penalization is not None:
            total_loss += interval_penalization.forward(inputs, total_loss)

        # 5. Hypernetwork Regularization Loss (HINT)
        if hnet_reg is not None:
            task_id = self.model.module.task_id if hasattr(self.model, 'module') else getattr(self.model, 'task_id', 0)
            
            if task_id > 0:
                reg_loss = hnet_reg.calc_fix_target_reg(
                    task_id=task_id,
                    lower_targets=self.hnet_lower_targets,
                    middle_targets=self.hnet_middle_targets,
                    upper_targets=self.hnet_upper_targets
                )
                
                beta = self.config['hnet_loss_reg']
                total_loss += (beta * reg_loss) / task_id

        # 6. Prompt Regularization Loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        
        # Retain graph if either penalization or regularization requires it
        retain = (interval_penalization is not None) or (hnet_reg is not None)
        total_loss.backward(retain_graph=retain)
        self.optimizer.step()
        
        # Increment iteration for epsilon and kappa scheduling
        self.current_iter += 1

        # Return middle logits for accuracy calculations
        return total_loss.detach(), middle_logits

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if isinstance(self.model, torch.nn.DataParallel):
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.classifier.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.classifier.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass
    
    def cuda(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config['gpuid'][0])
            self.model = self.model.cuda()
            self.criterion_fn = self.criterion_fn.cuda()
            # Multi-GPU
            if len(self.config['gpuid']) > 1:
                self.model = torch.nn.DataParallel(
                    self.model, 
                    device_ids=self.config['gpuid'], 
                    output_device=self.config['gpuid'][0]
                )
        else:
            self.model = self.model.cpu()
            self.criterion_fn = self.criterion_fn.cpu()
        return self

# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param,
                                                                               use_interval_activation=cfg['use_interval_activation'],
                                                                               use_hint=cfg['use_hint'])
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param,
                                                                               use_interval_activation=cfg['use_interval_activation'],
                                                                               use_hint=cfg['use_hint'])
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param,
                                                                               use_interval_activation=cfg['use_interval_activation'],
                                                                               use_hint=cfg['use_hint'])
        return model