# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from deepspeed.accelerator import get_accelerator
import torch
if get_accelerator().device_name() == 'cuda':
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD
else:
    from torch.optim import Adam
    from torch.optim import SGD

from megatron import get_args
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer


def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n == 'bias'])

    # XXX: temp hack to workaround the crash in apex FusedAdam's multi_tensor_applier
    #
    # it crashes when the param count is larger than a certain size which we hit at 200B over 80
    # A100 gpus - I think around 2.7B per gpu, so halving it works around the issue
    param_count = len(weight_decay_params['params'])
    first_half = weight_decay_params['params'][:param_count // 2]
    second_half = weight_decay_params['params'][param_count // 2:]

    first_half =  { 'params': first_half }
    second_half = { 'params': second_half }

    return first_half, second_half, no_weight_decay_params

    #return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)
    
    if args.cpu_optimizer:
        assert args.optimizer == 'adam', 'CPU offloading is for Adam'
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       betas=(args.adam_beta1, args.adam_beta2),
                                       eps=args.adam_eps)
    else:
        if args.optimizer == 'adam':
            if args.use_bnb_optimizer:
                import bitsandbytes as bnb
                adam_optimizer = bnb.optim.Adam8bit
            else:
                adam_optimizer = Adam
            optimizer = adam_optimizer(param_groups,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    betas=(args.adam_beta1, args.adam_beta2),
                                    eps=args.adam_eps)
        elif args.optimizer == 'sgd':
            optimizer = SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.sgd_momentum)
        else:
            raise Exception('{} optimizer is not supported.'.format(
                args.optimizer))

    if args.deepspeed:
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad)
