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

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import DiffGPTModel, DiffGPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os
import ipdb;
st = ipdb.set_trace
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn
    
data_debug=None
debug_t=None
debug_p_mask=None
debug_masked_indices=None

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = DiffGPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = DiffGPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator, eps=1e-3):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack: diff model's labels and token are the same
    tokens_ = data_b['text'].long()
    tokens = tokens_[:, :-1].contiguous()

    micro_batch_size, seq_length = tokens.size()
    t = torch.rand(micro_batch_size, device=tokens.device)
    
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, seq_length)

    masked_indices = torch.rand((micro_batch_size, seq_length), device=tokens.device) < p_mask
    noisy_input = torch.where(masked_indices, args.padded_vocab_size, tokens)

    # create attention mask (no need so all zeros)
    attention_mask = torch.ones(1, 1, seq_length, seq_length, device=tokens.device)
    attention_mask = (attention_mask < 0.5)

    # create position ids
    position_ids = torch.arange(seq_length, device=tokens.device)
    position_ids = position_ids[None, :].repeat(micro_batch_size, 1)

    return noisy_input, tokens, attention_mask, position_ids, masked_indices, p_mask


def get_batch_pipe(data, eps=1e-3):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""

    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    tokens = tokens_[:, :-1].contiguous() # remove last token

    micro_batch_size, seq_length = tokens.size()
    t = torch.rand((micro_batch_size,), device=tokens.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, seq_length)

    masked_indices = torch.rand((micro_batch_size, seq_length), device=tokens.device) < p_mask
    noisy_input = torch.where(masked_indices, tokenizer.vocab_size, tokens)
    
    # create attention mask (no need so all ones)
    attention_mask = torch.ones(1, 1, seq_length, seq_length, device=tokens.device)
    attention_mask = (attention_mask < 0.5)

    # create position ids
    position_ids = torch.arange(seq_length, device=tokens.device)
    position_ids = position_ids[None, :].repeat(micro_batch_size, 1)

    return (noisy_input, position_ids, attention_mask), (tokens, masked_indices, p_mask)


def loss_func(tokens, masked_indices, p_mask, output_loss_tensor):
    losses = output_loss_tensor.float()
    losses_selected = losses[masked_indices] / p_mask[masked_indices]
    loss = losses_selected.sum() / (tokens.shape[0] * tokens.shape[1])

    # regulation loss
    # reg_loss = losses[~masked_indices] / (1 - p_mask[~masked_indices])
    # reg_loss = reg_loss.sum() / (tokens.shape[0] * tokens.shape[1])

    # alpha = 0.5
    # total_loss = alpha * loss + (1 - alpha) * reg_loss

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    noisy_input, tokens, attention_mask, position_ids, masked_indices, p_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(noisy_input, position_ids, attention_mask,
                          labels=tokens)
    loss, output = output_tensor
    # check output using argmax
    output_argmax = torch.argmax(output, dim=-1)
    
    # print(f"First 10 noisy input: {noisy_input[0][:10].detach().cpu().numpy()}")
    # print(f"First 10 tokens: {tokens[0][:10].detach().cpu().numpy()}")
    # print(f"First 10 output: {output_argmax[0][:10].detach().cpu().numpy()}")
    # print(f"First 10 loss: {loss[0][:10].detach().cpu().numpy()}")

    # print(f"Noisy input: {noisy_input[masked_indices][:10].detach().cpu().numpy()}")
    # print(f"Tokens: {tokens[masked_indices][:10].detach().cpu().numpy()}")
    # print(f"Output: {output_argmax[masked_indices][:10].detach().cpu().numpy()}")
    # print(f"Loss: {loss[masked_indices][:10].detach().cpu().numpy()}")

    return loss, partial(loss_func, output_argmax, masked_indices, p_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
