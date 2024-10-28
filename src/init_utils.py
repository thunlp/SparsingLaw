# This file is used for various initializations, including the model, dataset, lr_scheduler, and optimizer.
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import re
import json
import torch
import pickle
import bmtrain as bmt

from cpm.dragonfly.modeling_dragonfly import Dragonfly, DragonflyConfig
from cpm.dragonfly.training_tasks.pretrain_indexed import CudaPrefetcher, MixedIndexedDataset, UnpadBatchedMixedDataset
from cpm.training_utils.lr_scheduler import WarmupStableExp
from cpm.utils import exporter
from collections import namedtuple
from transformers import LlamaTokenizerFast
from torch.utils.data import DataLoader

from model_utils import SparseDragonfly
from utils import ROOT_DIR

TrainArgs = namedtuple('TrainArgs', [
    'lr',
    'batch_size',
    'max_length',
    'weight_decay',
    'train_iters',
    'warmup_iters',
    'drop_iters',
    'grad_accum',
    'clip_grad',
    'inspect_training_dynamics',
])

SaveArgs = namedtuple('SaveArgs', [
    'save',
    'save_name',
    'save_iters',
    'tokenizer_path',
    'parallel_load_datastate',
])

def get_train_args(
        config_path: str,
        train_iters: int,
        inspect_training_dynamics: int,
    ) -> TrainArgs:
    """
    Reads training parameters from a configuration file and returns a TrainArgs object.

    Args:
        config_path (str): Path to the configuration file.
        inspect_training_dynamics (int): Flag indicating whether to inspect training dynamics.
        - 0: Do not inspect training dynamics.
        - Non-zero: Inspect training dynamics.
    Returns:
        TrainArgs: TrainArgs object containing the training parameters.
    """
    with open(config_path) as f:
        cfg = json.loads(f.read())

    grad_accum = cfg['pretrain']['glb_batch_size'] // cfg['pretrain']['batch_size'] // bmt.world_size()
    if train_iters != None:
        cfg['pretrain']['train_iters'] = train_iters
    bmt.print_rank('grad_accum =', grad_accum, 'train_iters =', cfg['pretrain']['train_iters'] * grad_accum)
    bmt.print_rank('batch size =', bmt.world_size() * cfg['pretrain']['batch_size'] * grad_accum)

    train_args = TrainArgs(
        lr=cfg['pretrain']['lr'],
        batch_size=cfg['pretrain']['batch_size'],
        max_length=cfg['pretrain']['max_length'],
        weight_decay=0.1,
        train_iters=cfg['pretrain']['train_iters'] * grad_accum,
        warmup_iters=cfg['pretrain']['warmup_iters'],
        drop_iters=0.,
        grad_accum=grad_accum,
        clip_grad=1,
        inspect_training_dynamics=inspect_training_dynamics,
    )
    bmt.print_rank(train_args)
    return train_args

def get_model(
    config_path: str,
    ckpt_path: str,
    activate_fn: str = 'silu',
) -> SparseDragonfly:
    model_config = DragonflyConfig.from_json_file(config_path)
    model_config.activate_fn = activate_fn
    bmt.print_rank("activate function is", activate_fn)
    
    model = SparseDragonfly(model_config)
    if ckpt_path is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + ckpt_path)
        args = namedtuple('Args', ['load'])(load=ckpt_path)
        exporter.load_model_ckpt(args, model)
    else:
        bmt.init_parameters(model)

    bmt.print_rank('number of parameter: {:d}'.format(sum(x._original_shape.numel() for x in model.parameters())))
    bmt.print_rank('number of parameter without embedding: {:d}'.format(
        sum(x._original_shape.numel() for k, x in model.named_parameters() if 'embed' not in k)))
    bmt.print_rank('dim_model / num_layer = {:d} / {:d} = {:.3f}'.format(
        model.config.dim_model, model.config.num_layers, model.config.dim_model / model.config.num_layers))
    return model

def get_model_from_name(
    model_name: str,
    ckpt_path: str,
):
    if '_' in model_name:
        model_id = model_name.split('_')[0]
        activate_fn = model_name.split('_')[1]
    else:
        model_id = model_name
        activate_fn = 'silu'

    config_path = os.path.join(ROOT_DIR, 'configs', 'model_configs', model_id + '.json')
    return get_model(
        config_path=config_path,
        ckpt_path=ckpt_path,
        activate_fn=activate_fn,
    )

def get_tokenizer(path: str) -> LlamaTokenizerFast:
    if os.path.isdir(path):
        all_files = os.listdir(path)
        pkl_files = [file for file in all_files if file.endswith('.pkl')]
        assert len(pkl_files) == 1
        tok_path = os.path.join(path, pkl_files[0])
    else:
        tok_path = path

    with open(tok_path, 'rb') as f:
        tokenizer: LlamaTokenizerFast = pickle.load(f)
    return tokenizer

def get_valid_dataset():
    VALID_DATASET_PATH = '/home/test/test06/pkl_valid_data'
    assert bmt.world_size() == 8
    dataloader = torch.load(os.path.join(VALID_DATASET_PATH, 'valid_data_{}.pkl'.format(bmt.rank())))
    return dataloader

def get_dataset(
        config_path: str,
        tokenizer,
        batch_size: int,
        max_length: int,
        ckpt_path: str = None,
        seed: int = 42,
    ):
    mixed_indexed_dataset = MixedIndexedDataset(
        cfg_path=config_path,
        cfg_json_str=None,
        tokenizer=tokenizer,
        max_length=max_length,
        nthreads=1,
        prefetch_slice=2,
        weight_by_size=True,
        seed=seed,
    )
    if ckpt_path != None:
        args = namedtuple('Args', ['load', 'parallel_load_datastate'])(load=ckpt_path, parallel_load_datastate=256)
        exporter.load_dataloader_ckpt(args, mixed_indexed_dataset)
        bmt.print_rank("dataloader is loaded!")

    batched_dataset = UnpadBatchedMixedDataset(
        mixed_indexed_dataset,
        batch_size=batch_size,
        max_length=max_length
    )
    dataloader = DataLoader(
        batched_dataset,
        batch_size=None,
        collate_fn=lambda x: x,
        num_workers=2,
        prefetch_factor=50,
    )

    data_iterator = CudaPrefetcher(dataloader, tp_size=1, tp_rank=bmt.config["tp_rank"])

    return mixed_indexed_dataset, data_iterator

def get_optimizer(
        model: Dragonfly,
        lr: float,
        weight_decay: float,
        ckpt_path = None,
        save_name = None,
        start_step = None,
    ):
    scale_lr_group = []
    normal_group = []
    scale_lr_group_name, normal_group_name = [], []
    for n, p in model.named_parameters():
        if n.endswith(".weight") and "layernorm" not in n and "embedding" not in n and "lm_head" not in n:
            scale_lr_group.append(p)
            scale_lr_group_name.append(n)
        else:
            normal_group.append(p)
            normal_group_name.append(n)
    
    param_groups = [
        {"params": scale_lr_group, "lr": lr / model.config.scale_width},
        {"params": normal_group, "lr": lr},
    ]
    optimizer = bmt.optim.AdamOffloadOptimizer(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    if ckpt_path != None:
        opt_num = sum(
            [1 if re.search(r"-{}.rank-\d+.opt".format(start_step), i) else 0 for i in os.listdir(ckpt_path)]
        )
        if opt_num != bmt.world_size():
            bmt.print_rank('number of optimizer checkpoint files mismatch, stop loading optimizer!')
        else:
            args = namedtuple('Args', ['load', 'save_name', 'start_step'])(
                load=ckpt_path,
                save_name=save_name,
                start_step=start_step,
            )
            exporter.load_optimizer_ckpt(args, optimizer)
            bmt.print_rank("optimizer is loaded!")
    return optimizer

def get_learning_rate_scheduler(
        optimizer,
        lr: float,
        train_iters: float,
        warmup_iters: float,
        drop_begin: int = -1,
        start_step: int = 0,
    ):
    if 0 < warmup_iters < 1:
        warmup_iters = int(train_iters * warmup_iters)
    else:
        warmup_iters = int(warmup_iters)

    bmt.print_rank('drop begin is', drop_begin)
    lr_scheduler = WarmupStableExp(
        optimizer,
        start_lr=lr,
        warmup_iter=warmup_iters,
        drop_begin=drop_begin,
        drop_rate=0.5,
        drop_iter=560,
        num_iter=start_step,
    )

    return lr_scheduler

def get_log_ckpt(ckpt_path: str):
    if ckpt_path != None:
        args = namedtuple('Args', ['load'])(load=ckpt_path)
        log_ckpt = exporter.load_log_ckpt(args) 
    else:
        log_ckpt = {
            "global_token_pass": 0,
            "iteration": 0,
        }
    return log_ckpt