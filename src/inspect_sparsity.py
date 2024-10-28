# This file is a script for calculating sparsity by calling functions from `model_utils.py`. Please refer to `run_inspect.sh` to execute it.
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import time
import torch
import argparse
import bmtrain as bmt

from cpm.dragonfly.training_tasks.pretrain_indexed import MixedIndexedDataset

from math import exp
from torch import Tensor
from collections import defaultdict
from typing import Iterable, Dict, Tuple

from model_utils import SparseDragonfly
from init_utils import get_dataset, get_tokenizer, get_model, get_train_args, get_valid_dataset
from utils import ROOT_DIR

def initialize(
        tokenizer_path: str,
        model_name: str,
        dataset_name: str,
        load_path: str,
        **kwargs, # drop other arguments
    ):
    bmt.init_distributed()

    model_id = model_name.split('_')[0]
    activate_fn = model_name.split('_')[1]

    if dataset_name == 'valid':
        bmt.print_rank("use valid dataset")
        dataloader = get_valid_dataset()
        dataset = None
        train_args = None
    else:
        train_config_id = model_id.split('-')[0]
        train_config_path = os.path.join(ROOT_DIR, 'configs', 'train_configs', train_config_id + '.json')
        train_args = get_train_args(
            train_config_path,
            train_iters=None,
            inspect_training_dynamics=0,
        )
        dataset_config_path = os.path.join(ROOT_DIR, 'configs', 'dataset_configs', dataset_name + '.json')
        tokenizer = get_tokenizer(tokenizer_path)
        dataset, dataloader = get_dataset(
            config_path=dataset_config_path,
            tokenizer=tokenizer,
            batch_size=train_args.batch_size,
            max_length=train_args.max_length,
            ckpt_path=None,
            seed=19260817,
        )
    # fetch the first 10 pieces of data
    fix_batch_size = 6 if train_args == None else None
    part_dataloader = []
    for iteration, data in enumerate(dataloader, start=1):
        if dataset != None:
            dataset.update_states(data['task_ids'], data['indexes'])

        if fix_batch_size != None:
            len = fix_batch_size * 4096
            data['inputs'] = data['inputs'][:, :len]
            data['position_ids'] = data['position_ids'][:, :len]
            data['targets'] = data['targets'][:, :len]
            data['cu_seqlens'] = torch.cat((
                data['cu_seqlens'][data['cu_seqlens'] < len],
                torch.tensor([len], dtype=data['cu_seqlens'].dtype).cuda()
            ), dim=-1)

        part_dataloader.append(data)
        if iteration == 10: break

    model_config_path = os.path.join(ROOT_DIR, 'configs', 'model_configs', model_id + '.json')
    model = get_model(
        config_path=model_config_path,
        ckpt_path=load_path,
        activate_fn=activate_fn,
    )

    return model, part_dataloader

def inspect_model(
        model: SparseDragonfly,
        dataloader: Iterable[dict[str, Tensor]],
    ):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    sparsity_by_calc_list = []
    sparsity_list = []
    sparse_loss_list = []
    ppl_ratio_list = []

    for iteration, data in enumerate(dataloader, start=1):
        # pure run
        model.set_sparse_activate(False)
        logits = model.forward(
            input=data['inputs'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            position_ids=data['position_ids'],
        )
        target = data['targets'].view(-1)
        loss = loss_func(logits.view(-1, logits.size(-1)), target)
        global_loss = bmt.sum_loss(loss).item()
        # end

        if model.prune_strategy == 'cett':
            att_mask=target.view(logits.size(0), -1) != -100
            global_model_sparsity_by_calc = model.calc_sparsity(
                cett_upper_bound=model.prune_arg,
                att_mask=att_mask,
            )

        model.set_sparse_activate(True)
        logits_sparse = model.forward(
            input=data['inputs'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            position_ids=data['position_ids'],
        )
        global_model_sparsity = bmt.sum_loss(model.get_sparsity_after_inference())
        if model.prune_strategy == 'cett':
            bmt.print_rank('calc sparsity: {:.6f}, real sparsity: {:.6f}'.format(global_model_sparsity_by_calc, global_model_sparsity))

        loss_sparse = loss_func(logits_sparse.view(-1, logits_sparse.size(-1)), target)
        global_loss_sparse = bmt.sum_loss(loss_sparse).item()
        ppl_ratio = exp(global_loss_sparse - global_loss)

        bmt.print_rank("| Iter: {iteration:6d} | loss: {loss:.4f} | sparsity: {sparsity:.6f} | sparse loss: {sparse_loss:.4f} | PPL-ratio: {ppl_ratio:.6f}".format(
            iteration=iteration,
            loss=global_loss,
            sparsity=global_model_sparsity,
            sparse_loss=global_loss_sparse,
            ppl_ratio=ppl_ratio,
        ))

        if model.prune_strategy == 'cett':
            sparsity_by_calc_list.append(global_model_sparsity_by_calc)
        sparsity_list.append(global_model_sparsity)
        sparse_loss_list.append(global_loss_sparse)
        ppl_ratio_list.append(ppl_ratio)

    if model.prune_strategy == 'cett':
        mean_sparsity_by_calc = torch.tensor(sparsity_by_calc_list).mean().item()
        bmt.print_rank('mean sparsity:', mean_sparsity_by_calc)
    mean_sparsity = torch.tensor(sparsity_list).mean().item()
    mean_sparse_loss = torch.tensor(sparse_loss_list).mean().item()
    mean_ppl_ratio = torch.tensor(ppl_ratio_list).mean().item()
    
    bmt.print_rank('mean real sparsity:', mean_sparsity)
    bmt.print_rank('mean sparse loss:', mean_sparse_loss)
    bmt.print_rank('mean ppl-ratio:', mean_ppl_ratio)

    return mean_sparsity, mean_ppl_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', required=True, type=str)
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--dataset-name', required=True, type=str)
    parser.add_argument('--load-path', required=False, default=None, type=str)

    parser.add_argument('--prune-strategy', required=True, type=str)
    parser.add_argument('--prune-arg', required=False, type=float)
    # for compatibility
    parser.add_argument('--cett-upper-bound', required=False, type=float)

    # parse arguments
    args = parser.parse_args()
    model, dataloader = initialize(**vars(args))
    prune_strategy, prune_arg, cett_upper_bound = args.prune_strategy, args.prune_arg, args.cett_upper_bound
    if prune_arg == None:
        prune_arg = cett_upper_bound

    model.prune_strategy = prune_strategy
    model.prune_arg = prune_arg
    inspect_model(
        model=model,
        dataloader=dataloader,
    )

if __name__ == '__main__':
    main()