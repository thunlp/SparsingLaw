# This file is used to calculate the thresholds of each layers under PPL-k% setting given a model checkpoint and a dataset. Please refer to `calc_threshold_for_each_layer.sh` to execute it.
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import gc
import torch
import argparse
import bmtrain as bmt

from math import exp
from torch import Tensor
from typing import Iterable, List

from init_utils import get_dataset, get_tokenizer, get_model_from_name, get_train_args, get_valid_dataset
from model_utils import SparseDragonfly
from utils import ROOT_DIR

from inspect_sparsity import inspect_model

def initialize(
    tokenizer_path: str,
    model_name: str,
    dataset_name: str,
    load_path: str,
    **kargs, # drop other args
):
    bmt.init_distributed()
    #assert dataset_name == 'valid'
    if dataset_name == 'valid':
        dataloader = get_valid_dataset()
    else:
        model_id = model_name.split('_')[0]
        train_config_id = model_id.split('-')[0]
        dataset_config_path = os.path.join(ROOT_DIR, 'configs', 'dataset_configs', dataset_name + '.json')
        train_config_path = os.path.join(ROOT_DIR, 'configs', 'train_configs', train_config_id + '.json')
        train_args = get_train_args(
            train_config_path,
            train_iters=None,
            inspect_training_dynamics=0,
        )
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
        part_dataloader = []
        for iteration, data in enumerate(dataloader, start=1):
            dataset.update_states(data['task_ids'], data['indexes'])
            part_dataloader.append(data)
            if iteration == 10: break

    model = get_model_from_name(model_name, load_path)

    return part_dataloader, model

def calc_cett_for_target_ppl_ratio(
    model: SparseDragonfly,
    target_ppl_ratio: float,
    dataloader: Iterable[dict[str, Tensor]],
) -> float:
    cett_lower = 0.
    cett_upper = 1.
    tolerance = 0.001
    while cett_upper - cett_lower > tolerance:
        cett = (cett_lower + cett_upper) / 2
        bmt.print_rank("cett now is : {:.4f}".format(cett))

        model.prune_strategy = 'cett'
        model.prune_arg = cett
        sparsity, ppl_ratio = inspect_model(
            model=model,
            dataloader=dataloader,
        )

        bmt.print_rank("inspect results with cett {:.6f} : mean sparsity {:.6f} , mean ppl-ratio {:.6f}".format(
            cett, sparsity, ppl_ratio))

        if ppl_ratio < target_ppl_ratio:
            cett_lower = cett
        else:
            cett_upper = cett

    return (cett_lower + cett_upper) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', required=True, type=str)
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--dataset-name', required=True, type=str)
    parser.add_argument('--load-path', required=True, type=str)
    parser.add_argument('--target-ppl-ratio', required=True, type=float)

    # parse arguments
    args = parser.parse_args()
    dataloader, model = initialize(**vars(args))
    target_ppl_ratio = args.target_ppl_ratio

    cett = calc_cett_for_target_ppl_ratio(
        model=model,
        target_ppl_ratio=target_ppl_ratio,
        dataloader=dataloader,
    )
    bmt.print_rank('The result CETT is {:.4f}'.format(cett))
    threshold_list = {}
    num_layers = model.config.num_layers
    for lid in range(num_layers):
        threshold_list[lid] = []

    for iteration, data in enumerate(dataloader, start=1):
        model.set_sparse_activate(False)
        logits = model.forward(
            input=data['inputs'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            position_ids=data['position_ids'],
        )

        att_mask=data['targets'].view(logits.size(0), -1) != -100
        global_model_sparsity = model.calc_sparsity(
            cett_upper_bound=cett,
            att_mask=att_mask,
        )

        for lid in range(num_layers):
            threshold_list[lid].append(float(model.threshold_per_layer[lid]))

        bmt.print_rank("| Iter: {iteration:6d} | sparsity: {sparsity:.6f}".format(
            iteration=iteration, sparsity=global_model_sparsity,))

    res = []
    for lid in range(num_layers):
        t = sum(threshold_list[lid]) / len(threshold_list[lid])
        bmt.print_rank('layer: {}, t: {:.5e}'.format(lid, t))
        res.append(t)
    bmt.print_rank(','.join(str(x) for x in res))

if __name__ == '__main__':
    main()