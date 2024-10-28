# This file is used to calculate the CETT value under PPL-k% setting given several model checkpoints and a dataset.
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

from init_utils import get_dataset, get_tokenizer, get_model, get_train_args
from utils import ROOT_DIR

from inspect_sparsity import inspect_model

def initialize(
    tokenizer_path: str,
    model_name: str,
    dataset_name: str,
    checkpoint_list: str,
    **kargs, # drop other args
):
    bmt.init_distributed()

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
    data_list = []
    for iteration, data in enumerate(dataloader, start=1):
        dataset.update_states(data['task_ids'], data['indexes'])
        data_list.append(data)
        if iteration == 10: break

    checkpoint_list = checkpoint_list.split(',')

    return data_list, checkpoint_list

def calc_cett_for_target_ppl_ratio(
    model_name: str,
    target_ppl_ratio: float,
    dataloader: Iterable[dict[str, Tensor]],
    checkpoint_list: List[str],
) -> float:
    model_id = model_name.split('_')[0]
    activate_fn = model_name.split('_')[1]
    if not activate_fn:
        activate_fn = 'silu'
    model_config_path = os.path.join(ROOT_DIR, 'configs', 'model_configs', model_id + '.json')

    cett_lower = 0.
    cett_upper = 1.
    tolerance = 0.001
    while cett_upper - cett_lower > tolerance:
        cett = (cett_lower + cett_upper) / 2
        bmt.print_rank("cett now is : {:.4f}".format(cett))

        ppl_ratio_list = []
        for load_path in checkpoint_list:
            model = get_model(
                config_path=model_config_path,
                ckpt_path=load_path,
                activate_fn=activate_fn,
            )
            model.prune_strategy = 'cett'
            model.prune_arg = cett

            sparsity, ppl_ratio = inspect_model(
                model=model,
                dataloader=dataloader,
            )
            ppl_ratio_list.append(ppl_ratio)

            bmt.print_rank("on checkpoint: {:s}".format(load_path))
            bmt.print_rank("inspect results with cett {:.6f} : mean sparsity {:.6f} , mean ppl-ratio {:.6f}".format(
                cett, sparsity, ppl_ratio
            ))

            del model
            torch.cuda.empty_cache()
            gc.collect()

        mean_ppl_ratio = sum(ppl_ratio_list) / len(ppl_ratio_list)

        if mean_ppl_ratio < target_ppl_ratio:
            cett_lower = cett
        else:
            cett_upper = cett

    return (cett_lower + cett_upper) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-path', required=True, type=str)
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--dataset-name', required=True, type=str)
    parser.add_argument('--checkpoint-list', required=True, type=str)
    parser.add_argument('--target-ppl-ratio', required=True, type=float)
    # The found value of CETT will be written to {result_file_name}
    parser.add_argument('--result-file-name', required=True, type=str)

    # parse arguments
    args = parser.parse_args()
    dataloader, checkpoint_list = initialize(**vars(args))
    model_name, target_ppl_ratio, result_file_name = args.model_name, args.target_ppl_ratio, args.result_file_name

    cett = calc_cett_for_target_ppl_ratio(
        model_name=model_name,
        target_ppl_ratio=target_ppl_ratio,
        dataloader=dataloader,
        checkpoint_list=checkpoint_list
    )
    bmt.print_rank('The result CETT is {:.4f}'.format(cett))

    if bmt.rank() == 0:
        with open(result_file_name, 'w') as f:
            print('{:.4f}'.format(cett), file=f)
            bmt.print_rank('write result to {}'.format(result_file_name))

if __name__ == '__main__':
    main()