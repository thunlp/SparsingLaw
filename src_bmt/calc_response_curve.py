# This file is used to calc the distribution of the neuron activation frequencies. You can execute `torchrun calc_response_curve.py` to run this file.
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import gc
import os
import pickle
import torch
import argparse
import bmtrain as bmt

from torch import Tensor
from transformers import LlamaTokenizerFast
from typing import Dict, List

from init_utils import get_tokenizer, get_model_from_name, get_dataset
from model_utils import SparseDragonfly
from utils import ROOT_DIR, MODEL_TO_CETT
from utils import MODEL_TO_FINAL_CHECKPOINT_PATH

model_list = {
    '0.1b_silu',
    '0.2b_silu',
    '0.4b-v3_silu',
    '0.8b_silu',
    '1.2b_silu',
}

TOKENIZER_PATH = '/home/test/test06/lyq/tokenizer.pkl'

def work_for_dataset(dataset_name: str):
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    dataset_config_path = os.path.join(ROOT_DIR, 'configs', 'dataset_configs', dataset_name + '.json')
    dataset, dataloader = get_dataset(
        config_path=dataset_config_path,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=4096,
        ckpt_path=None,
        seed=19260819,
    )

    data_list = []
    # fetch the first 10 pieces of data
    for iteration, data in enumerate(dataloader, start=1):
        dataset.update_states(data['task_ids'], data['indexes'])
        data['att_mask'] = data['targets'].view(data['inputs'].size()) != -100
        data_list.append(data)
        if iteration == 10:
            # print an example data
            example = tokenizer.decode(data['inputs'].view(-1))
            bmt.print_rank('example data:', example)
            break

    result = {}
    from tqdm import tqdm
    for model_name in model_list:
        model = get_model_from_name(model_name, MODEL_TO_FINAL_CHECKPOINT_PATH[model_name])

        for data in tqdm(data_list):
            logits = model.forward(
                input=data['inputs'],
                cu_seqlens=data['cu_seqlens'],
                max_seqlen=data['max_seqlen'],
                position_ids=data['position_ids'],
            )
            activation = 1 - model.calc_sparsity(
                cett_upper_bound=MODEL_TO_CETT[model_name],
                att_mask=data['att_mask'],
            )

        activation_rate = model.activation_count_per_neuron / model.tokens_count_all # num_layers, dim_ff
        result[model_name] = activation_rate.cpu().numpy()

        del model
        torch.cuda.empty_cache()
        gc.collect()

    with open('/home/test/test06/lyq/outputs/response_curve_{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(result, file=f)

def main():
    bmt.init_distributed()

    work_for_dataset('fit')
    work_for_dataset('wiki')
    work_for_dataset('math')
    work_for_dataset('code')
    work_for_dataset('chinese')

if __name__ == '__main__':
    main()