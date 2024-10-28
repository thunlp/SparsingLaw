# This file is used to calculate the activation ratio distributions among tokens. You can execute `torchrun calc_activation_along_token.py` to run this file.
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
from utils import ROOT_DIR, MODEL_TO_FINAL_CHECKPOINT_PATH, MODEL_TO_CETT

TOKENIZER_PATH = '/home/test/test06/lyq/tokenizer.pkl'

def calc(
    data: Dict[str, Tensor],
    model: SparseDragonfly,
    lid: int,
) -> Tensor:
    state_dict = model.state_dict()

    weight: Tensor = state_dict[f"encoder.layers.{lid}.ffn.ffn.w_out.weight"].float().cuda()  # dim_model, dim_ff
    activate_value = model.intermedia[lid].float() # 1, n, dim_ff
    activate_value = activate_value[data['att_mask']] # m=sum(att_mask), dim_ff
    norm: Tensor = activate_value.abs() * weight.norm(dim=0)  # m, dim_ff
    threshold = model.threshold_per_layer[lid]

    activation_per_token = torch.mean((norm > threshold) * 1.0, dim=-1) # m,
    return activation_per_token

def work_for_result(
    data,
    result: Dict,
    return_logits: bool = False
):
    result.setdefault('data', {})
    logits_model = {}

    for model_name in MODEL_TO_FINAL_CHECKPOINT_PATH:
        model = get_model_from_name(model_name, MODEL_TO_FINAL_CHECKPOINT_PATH[model_name])

        logits = model.forward(
            input=data['inputs'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            position_ids=data['position_ids'],
        )
        logits_model[model_name] = logits
        activation = 1 - model.calc_sparsity(
            cett_upper_bound=MODEL_TO_CETT[model_name],
            att_mask=data['att_mask'],
        )

        res_of_nth_layer = []
        nl = model.config.num_layers
        for lid in range(nl):
            res_of_nth_layer.append(
                calc(
                    data=data,
                    model=model,
                    lid=lid,
                ).cpu().numpy()
            )

        result['data'].setdefault(model_name, {})
        result['data'][model_name].setdefault('average', ())
        result['data'][model_name]['average'] += sum(res_of_nth_layer) / nl,
        for pos in ['first', 'middle', 'last']:
            lid = {'first': 0, 'middle': nl//2, 'last': nl-1}[pos]
            #result['data'][model_name][pos] = res_of_nth_layer[lid]
            result['data'][model_name].setdefault(pos, ())
            result['data'][model_name][pos] += res_of_nth_layer[lid],

        del model
        torch.cuda.empty_cache()
        gc.collect()

    if return_logits:
        return logits_model

def work_single_sentence(sentence: str):
    tokenizer = get_tokenizer(TOKENIZER_PATH)

    data: Dict[str, Tensor] = {}
    data['inputs'] = tokenizer.encode(sentence, return_tensors='pt').cuda()
    n = data['inputs'].size(-1)
    data['cu_seqlens'] = torch.tensor([0, n], dtype=torch.int32).cuda()
    data['max_seqlen'] = 4096
    data['position_ids'] = torch.arange(n).view(1, n).cuda()
    data['att_mask'] = torch.ones(1, n, dtype=torch.bool).cuda()

    target = torch.zeros_like(data['inputs']).cuda()
    target[0, :-1] = data['inputs'][0, 1:]
    target[0, -1] = -100
    loss_func = bmt.loss.FusedCrossEntropy(reduction='none')

    result = {}
    logits_model = work_for_result(data, result, return_logits=True)
    result['tokens'] = tokenizer.tokenize(sentence, add_special_tokens=True)
    result['loss'] = {}
    with torch.no_grad():
        for model_name in logits_model:
            print(logits_model[model_name].view(n, -1))
            print(target)
            non_reduced_loss = loss_func(logits_model[model_name].view(n, -1), target)
            print(non_reduced_loss)
            result['loss'][model_name] = non_reduced_loss.view(-1).cpu().numpy()

    with open('/home/test/test06/lyq/outputs/activation_along_token_single_sentence.pkl', 'wb') as f:
        pickle.dump(result, file=f)

def work_for_dataset(dataset_name: str):
    tokenizer = get_tokenizer(TOKENIZER_PATH)
    dataset_config_path = os.path.join(ROOT_DIR, 'configs', 'dataset_configs', dataset_name + '.json')
    dataset, dataloader = get_dataset(
        config_path=dataset_config_path,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=4096,
        ckpt_path=None,
        seed=19260817,
    )

    result = {}
    # work the first 10 pieces of data
    for iteration, data in enumerate(dataloader, start=1):
        dataset.update_states(data['task_ids'], data['indexes'])
        data['att_mask'] = data['targets'].view(data['inputs'].size()) != -100
        
        work_for_result(data, result)
        bmt.print_rank('{}/10'.format(iteration))
        if iteration == 10: break

    with open('/home/test/test06/lyq/outputs/activation_along_many_token.pkl', 'wb') as f:
        pickle.dump(result, file=f)

def main():
    bmt.init_distributed()

    #sentence = '国足18强赛迎来一场焦点之争，上轮惨败的国足，回到主场迎战沙特。全场比赛国足开场取得领先，并且在对手罚下一人的情况下，竟然惨遭对手逆转，最终以1:2不敌对手。遭遇小组两连败的同时，排名小组垫底。'
    #work_single_sentence(sentence)

    work_for_dataset('fit')

if __name__ == '__main__':
    main()