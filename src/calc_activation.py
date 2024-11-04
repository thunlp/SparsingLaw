import os
import torch
import argparse
import numpy as np

from math import exp
from torch import Tensor
from typing import Dict, List
from transformers import AutoTokenizer
from model_utils import SparseMiniCPMForCausalLM

def calc_cett_for_target_ppl_ratio(
    model: SparseMiniCPMForCausalLM,
    inputs: Dict,
) -> float:
    assert model.prune_strategy == 'pplp'
    target_ppl_ratio = model.prune_arg

    # temporarily change to 'cett' for binary search
    model.prune_strategy = 'cett'

    cett_lower = 0.
    cett_upper = 1.
    tolerance = 1e-6
    while cett_upper - cett_lower > tolerance:
        cett = (cett_lower + cett_upper) / 2
        model.prune_arg = cett
        _, ppl_ratio = inspect_model(model, inputs, False)
        if ppl_ratio < target_ppl_ratio:
            cett_lower = cett
        else:
            cett_upper = cett

    model.prune_strategy = 'pplp'
    model.prune_arg = target_ppl_ratio
    return (cett_lower + cett_upper) / 2

def inspect_model(
    model: SparseMiniCPMForCausalLM,
    inputs: Dict,
    print_thresholds_list: bool = True,
):
    """
    Calculate the neuron activation ratios for each token by layer and compute the increase in perplexity (PPL ratio)
    for the provided input data. The analysis is conducted based on the model's pruning strategy 
    (`model.prune_strategy`) and pruning parameter (`model.prune_arg`).

    The meaning of `model.prune_arg` depends on `model.prune_strategy`:
    - The threshold epsilon for "fat" which represents FAT-eps.
    - The fixed ratio of activated neurons in each layer for "topk", which means that the value of k in 
      Top-k equals prune_arg * ffn_intermediate_dimension.
    - The CETT value for "cett".
    - 1 + p% for PPL-p% sparsity (e.g. prune_arg=1.01 for PPL-1% sparsity).

    Returns:
        - A Tensor of size (num_layers, batch_size, seq_len) represents the neuron activation ratios,
        - A float value represents the PPL ratio.
    """
    if model.prune_strategy == 'pplp':
        cett = calc_cett_for_target_ppl_ratio(model, inputs)
        target_ppl_ratio = model.prune_arg

        # temporarily change to 'cett'
        model.prune_strategy = 'cett'
        model.prune_arg = cett
        ret = inspect_model(model, inputs)
        model.prune_strategy = 'pplp'
        model.prune_arg = target_ppl_ratio
        return ret

    # dense run
    if model.prune_strategy == 'cett':
        model.save_intermedia = True
    model.set_sparse_activate(False)
    loss = model.forward(**inputs).loss
    if model.prune_strategy == 'cett':
        thresholds_list = model.calc_thresholds_given_cett(model.prune_arg, inputs['attention_mask'])
        if print_thresholds_list:
            print('Thresholds List for Evaluation:')
            print('thresholds=' + ','.join('{:.4e}'.format(x) for x in thresholds_list))
        model.set_thresholds(thresholds_list)
        model.save_intermedia = False

    # sparse run
    model.set_sparse_activate(True)
    loss_sparse = model.forward(**inputs).loss

    ppl_ratio = exp(loss_sparse - loss)
    return model.get_activation_data(), ppl_ratio

def draw(act_data: Tensor, tokens: List[str], setting: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    assert act_data.dim() == 3
    nl, bs, n = act_data.size()
    assert bs == 1
    act_data = act_data.squeeze(1).float().cpu().numpy() # nl, n

    fig, ax = plt.subplots(figsize=(n/5 + 1, nl/5))
    colors = ['#FFFFFF', '#3388BB']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    cax = ax.imshow(
        act_data,
        cmap=cmap,
        interpolation='nearest',
    )
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(tokens)) + 0.5)
    ax.set_xticklabels(tokens, rotation=65, fontsize=7, ha='right')

    ax.set_title(f'Neuron Activation Ratios of Tokens by Each Layer\nsetting: {setting}')
    ax.set_xlabel('Token')
    ax.set_ylabel('Index of Layer')

    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/activation.jpg', dpi=300)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-pretrained', required=True, type=str)
    parser.add_argument('--input-file', required=True, type=str)
    parser.add_argument('--prune-strategy', required=True, type=str)
    parser.add_argument('--prune-arg', required=True, type=float)
    parser.add_argument('--output-image', action='store_true')
    args = parser.parse_args()

    # init model and tokenizer
    with open(args.input_file) as f:
        sentence = f.read()
    path = args.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(path)
    model: SparseMiniCPMForCausalLM = SparseMiniCPMForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda()
    model.prune_strategy = args.prune_strategy
    model.prune_arg = args.prune_arg
    print('Input Text:', sentence)

    # tokenization
    inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
    inputs['labels'] = inputs['input_ids']
    # run inspection
    act_data, ppl_ratio = inspect_model(model, inputs)
    print('Average activation: {:2.2f}%'.format(act_data.mean() * 100))
    print('PPL ratio: {:.3f}%'.format(ppl_ratio * 100))

    # save to image if required
    if args.output_image:
        draw(
            act_data=act_data,
            tokens=tokenizer.tokenize(sentence, add_special_tokens=True),
            setting='{}, {:.4f}'.format(model.prune_strategy, model.prune_arg)
        )

if __name__ == '__main__':
    main()
