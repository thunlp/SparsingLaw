# This file contains the implementation of the sparsified model, including functions for calculating sparsity under three settings: CETT, FAT, and Top-k. It also implements sparse inference.
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import types
import torch
import bmtrain as bmt

from typing import Dict
from torch import Tensor
from cpm.dragonfly.modeling_dragonfly import Dragonfly, DragonflyConfig, NormalLinear, DenseGatedACT


def prune_fat(x: Tensor, eps: float):
    mask = x.float().abs() < (eps + 1e-18)
    sparsity = mask.float().mean()
    x[mask] = 0.
    return x, sparsity


def prune_topk(x: Tensor, k: float):
    num_to_zero = int(x.numel() * (1 - k))

    flat_x = x.view(-1)
    flat_abs_x = flat_x.abs()
    values, indices = torch.topk(flat_abs_x, num_to_zero, largest=False)
    flat_x.scatter_(dim=0, index=indices, src=torch.zeros_like(values))

    flat_abs_x = flat_x.abs()
    bmt.print_rank('threshold = {:.3e}'.format(flat_abs_x[flat_abs_x > 1e-18].min()))
    return flat_x.view_as(x), (flat_abs_x < 1e-18).float().mean()


class SparseDragonfly(Dragonfly):
    def __init__(self, config: DragonflyConfig):
        super().__init__(config)
        self.intermedia: Dict[int, Tensor] = {}
        self.pre_intermedia: Dict[int, Tensor] = {}
        self.threshold_per_layer: Dict[int, Tensor] = {} # The values are single-value Tensors (size=1)
        self.prune_strategy = 'none'
        self.prune_arg: float = None
        self.sparse_activate = False
        self.sparsity_by_layer = {}
        self.save_intermedia = True
        self.save_pre_intermedia = False

        self.activation_count_per_neuron = torch.zeros(config.num_layers, config.dim_ff).cuda()
        self.tokens_count_all = 0

        for name, module in self.named_modules():
            # for intermedia
            if name.endswith('w_out'):
                assert(isinstance(module, NormalLinear))
                layer_id = int(name.split('.')[2])
                setattr(module, 'layer_id', layer_id)
                setattr(module, 'old_forward', module.forward)

                def new_forward(module_self: NormalLinear, x: Tensor):
                    lid = getattr(module_self, 'layer_id')
                    if self.save_intermedia:
                        self.intermedia[lid] = x.clone()
                    if self.sparse_activate and self.prune_strategy == 'cett':
                        norm_x: Tensor = x.abs() * module_self.weight.view(-1, x.size(-1)).norm(dim=0)  # bs, n, dim_ff
                        mask = norm_x < self.threshold_per_layer[lid]
                        self.sparsity_by_layer[lid] = mask.float().mean()
                        x[mask] = 0.
                    x: Tensor = getattr(module_self, 'old_forward')(x)
                    return x
                setattr(module, 'forward', types.MethodType(new_forward, module))

            # for pre_intermedia
            if name.endswith('w_in'):
                assert(isinstance(module, DenseGatedACT))
                layer_id = int(name.split('.')[2])
                setattr(module, 'layer_id', layer_id)

                def new_forward(module_self: DenseGatedACT, x: Tensor):
                    lid = getattr(module_self, 'layer_id')
                    pre_score: Tensor = module_self.w_0(x)
                    if self.save_pre_intermedia:
                        self.pre_intermedia[lid] = pre_score.clone()
                    gate_score = module_self.act(pre_score)
                    if self.sparse_activate:
                        if self.prune_strategy == 'topk':
                            gate_score, sparsity = prune_topk(gate_score, self.prune_arg)
                        elif self.prune_strategy == 'fat':
                            gate_score, sparsity = prune_fat(gate_score, self.prune_arg)
                        else:
                            sparsity = 0.
                        self.sparsity_by_layer[lid] = sparsity
                        
                    x = module_self.w_1(x)
                    x = gate_score * x
                    return x
                setattr(module, 'forward', types.MethodType(new_forward, module))

    def set_sparse_activate(self, value: bool):
        self.sparse_activate = value

    # only avaliable when self.sparse_activate == True
    def get_sparsity_after_inference(self):
        return sum(self.sparsity_by_layer[lid] for lid in range(self.config.num_layers)) / self.config.num_layers

    def calc_sparsity_zero(self, att_mask: Tensor):
        """
        calc sparsity for cett_upper_bound = 0
        """
        sparsity_per_layer = []
        state_dict = self.state_dict()
        
        with torch.no_grad():
            for lid in range(self.config.num_layers):
                weight: Tensor = state_dict[f"encoder.layers.{lid}.ffn.ffn.w_out.weight"].float().cuda()  # dim_model, dim_ff
                activate_value = self.intermedia[lid].float() # bs, n, dim_ff
                activate_value = activate_value[att_mask] # m=sum(att_mask), dim_ff
                norm: Tensor = activate_value.abs() * weight.norm(dim=0)  # m, dim_ff

                sparse_ratio = torch.sum(norm < 1e-7) / norm.numel()
                sparse_ratio = bmt.sum_loss(sparse_ratio)

                bmt.print_rank('layer: {:2d}, sparse ratio: {:.3f}'.format(lid, sparse_ratio))
                sparsity_per_layer.append(sparse_ratio)

                self.threshold_per_layer[lid] = 1e-7
        
        return bmt.sum_loss(sum(sparsity_per_layer) / len(sparsity_per_layer)).item()

    def calc_sparsity(
            self,
            cett_upper_bound: float,
            att_mask: Tensor, # bs, n
        ) -> float:
        if cett_upper_bound < 1e-9:
            return self.calc_sparsity_zero(att_mask)

        sparsity_per_layer = []
        state_dict = self.state_dict()
        with torch.no_grad():
            for lid in range(self.config.num_layers):
                weight: Tensor = state_dict[f"encoder.layers.{lid}.ffn.ffn.w_out.weight"].float().cuda()  # dim_model, dim_ff
 
                # intermedia values
                activate_value = self.intermedia[lid].float() # bs, n, dim_ff
                activate_value = activate_value[att_mask] # m=sum(att_mask), dim_ff

                # norm of each neuron's output
                norm: Tensor = activate_value.abs() * weight.norm(dim=0)  # m, dim_ff

                # generate threshold candidates
                num_thresholds = 65536
                min_value = bmt.distributed.all_reduce(norm.min(), op='min')
                max_value_this_rank = norm.view(-1)[::norm.numel() // num_thresholds].quantile(0.99)
                max_value = bmt.distributed.all_reduce(max_value_this_rank, op='max')
                thresholds = torch.linspace(min_value, max_value, int(num_thresholds)) # sum(m*dim_ff) across all gpus

                # original result of FeedForward
                x = torch.matmul(activate_value, weight.t()) # m, dim_model
                norm_x: Tensor = x.norm(dim=-1) # m,

                # binary search
                left = 0
                right = len(thresholds) - 1
                while left < right:
                    mid = (left + right) // 2
                    # calc CETT
                    t = thresholds[mid]
                    tiny_activate_value = activate_value.clone() # m, dim_ff
                    tiny_activate_value[norm > t] = 0.
                    difference = torch.matmul(tiny_activate_value, weight.t()) # m, dim_model
                    cett = torch.mean(difference.norm(dim=-1) / norm_x)
                    cett = bmt.distributed.all_reduce(cett, op='sum') / bmt.world_size()
                    
                    if cett > cett_upper_bound:
                        right = mid
                    else:
                        left = mid + 1

                t = thresholds[left - 1]
                tiny_activate_value = activate_value.clone() # m, dim_ff
                tiny_activate_value[norm > t] = 0.
                difference = torch.matmul(tiny_activate_value, weight.t()) # m, dim_model

                sparse_ratio = torch.sum(norm < t) / norm.numel()
                sparse_ratio = bmt.sum_loss(sparse_ratio)
                real_cett = torch.mean(difference.norm(dim=-1) / norm_x)
                real_cett = bmt.sum_loss(real_cett)

                self.activation_count_per_neuron[lid] += bmt.distributed.all_reduce(
                    torch.sum(norm > t, dim=0) * 1.,
                    op='sum'
                )

                bmt.print_rank('layer: {:2d}, threshold: {:.2e}, sparse ratio: {:.3f}, real cett: {:.3f}'.format(
                        lid, t, sparse_ratio, real_cett))

                sparsity_per_layer.append(sparse_ratio)
                self.threshold_per_layer[lid] = t

        self.tokens_count_all += bmt.distributed.all_reduce(
            att_mask.sum(),
            op='sum'
        ).item()
        
        return bmt.sum_loss(sum(sparsity_per_layer) / len(sparsity_per_layer)).item()
