import os
import types
import torch

from torch import Tensor
from typing import Dict, List

from model import MiniCPMForCausalLM, MiniCPMConfig
from model.modeling_minicpm import VanillaMLP

def prune_fat(x: Tensor, eps: float):
    mask = x.float().abs() < (eps + 1e-18)
    return mask


def prune_topk(x: Tensor, k: float):
    num_to_zero = int(x.numel() * (1 - k))

    flat_x = x.view(-1)
    flat_abs_x = flat_x.abs()
    values, indices = torch.topk(flat_abs_x, num_to_zero, largest=False)

    flat_x.scatter_(dim=0, index=indices, src=torch.zeros_like(values))
    flat_abs_x = flat_x.abs()
    mask = flat_abs_x < 1e-18
    return mask.view_as(x)

class SparseMiniCPMForCausalLM(MiniCPMForCausalLM):
    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.config: MiniCPMConfig

        self.sparse_activate = False
        self.prune_strategy = 'none'
        self.prune_arg: float = None
        self.neuron_thresholds_by_layer: Dict[int, Tensor] = {} # dim_ff,

        self.intermedia: Dict[int, Tensor] = {}
        self.save_intermedia = False

        self.activation_by_layer: Dict[int, Tensor] = {}
        
        for name, module in self.named_modules():
            if name.endswith('mlp'):
                assert(isinstance(module, VanillaMLP))
                layer_id = int(name.split('.')[2])
                setattr(module, 'layer_id', layer_id)

                def new_forward(mlp_self: VanillaMLP, x: Tensor): 
                    assert not mlp_self.config.pretraining_tp > 1
                    lid = getattr(mlp_self, 'layer_id')

                    gate_score = mlp_self.act_fn(mlp_self.gate_proj(x))
                    x = gate_score * mlp_self.up_proj(x)
                    if self.save_intermedia:
                        self.intermedia[lid] = x.clone().detach()

                    if self.sparse_activate:
                        if self.prune_strategy == 'pplp' or self.prune_strategy == 'cett':
                            assert lid in self.neuron_thresholds_by_layer, "Please call `set_thresholds` firstly."
                            mask = x.abs() < self.neuron_thresholds_by_layer[lid] # bs, n, dim_ff
                        elif self.prune_strategy == 'topk':
                            mask = prune_topk(gate_score, self.prune_arg)
                        elif self.prune_strategy == 'fat':
                            mask = prune_fat(gate_score, self.prune_arg)
                        else:
                            assert False

                        self.activation_by_layer[lid] = 1 - mask.float().mean(dim=-1) # bs, n
                        x[mask] = 0.

                    down_proj = mlp_self.down_proj(x)
                    return down_proj

                setattr(module, 'forward', types.MethodType(new_forward, module))
    
    def set_sparse_activate(self, value: bool):
        self.sparse_activate = value

    def get_activation_data(self) -> Tensor:
        return torch.cat(
            [self.activation_by_layer[lid].unsqueeze(0) for lid in range(self.config.num_hidden_layers)],
            dim=0
        ) # num_layers, bs, n

    def set_thresholds(self, thresholds_list: List[float]) -> None:
        assert len(thresholds_list) == self.config.num_hidden_layers

        for name, module in self.named_modules():
            if name.endswith('mlp'):
                assert(isinstance(module, VanillaMLP))
                layer_id = int(name.split('.')[2])
                out_norm = module.down_proj.weight.norm(dim=0)
                self.neuron_thresholds_by_layer[layer_id] = thresholds_list[layer_id] / out_norm

    def calc_thresholds_given_cett(
        self,
        cett: float,
        att_mask: Tensor, # bs, n
    ) -> List[float]:
        thresholds_list = []
        state_dict = self.state_dict()
        att_mask = att_mask.to(torch.bool)
        with torch.no_grad():
            for lid in range(self.config.num_hidden_layers):
                weight: Tensor = state_dict[f"model.layers.{lid}.mlp.down_proj.weight"].float().cuda()  # dim_model, dim_ff
 
                # intermedia values
                activate_value = self.intermedia[lid].float() # bs, n, dim_ff
                activate_value = activate_value[att_mask] # m=sum(att_mask), dim_ff

                # norm of each neuron's output
                norm: Tensor = activate_value.abs() * weight.norm(dim=0)  # m, dim_ff

                # generate threshold candidates
                num_thresholds = 65536
                min_value = norm.min()
                max_value = norm.view(-1)[::max(norm.numel() // num_thresholds, 1)].quantile(0.99) # drop the extreme values
                thresholds = torch.linspace(min_value, max_value, int(num_thresholds))

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
                    cett_for_this_t = torch.mean(difference.norm(dim=-1) / norm_x)
                    if cett_for_this_t > cett:
                        right = mid
                    else:
                        left = mid + 1
                thresholds_list.append(float(thresholds[left - 1]))

        return thresholds_list
