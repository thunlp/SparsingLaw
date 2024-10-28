# Sparsing Law: Towards Large Language Models with Greater Activation Sparsity

[paper](www.baidu.com)

## Citation

## Quick Links

[Overview](#overview)  
[Requirements](#requirements)  
[Experiments](#experiments)

## Overview

## Requirements

python==3.10.8

Run `pip install -r requirements.txt` to install the required Python packages.

For `CPM-Live` package, please refer to [TODO].

## Experiments

To run scripts, first copy the `*.sh` files to the `src` directory. Then, set the environment variables appropriately (e.g., `model`, `dataset`, etc.).

### Calculate sparsity of a single checkpoint

```
bash run_inspect.sh
```

### Calculate PPL-k% sparsities of all checkpoints during pre-training

```
bash calc_sparsity.sh
```

### Calculate the thresholds of each layer for evaluation

```
bash calc_threshold_for_each_layer.sh
```

### Evaluation

We use [UltraEval](https://github.com/OpenBMB/UltraEval) to evaluate the models. To support inference under PPL-k% sparsity setting during evaluation, we modify the forward code in the `FFNBlock` as shown below. The environment variable `thresholds` can be obtained by running `bash calc_threshold_for_each_layer.sh`.

```python
def forward(self, x: Tensor): 
    if os.environ.get('mode') == 'sparse' and self.threshold == None:
        with torch.no_grad():
            t = float(os.environ.get('thresholds').split(',')[self.layer_idx])
            out_norm = self.down_proj.weight.norm(dim=0)
            self.threshold = t / out_norm

    x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

    if os.environ.get('mode') == 'sparse':
        x[x.abs() < self.threshold] = 0.
    down_proj = self.down_proj(x)
    
    return down_proj
```