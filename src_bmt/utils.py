# This file contains the dictionaries `MODEL_TO_FINAL_CHECKPOINT_PATH` and `MODEL_TO_CETT`, which are used to calculate neuron activation frequencies after pre-training. 
# Author: Luo Yuqi
# Date: 2024-10
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import torch

import bmtrain as bmt
from typing import Dict

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

MODEL_TO_FINAL_CHECKPOINT_PATH = {
    '0.1b_relu' : '',
    '0.2b_relu' : '',
    '0.4b-v3_relu' : '',
    '0.8b_relu' : '',
    '1.2b_relu' : '',
    
    '0.1b_silu' : '',
    '0.2b_silu' : '',
    '0.4b-v3_silu' : '',
    '0.8b_silu' : '',
    '1.2b_silu' : '',
}
MODEL_TO_CETT = {
    '0.1b_relu' : 0,
    '0.4b-v3_relu' : 0,
    '0.2b_relu' : 0,
    '0.8b_relu' : 0,
    '1.2b_relu' : 0,

    '0.1b_silu' : 0,
    '0.2b_silu' : 0,
    '0.4b-v3_silu' : 0,
    '0.8b_silu' : 0,
    '1.2b_silu' : 0,
}
