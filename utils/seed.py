#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
