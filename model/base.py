import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd

import torch
from torch import nn

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
