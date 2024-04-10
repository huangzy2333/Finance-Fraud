import torch
import time
from torch import nn
import torch.nn.functional as F
from functools import reduce


class TensorFusion(nn.Module):

    def __init__(self, in_dimensions, out_dimension):
        super(TensorFusion, self).__init__()
        self.tensor_size = reduce(lambda x, y: x * y, in_dimensions)
        self.linear_layer = nn.Linear(self.tensor_size, out_dimension)
        self.in_dimensions = in_dimensions
        self.out_dimension = out_dimension

    def __call__(self, in_modalities):
        return self.fusion(in_modalities)

    def fusion(self, in_modalities):
        bs = in_modalities[0].shape[0]
        tensor_product = in_modalities[0]

        # calculating the tensor product

        for in_modality in in_modalities[1:]:
            tensor_product = torch.bmm(tensor_product.unsqueeze(2), in_modality.unsqueeze(1))
            tensor_product = tensor_product.view(bs, -1)

        return self.linear_layer(tensor_product)

    def forward(self, x):
        print("Not yet implemented for nn.Sequential")
        exit(-1)

