import torch
import torch.nn as nn
from torch import Tensor
from pyro.nn import PyroModule
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer, FirstLaplacianLayer,FirstCauchyLayer
from src.dgp_rff.inner_layer_nobias import SecondLayerNoBias
from src.dgp_rff.outer_layer import SingleGP, SingleCauchyGP, SingleLaplacianGP

class DeepGP(PyroModule):
    def __init__(
            self,
            in_dim_list: [1,1,1],
            out_dim_list: [1,1,1],
            J_list: [1,1,1],
            #这里不会写变量类型
    ) -> None:
        super().__init__()

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0 # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(FirstLayer(in_dim_list[i], 2 * J_list[i]))
            layer_list.append(SecondLayer(2 * J_list[i], out_dim_list[i]))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        mu = x

        return mu

class DeepGPNoBias(PyroModule):
    def __init__(
            self,
            in_dim_list: [1,1,1],
            out_dim_list: [1,1,1],
            J_list: [1,1,1],
            #这里不会写变量类型
    ) -> None:
        super().__init__()

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0 # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(FirstLayer(in_dim_list[i], 2 * J_list[i]))
            layer_list.append(SecondLayerNoBias(2 * J_list[i], out_dim_list[i]))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
        layer_list.append(PyroModule[nn.Linear](out_dim_list[-1], out_dim_list[-1], bias=True))
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        mu = x

        return mu
