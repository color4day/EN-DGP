import torch

from torch import Tensor
from pyro.nn import PyroModule, PyroSample
from src.dgp_rff.inner_layer import FirstLayer, SecondLayer, FirstLaplacianLayer,FirstCauchyLayer
from src.dgp_rff.outer_layer import SingleGP, SingleCauchyGP, SingleLaplacianGP

class EnsembleGP(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list_RBF = [FirstLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]
        self.layers_RBF = PyroModule[torch.nn.ModuleList](layer_list_RBF)
        layer_list_Cauchy = [FirstCauchyLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]
        self.layers_Cauchy = PyroModule[torch.nn.ModuleList](layer_list_Cauchy)
        self.layers_Laplacian = SingleLaplacianGP(in_dim, out_dim, 2 * J)

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
        x_rbf = x
        x_cauchy = x
        for layer in self.layers_RBF:
            x_rbf = layer(x_rbf)
        for layer in self.layers_Cauchy:
            x_cauchy = layer(x_cauchy)
        mu = torch.cat([x_rbf, x_cauchy], dim=1)

        return mu

class FinalLayer(PyroModule):
    def __init__(
            self,
            in_dim = 1,
            out_dim = 1,
    ) -> None:

        super().__init__()#继承父类
        assert in_dim > 0 and out_dim > 0
        self.linear = PyroModule[nn.Linear](2* self.in_dim, self.out_dim, bias=False)
        self.layer.weight = PyroSample(
            dist.Normal(0., torch.tensor(1.0, device='cuda')).expand([self.out_dim, 2* self.in_dim]).to_event(2))
    def forward(
            self,
            x: Tensor
    ) -> Tensor:

        return self.linear(x)