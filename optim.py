from typing import List

from helpers import dedup
from tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    Manages parameters, gradients, and learning rate for optimization.
    """

    def __init__(self, params: List[Tensor], lr: float):
        """
        Initialize the optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float): Learning rate.
        """
        # Set `requires_grad` to True for params if None
        for param in params:
            if param.requires_grad is None:
                param.requires_grad = True

        self.params = dedup([param for param in params if param.requires_grad])
        assert self.params, "Optimizer must have at least one parameter."

        self.device = self.params[0].device
        self.buffers = dedup(
            [param for param in params if not param.requires_grad]
        )  # Buffers are still realized
        self.lr = Tensor([lr], requires_grad=False, device=self.device).contiguous()

    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.params:
            param.grad = None

    def realize(self, extra=None):
        """
        Ensure all tensors are realized on the device.

        Args:
            extra (list, optional): Additional tensors to realize.
        """
        Tensor.corealize(
            extra + self.params + self.buffers
            if extra is not None
            else self.params + self.buffers
        )


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum, weight decay, and Nesterov acceleration.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr=0.001,
        momentum=0,
        weight_decay=0.0,
        nesterov=False,
    ):
        """
        Initialize the SGD optimizer.

        Args:
            params (list): Parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 0.001.
            momentum (float, optional): Momentum factor. Defaults to 0.
            weight_decay (float, optional): Weight decay factor. Defaults to 0.0.
            nesterov (bool, optional): Use Nesterov momentum. Defaults to False.
        """
        super().__init__(params, lr)
        self.momentum, self.wd, self.nesterov = momentum, weight_decay, nesterov
        self.b = (
            [
                Tensor.zeros(*param.shape, device=param.device, requires_grad=False)
                for param in self.params
            ]
            if self.momentum
            else []
        )

    def step(self) -> None:
        """
        Perform a single optimization step.

        Updates parameters using gradients and momentum if applicable.
        """
        for i, param in enumerate(self.params):
            assert param.grad is not None
            grad = param.grad.realize() + self.wd * param.detach()
            if self.momentum:
                self.b[i].assign(
                    self.momentum * self.b[i] + grad
                ).realize()  # First run: self.b[i] is zero, no if required
                grad = grad + self.momentum * self.b[i] if self.nesterov else self.b[i]
            param.assign(param.detach() - grad * self.lr)
        self.realize(self.b)


# LAMB is essentially just the trust ratio part of LARS applied to Adam/W.
# If the trust ratio is set to 1.0, it's equivalent to Adam/W.
def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """
    Create an AdamW optimizer.

    Args:
        params (list): Parameters to optimize.
        lr (float): Learning rate.
        b1 (float): Exponential decay rate for the first moment.
        b2 (float): Exponential decay rate for the second moment.
        eps (float): Small constant for numerical stability.
        wd (float): Weight decay factor.

    Returns:
        LAMB: An instance of the LAMB optimizer configured as AdamW.
    """
    return LAMB(params, lr, b1, b2, eps, wd, adam=True)


def Adam(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """
    Create an Adam optimizer.

    Args:
        params (list): Parameters to optimize.
        lr (float): Learning rate.
        b1 (float): Exponential decay rate for the first moment.
        b2 (float): Exponential decay rate for the second moment.
        eps (float): Small constant for numerical stability.

    Returns:
        LAMB: An instance of the LAMB optimizer configured as Adam.
    """
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


class LAMB(Optimizer):
    """
    Layer-wise Adaptive Moments optimizer with optional Adam behavior.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr=0.001,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        wd=0.0,
        adam=False,
    ):
        """
        Initialize the LAMB optimizer.

        Args:
            params (list): Parameters to optimize.
            lr (float): Learning rate.
            b1 (float): Exponential decay rate for the first moment.
            b2 (float): Exponential decay rate for the second moment.
            eps (float): Small constant for numerical stability.
            wd (float): Weight decay factor.
            adam (bool): Use Adam behavior instead of LAMB. Defaults to False.
        """
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, wd, adam
        self.t = Tensor([0], requires_grad=False).realize()
        self.m = [
            Tensor.zeros(*param.shape, device=param.device, requires_grad=False)
            for param in self.params
        ]
        self.v = [
            Tensor.zeros(*param.shape, device=param.device, requires_grad=False)
            for param in self.params
        ]

    def step(self) -> None:
        """
        Perform a single optimization step.

        Updates parameters using LAMB or Adam methodology.
        """
        self.t.assign(self.t + 1).realize()
        for i, param in enumerate(self.params):
            assert param.grad is not None
            grad = param.grad.realize()
            self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * grad).realize()
            self.v[i].assign(
                self.b2 * self.v[i] + (1.0 - self.b2) * (grad * grad)
            ).realize()

            m_hat = self.m[i] / (1.0 - self.b1**self.t)
            v_hat = self.v[i] / (1.0 - self.b2**self.t)
            update = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * param.detach()

            if not self.adam:
                r1 = param.detach().square().sum().sqrt()
                r2 = update.square().sum().sqrt()
                ratio = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
            else:
                ratio = 1.0

            param.assign(param.detach() - self.lr * ratio * update)

        self.realize([self.t] + self.m + self.v)
