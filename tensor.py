import math
import time
from collections import defaultdict
from functools import reduce
from itertools import accumulate

import numpy as np

from helpers import DType, Dtypes, all_int, argfix, flatten, make_pair, round_up
from lazy import LazyBuffer
from ops import Device, LoadOps


class Function:
    """
    Base class for differentiable operations. Represents a computational function
    with forward and backward methods for autograd.
    """

    def __init__(self, device, *tensors):
        """
        Initialize the function context.

        Args:
            device (Device): The device where computation takes place.
            *tensors (Tensor): Input tensors to the function.
        """
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = (
            True
            if any(self.needs_input_grad)
            else None
            if None in self.needs_input_grad
            else False
        )
        if self.requires_grad:
            self.parents = tensors

    def forward(self, *args, **kwargs) -> LazyBuffer:
        """
        Perform the forward computation.

        Args:
            *args: Arguments required for forward computation.
            **kwargs: Keyword arguments required for forward computation.

        Returns:
            LazyBuffer: Result of the forward computation.
        """
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs) -> LazyBuffer:
        """
        Perform the backward computation (gradient calculation).

        Args:
            *args: Arguments required for backward computation.
            **kwargs: Keyword arguments required for backward computation.

        Returns:
            LazyBuffer: Gradient with respect to the inputs.
        """
        raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(cls, *x, **kwargs):
        """
        Apply the function to the given inputs.

        Args:
            cls (Function): The function class.
            *x (Tensor): Input tensors for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            Tensor: The result of the forward computation.
        """
        ctx = cls(x[0].device, *x)
        result = Tensor(
            ctx.forward(*[t.lazydata for t in x], **kwargs),
            device=ctx.device,
            requires_grad=ctx.requires_grad,
        )
        if ctx.requires_grad and not Tensor.no_grad:
            result._ctx = ctx  # pyright: ignore
        return result


import mlops  # noqa [E402]


class Tensor:
    """
    A core class representing a multidimensional array with support for autograd and operations.

    Attributes:
        lazydata: Internal representation of the tensor's data.
        requires_grad: Whether the tensor requires gradient computation.
        grad: The gradient of the tensor if requires_grad is True.
        _ctx: Context for autograd operations.
        training (bool): Indicates if the tensor is in training mode.
        no_grad (bool): Indicates if gradient computation is disabled.
        default_type: Default data type for tensors.
        _seed: Seed used for random number generation.
    """

    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)
    training = False
    no_grad = False
    default_type = Dtypes.float32
    _seed: int = int(time.time())

    @property
    def shape(self):
        """Returns the shape of the tensor."""
        return self.lazydata.shape

    @property
    def dtype(self):
        """Returns the data type of the tensor."""
        return self.lazydata.dtype

    @property
    def device(self):
        """Returns the device where the tensor is located."""
        return self.lazydata.device

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return len(self.shape)

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        """
        Creates a tensor with values ranging from `start` to `stop` with a step size.

        Args:
            start (float): The starting value.
            stop (float, optional): The stopping value.
            step (float): The step size between values.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor containing the range of values.
        """
        if stop is None:
            stop, start = start, 0
        return Tensor.full(
            (math.ceil((stop - start) / step),), step, **kwargs
        ).cumsum() + (start - step)

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs):
        """
        Creates a tensor with values uniformly sampled between `low` and `high`.

        Args:
            *shape: Shape of the tensor.
            low (float): Minimum value of the uniform distribution.
            high (float): Maximum value of the uniform distribution.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor with uniformly distributed values.
        """
        dtype = kwargs.pop("dtype", Tensor.default_type)
        return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    @staticmethod
    def stack(tensors, dim=0):
        """
        Stacks a sequence of tensors along a new dimension.

        Args:
            tensors (list[Tensor]): List of tensors to stack.
            dim (int): Dimension along which to stack the tensors.

        Returns:
            Tensor: A stacked tensor.
        """
        first = tensors[0].unsqueeze(dim)
        unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
        return first.cat(*unsqueezed_tensors, dim=dim)

    @staticmethod
    def _loadop(op, sz, device=None, dtype=None, arg=None, **kwargs):
        """
        Internal helper to load an operation into a tensor.

        Args:
            op: Operation to perform.
            sz (int): Size of the tensor.
            device: Device where the tensor will reside.
            dtype: Data type of the tensor.
            arg: Argument for the operation.
            **kwargs: Additional arguments.

        Returns:
            Tensor: Resulting tensor after applying the operation.
        """
        assert isinstance(sz, int), f"Cannot create with symbolic size {sz}"
        return Tensor(
            LazyBuffer.loadop(
                op, (sz,), Tensor.default_type if dtype is None else dtype, arg
            ),
            dtype=dtype,
            device=device,
            **kwargs,
        )

    @staticmethod
    def rand(*shape, **kwargs):
        """
        Creates a tensor with values sampled from a random uniform distribution.

        Args:
            *shape: Shape of the tensor.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor with random values.
        """
        Tensor._seed += 1
        return Tensor._loadop(
            LoadOps.RAND,
            math.prod((shape := argfix(*shape))),
            arg=Tensor._seed,
            **kwargs,
        ).reshape(shape)

    @staticmethod
    def scaled_uniform(*shape, **kwargs):
        """
        Creates a tensor with values scaled uniformly around zero.

        Args:
            *shape: Shape of the tensor.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor with scaled uniform values.
        """
        return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(
            math.prod(shape) ** -0.5
        )

    @staticmethod
    def corealize(lst):
        """
        Ensures all tensors in the list are realized in memory.

        Args:
            lst (list[Tensor]): List of tensors to realize.
        """
        seen = set()
        sched = []
        for t in lst:
            sched += t.lazydata.schedule(seen)

    @staticmethod
    def zeros(*shape, **kwargs):
        """
        Creates a tensor filled with zeros.

        Args:
            *shape: Shape of the tensor.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        return Tensor.full(argfix(*shape), 0, **kwargs)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        """
        Creates a tensor filled with a specific value.

        Args:
            shape (tuple): Shape of the tensor.
            fill_value: The value to fill the tensor with.
            **kwargs: Additional arguments for tensor creation.

        Returns:
            Tensor: A tensor filled with the specified value.
        """
        return (
            Tensor(fill_value, **kwargs)
            .reshape([1] * len(new_shape := argfix(shape)))
            .expand(new_shape)
        )

    def __hash__(self):
        return id(self)

    def __neg__(self):
        return self.neg()

    def __add__(self, x):
        return self.add(x)

    def __sub__(self, x):
        return self.sub(x)

    def __mul__(self, x):
        return self.mul(x)

    def __pow__(self, x):
        return self.pow(x)

    def __truediv__(self, x):
        return self.div(x)

    def __matmul__(self, x):
        return self.matmul(x)

    def __radd__(self, x):
        return self.add(x, True)

    def __rsub__(self, x):
        return self.sub(x, True)

    def __rmul__(self, x):
        return self.mul(x, True)

    def __rpow__(self, x):
        return self.pow(x, True)

    def __rtruediv__(self, x):
        return self.div(x, True)

    def __rmatmul__(self, x):
        return self.matmul(x, True)

    def __lt__(self, x):
        return mlops.Less.apply(*self._broadcasted(x, False))

    def __gt__(self, x):
        return mlops.Less.apply(*self._broadcasted(x, True))

    def __ge__(self, x):
        return 1.0 - (self < x)

    def __le__(self, x):
        return 1.0 - (self > x)

    def __ne__(self, x):
        return (self < x) + (self > x)

    def __eq__(self, x):
        return 1.0 - (self != x)

    def _to_float(self, x):
        return x

    def __getitem__(self, val):
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz:
                return e if e != -1 else dim_sz - 1
            raise IndexError(
                f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}"
            )

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)

        # Categorize slices by type (int, slice, Tensor, Ellipsis, etc.)
        for i, v in enumerate(orig_slices):
            count[type(v)].append(i)

        # Validate the number of slices and ellipses
        if (
            num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])
        ) > len(self.shape):
            raise IndexError(
                f"too many indices for tensor of dimension {len(self.shape)}"
            )
        if len(ellipsis_found := count[type(Ellipsis)]) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # Replace ellipsis with appropriate number of slices
        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
            len(self.shape) - num_slices
        )

        # Normalize slices and integers
        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [
            v
            if isinstance(v, slice)
            else slice(y_ := normalize_int(v, i, dim_sz), y_ + 1)
            if isinstance(v, int)
            else slice(None)
            for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
        ]

        # Compute slice start, stop, and strides
        start, stop, strides = (
            zip(*y)
            if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)])
            else ((), (), ())
        )

        # Adjust slices for shrinking and flipping
        new_slice = tuple(
            ((0, 0) if e < s else (s, e))
            if st > 0
            else ((0, 0) if e > s else (e + 1, s + 1))
            for s, e, st in zip(start, stop, strides)
        )
        sliced_tensor = self.shrink(new_slice).flip(
            axis=[i for i, s in enumerate(strides) if s < 0]
        )
        new_shape = sliced_tensor.shape

        # Handle strides not equal to 1 (e.g., strided slicing)
        if any(abs(s) != 1 for s in strides):
            strides = tuple(abs(s) for s in strides)
            padded_tensor = sliced_tensor.pad(
                tuple(
                    (0, s - (dim_sz % s) if dim_sz % s != 0 else 0)
                    for s, dim_sz in zip(strides, sliced_tensor.shape)
                )
            )
            reshaped_tensor = padded_tensor.reshape(
                flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides))
            )
            new_shape = reshaped_tensor.shape[::2]
            sliced_tensor = reshaped_tensor.shrink(
                tuple(flatten(((0, sh), (0, 1)) for sh in new_shape))
            )

        # Recalculate the final shape and handle new axes
        final_shape, it_shape, dim, tensors, dim_collapsed = (
            [],
            iter(new_shape),
            [],
            [],
            0,
        )
        for i, s in enumerate(orig_slices):
            if s is None:
                final_shape.append(1)
            else:  # s is int, slice, or Tensor
                dim_shape = next(it_shape)
                if isinstance(s, int):
                    dim_collapsed += 1
                else:
                    assert isinstance(
                        dim_shape, int
                    ), f"does not support symbolic shape {dim_shape}"
                    final_shape.append(dim_shape)
                    if isinstance(s, Tensor):
                        tensors.append(s)
                        dim.append(i - dim_collapsed)
        ret = sliced_tensor.reshape(tuple(final_shape))

        # Handle fancy indexing (using tensors)
        if tensors:
            idx = [
                t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t
                for d, t in zip(dim, tensors)
            ]
            max_dim = max(i.ndim for i in idx)
            sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]
            arange = [
                Tensor.arange(
                    ret.shape[d],
                    dtype=Dtypes.int32,
                    requires_grad=False,
                    device=self.device,
                ).reshape(
                    *[1] * sd, ret.shape[d], *[1] * (ret.ndim + max_dim - n - sd - 1)
                )
                for n, (sd, d) in enumerate(zip(sum_dim, dim))
            ]
            first_idx = [
                idx[0].reshape(
                    *[1] * dim[0],
                    *[1] * (1 + max_dim - idx[0].ndim),
                    *idx[0].shape,
                    *[1] * (ret.ndim - dim[0] - 1),
                )
            ]
            rest_idx = [
                i.reshape(
                    *[1] * dim[0],
                    *[1] * (max_dim - i.ndim),
                    *i.shape,
                    *[1] * (ret.ndim - dim[0] - n),
                )
                for n, i in enumerate(idx[1:], 1)
            ]
            idx = first_idx + rest_idx
            ret = ret.reshape(
                *ret.shape[: sum_dim[0] + 1],
                *[1] * max_dim,
                *ret.shape[sum_dim[0] + 1 :],
            )
            for a, i, sd in zip(arange, idx, sum_dim):
                ret = (a == i).mul(ret).sum(sd)

            # Handle non-contiguous dimensions after fancy indexing
            if (
                dim[0] != 0
                and len(dim) != 1
                and dim != list(range(dim[0], dim[-1] + 1))
            ):
                ret_dims = list(range(ret.ndim))
                ret = ret.permute(
                    ret_dims[dim[0] : dim[0] + max_dim]
                    + ret_dims[: dim[0]]
                    + ret_dims[dim[0] + max_dim :]
                )
        return ret

    def __setitem__(self, s, v):
        return self.__getitem__(s).assign(v)

    def __init__(self, data, device=None, dtype=None, requires_grad=None):
        """
        Initialize a Tensor object.

        Args:
            data: The input data for the tensor. It can be one of the following:
                - LazyBuffer: A buffer representing tensor data.
                - int, float: Scalar values to be converted into tensors.
                - list: A list of values to be converted into a tensor.
                - np.ndarray: A NumPy array to be converted into a tensor.
                - bytes: A byte array to be converted into a tensor.
                - None: An empty tensor.
            device (optional): The device where the tensor will be allocated.
            dtype (optional): The data type of the tensor.
            requires_grad (optional): Whether the tensor requires gradient computation.

        Raises:
            RuntimeError: If the input data cannot be converted into a tensor.
            AssertionError: If dtype is invalid or incompatible with the input data.
        """
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
        device = Device.canonicalize()
        self.grad = None
        self.requires_grad = requires_grad  # Can be True, False, or None
        self._ctx = None  # Internal variable for autograd graph construction

        if isinstance(data, LazyBuffer):
            assert (
                dtype is None or dtype == data.dtype
            ), "dtype doesn't match, and casting isn't supported"
        elif isinstance(data, (int, float)):
            data = LazyBuffer.loadop(
                LoadOps.CONST, tuple(), dtype or Tensor.default_type, data
            )
        elif data is None or data.__class__ is list:
            assert (
                dtype is None or dtype.np is not None
            ), f"{dtype} doesn't have a numpy dtype"
            data = LazyBuffer.fromCPU(
                np.array(
                    [] if data is None else data,
                    dtype=(dtype or Tensor.default_type).np,
                )
            )
        elif isinstance(data, bytes):
            data = LazyBuffer.fromCPU(np.frombuffer(data, np.uint8))
        elif isinstance(data, np.ndarray):
            assert (
                dtype is None or dtype.np is not None
            ), f"{dtype} doesn't have a numpy dtype"
            if data.shape == ():
                data = LazyBuffer.loadop(
                    LoadOps.CONST,
                    tuple(),
                    dtype or Dtypes.from_np(data.dtype),
                    data.item(),
                )
            else:
                data = LazyBuffer.fromCPU(
                    data.astype(dtype.np)
                    if dtype is not None and dtype.np is not None
                    else data
                )

        if not isinstance(data, LazyBuffer):
            raise RuntimeError(
                f"can't create Tensor from {data!r} with type {type(data)}"
            )
        self.lazydata = data if data.device == device else data.copy_to_device(device)

    def detach(self):
        """
        Create a new tensor that shares the same data but does not track gradients.

        Returns:
            Tensor: A detached tensor with `requires_grad=False`.
        """
        return Tensor(self.lazydata, device=self.device, requires_grad=False)

    def cast(self, dtype: DType):
        """
        Cast the tensor to a specified data type.

        Args:
            dtype (DType): The target data type.

        Returns:
            Tensor: A tensor with the specified data type.
        """
        return mlops.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self

    def numpy(self) -> np.ndarray:
        """
        Convert the tensor into a NumPy array.

        Returns:
            np.ndarray: The tensor represented as a NumPy array.

        Raises:
            AssertionError: If the tensor has symbolic shape or unsupported dtype.
        """
        assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}"
        assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
        return (
            self.detach()
            .cast(Dtypes.from_np(self.dtype.np))
            .contiguous()
            .to("CPU")
            .realize()
            .lazydata.realized.toCPU()
            .reshape(self.shape)
        )

    def to(self, device):
        """
        Move the tensor to a specified device.

        Args:
            device: The target device.

        Returns:
            Tensor: A tensor on the specified device.
        """
        if device is None or device == self.device:
            return self
        ret = Tensor(self.lazydata, device)
        if self.grad:
            ret.grad = self.grad.to(device)
        return ret

    def realize(self):
        """
        Realize the tensor by finalizing its lazy computation graph.

        Returns:
            Tensor: The realized tensor.
        """
        self.lazydata.schedule()
        return self

    def assign(self, x):
        """
        Assign the values of another tensor to this tensor.

        Args:
            x (Tensor or compatible): The tensor or data to assign.

        Returns:
            Tensor: The tensor with updated values.

        Raises:
            AssertionError: If the shapes, devices, or data types are mismatched.
        """
        if x.__class__ is not Tensor:
            x = Tensor(x, device=self.device, dtype=self.dtype)
        assert (
            self.shape == x.shape and self.device == x.device
        ), f"assign shape mismatch {self.shape} != {x.shape} or device mismatch {self.device} != {x.device}"
        assert not x.requires_grad  # self.requires_grad is okay
        if self.dtype == x.dtype and self.lazydata.realized is not None:
            x.lazydata.output_buffer = self.lazydata.realized  # pyright: ignore
        self.lazydata = x.lazydata
        return self

    def full_like(self, fill_value, **kwargs):
        """
        Create a tensor with the same shape as the current tensor, filled with a specified value.

        Args:
            fill_value: The value to fill the tensor with.
            **kwargs: Additional parameters such as `dtype` or `device`.

        Returns:
            Tensor: A tensor with the same shape and filled with `fill_value`.
        """
        return Tensor.full(
            self.shape,
            fill_value=fill_value,
            dtype=kwargs.pop("dtype", self.dtype),
            device=kwargs.pop("device", self.device),
            **kwargs,
        )

    def ones_like(self, **kwargs):
        """
        Create a tensor of ones with the same shape and other attributes as the current tensor.

        Args:
            **kwargs: Additional attributes such as dtype or device.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return self.full_like(1, **kwargs)

    def contiguous(self):
        """
        Ensure the tensor is stored in a contiguous memory layout.

        Returns:
            Tensor: The tensor in a contiguous memory layout.
        """
        return mlops.Contiguous.apply(self)

    def contiguous_backward(self):
        """
        Perform a backward operation for a contiguous tensor.

        Returns:
            Tensor: The result of the backward operation.
        """
        return mlops.ContiguousBackward.apply(self)

    def reshape(self, shape, *args):
        """
        Reshape the tensor into the specified shape.

        Args:
            shape: The target shape.
            *args: Additional shape dimensions.

        Returns:
            Tensor: The reshaped tensor.

        Raises:
            ValueError: If the total number of elements cannot match the new shape.
        """
        new_shape = argfix(shape, *args)
        return mlops.Reshape.apply(
            self,
            shape=tuple(
                [
                    -math.prod(self.shape) // math.prod(new_shape)
                    if s == -1
                    else (s if s is not None else self.shape[i])
                    for i, s in enumerate(new_shape)
                ]
            ),
        )

    def _broadcasted(self, y, reverse: bool = False):
        """
        Broadcast two tensors to a common shape.

        Args:
            y: The tensor or scalar to broadcast.
            reverse (bool, optional): If True, swap the order of self and y.

        Returns:
            Tuple[Tensor, Tensor]: The broadcasted tensors.
        """
        x: Tensor = self
        if not isinstance(y, Tensor):
            if 0 in x.shape:
                return x, x.full_like(y)
            y = Tensor(
                y,
                device=self.device,
                requires_grad=False,
                dtype=self.dtype if self.dtype != Dtypes.bool else Dtypes.float32,
            )
        if reverse:
            x, y = y, x
        if (xshape := x.shape) == (yshape := y.shape):
            return (x, y)

        shape_delta = len(xshape) - len(yshape)
        if shape_delta > 0:
            y = y.reshape((1,) * shape_delta + yshape)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + xshape)
        if (xshape := x.shape) == (yshape := y.shape):
            return (x, y)

        shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
        if xshape != shape_ret:
            x = x.expand(shape_ret)
        if yshape != shape_ret:
            y = y.expand(shape_ret)
        return (x, y)

    def maximum(self, x):
        """
        Compute the element-wise maximum of two tensors.

        Args:
            x: The tensor or scalar to compare.

        Returns:
            Tensor: A tensor with the element-wise maximum values.
        """
        return (
            (self < x)
            .detach()
            .where(x, (self > x).detach().where(self, (self + x) / 2))
        )

    def minimum(self, x):
        """
        Compute the element-wise minimum of two tensors.

        Args:
            x: The tensor or scalar to compare.

        Returns:
            Tensor: A tensor with the element-wise minimum values.
        """
        return -((-self).maximum(-x))

    def where(self, input, other):
        """
        Select elements from `input` or `other` based on the condition.

        Args:
            input: The tensor for true conditions.
            other: The tensor for false conditions.

        Returns:
            Tensor: A tensor with selected elements.
        """
        x_, y = self._broadcasted(input)
        x, z = x_._broadcasted(other)
        return mlops.Where.apply(x, *y._broadcasted(z))

    def flip(self, axis, *args):
        """
        Reverse the order of elements along the specified axes.

        Args:
            axis: The axes to reverse.
            *args: Additional axes.

        Returns:
            Tensor: A tensor with reversed elements along the specified axes.
        """
        return mlops.Flip.apply(
            self,
            axis=[x if x >= 0 else x + len(self.shape) for x in argfix(axis, *args)],
        )

    def permute(self, order, *args):
        """
        Permute the dimensions of the tensor.

        Args:
            order: The desired order of dimensions.
            *args: Additional dimension orders.

        Returns:
            Tensor: A tensor with permuted dimensions.
        """
        return mlops.Permute.apply(self, order=argfix(order, *args))

    def pad(self, arg, value=0.0):
        """
        Pad the tensor along specified dimensions.

        Args:
            arg: The padding configuration for each dimension.
            value (float, optional): The value to pad with.

        Returns:
            Tensor: A padded tensor.
        """
        if all(x is None or x == (0, 0) for x in arg):
            return self
        ret = mlops.Pad.apply(
            self, arg=(narg := tuple(x if x is not None else (0, 0) for x in arg))
        )
        return (
            ret
            if 0 == value
            else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)
        )

    def mul(self, x, reverse=False):
        """
        Multiply the tensor with another tensor or scalar.

        Args:
            x: The tensor or scalar to multiply with.
            reverse (bool, optional): If True, reverse the order of multiplication.

        Returns:
            Tensor: The result of the multiplication.
        """
        x = self._to_float(x)
        if x.__class__ is not Tensor and x == 0.0:
            return mlops.Zero.apply(self)
        if x.__class__ is not Tensor and x == -1.0:
            return -self
        return (
            mlops.Mul.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor or x != 1.0
            else self
        )

    def sub(self, x, reverse=False):
        """
        Subtract another tensor or scalar from this tensor.

        Args:
            x: The tensor or scalar to subtract.
            reverse (bool, optional): If True, reverse the order of subtraction.

        Returns:
            Tensor: The result of the subtraction.
        """
        x = self._to_float(x)
        return (
            mlops.Sub.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor or x
            else (-self if reverse else self)
        )

    def div(self, x, reverse=False):
        """
        Divide this tensor by another tensor or scalar.

        Args:
            x: The tensor or scalar to divide by.
            reverse (bool, optional): If True, reverse the order of division.

        Returns:
            Tensor: The result of the division.
        """
        x = self._to_float(x)
        return (
            mlops.Div.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor
            or reverse
            or not x
            or not Dtypes.is_float(self.dtype)
            else self.mul(1 / x)
        )

    def sqrt(self):
        """
        Compute the element-wise square root of the tensor.

        Returns:
            Tensor: A tensor with the square root of each element.
        """
        return mlops.Sqrt.apply(self)

    def exp(self):
        """
        Compute the element-wise exponential (e^x) of the tensor.

        Returns:
            Tensor: A tensor with the exponential of each element.
        """
        return mlops.Exp.apply(self)

    def reciprocal(self):
        """
        Compute the element-wise reciprocal (1/x) of the tensor.

        Returns:
            Tensor: A tensor with the reciprocal of each element.
        """
        return 1.0 / self

    def sin(self):
        """
        Compute the element-wise sine of the tensor.

        Returns:
            Tensor: A tensor with the sine of each element.
        """
        return mlops.Sin.apply(self)

    def sign(self):
        """
        Compute the element-wise sign of the tensor (-1, 0, 1).

        Returns:
            Tensor: A tensor indicating the sign of each element.
        """
        return self / (self.abs() + 1e-10)

    def relu(self):
        """
        Apply the ReLU (Rectified Linear Unit) activation function element-wise.

        Returns:
            Tensor: A tensor with ReLU applied to each element.
        """
        return mlops.Relu.apply(self)

    def abs(self):
        """
        Compute the element-wise absolute value of the tensor.

        Returns:
            Tensor: A tensor with the absolute values of each element.
        """
        return self.relu() + (-self).relu()

    def log(self):
        """
        Compute the element-wise natural logarithm (log base e) of the tensor.

        Returns:
            Tensor: A tensor with the logarithm of each element.
        """
        return mlops.Log.apply(self)

    def cos(self):
        """
        Compute the element-wise cosine of the tensor.

        Returns:
            Tensor: A tensor with the cosine of each element.
        """
        return ((math.pi / 2) - self).sin()

    def clip(self, min_, max_):
        """
        Clip the tensor values to a specified range [min_, max_].

        Args:
            min_ (float): The minimum value.
            max_ (float): The maximum value.

        Returns:
            Tensor: A tensor with clipped values.
        """
        return self.maximum(min_).minimum(max_)

    def trunc(self):
        """
        Truncate the tensor values to their integer parts.

        Returns:
            Tensor: A tensor with truncated values.
        """
        return self.cast(Dtypes.int32).contiguous().cast(self.dtype)

    def matmul(self, x, reverse=False):
        """
        Perform matrix multiplication with another tensor.

        Args:
            x: The tensor to multiply with.
            reverse (bool, optional): If True, reverse the order of multiplication.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        return x.dot(self) if reverse else self.dot(x)

    def neg(self):
        """
        Compute the element-wise negation of the tensor.

        Returns:
            Tensor: A tensor with negated values.
        """
        return mlops.Neg.apply(self)

    def add(self, x, reverse=False):
        """
        Add another tensor or scalar to this tensor.

        Args:
            x: The tensor or scalar to add.
            reverse (bool, optional): If True, reverse the order of addition.

        Returns:
            Tensor: The result of the addition.
        """
        x = self._to_float(x)
        return (
            mlops.Add.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor or x
            else self
        )

    def expand(self, shape, *args):
        """
        Expand the tensor to a specified shape.

        Args:
            shape: The target shape.
            *args: Additional shape dimensions.

        Returns:
            Tensor: The expanded tensor.
        """
        return mlops.Expand.apply(
            self,
            shape=tuple(
                [x if x != -1 else s for s, x in zip(self.shape, argfix(shape, *args))]
            ),
        )

    def shrink(self, arg):
        """
        Reduce the size of the tensor by removing padding.

        Args:
            arg: The amount of shrinkage for each dimension.

        Returns:
            Tensor: The shrunk tensor.
        """
        return (
            mlops.Shrink.apply(
                self,
                arg=tuple(
                    x if x is not None else (0, s) for x, s in zip(arg, self.shape)
                ),
            )
            if any(x is not None and x != (0, s) for x, s in zip(arg, self.shape))
            else self
        )

    def slice(self, arg, value: float = 0):
        """
        Slice the tensor along specified dimensions, optionally padding with a value.

        Args:
            arg: The slicing configuration for each dimension.
            value (float, optional): The value to pad with.

        Returns:
            Tensor: The sliced tensor.
        """
        arg_ = tuple([a if a is not None else (0, s) for s, a in zip(self.shape, arg)])
        padding = tuple(
            [(max(0, -p[0]), max(0, p[1] - self.shape[i])) for i, p in enumerate(arg_)]
        )
        return self.pad(padding, value=value).shrink(
            tuple(
                [
                    (p[0] + padding[i][0], p[1] + padding[i][0])
                    for i, p in enumerate(arg_)
                ]
            )
        )

    def transpose(self, ax1=1, ax2=0):
        """
        Transpose two dimensions of the tensor.

        Args:
            ax1 (int): The first axis to transpose.
            ax2 (int): The second axis to transpose.

        Returns:
            Tensor: The transposed tensor.
        """
        order = list(range(len(self.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    def pad2d(self, padding, value=0):
        """
        Apply 2D padding to the tensor.

        Args:
            padding: The padding configuration for each dimension.
            value (float, optional): The value to pad with.

        Returns:
            Tensor: The padded tensor.
        """
        slc = [
            (-p0, s + p1)
            for p0, p1, s in zip(padding[::2], padding[1::2], self.shape[::-1])
        ][::-1]
        return self.slice(
            [(0, s) for s in self.shape[: -(len(padding) // 2)]] + slc, value=value
        )

    def pow(self, x, reverse=False):
        """
        Compute the element-wise power of the tensor raised to the given exponent.

        This function handles various special cases, such as integer powers (e.g., 2.0, 3.0),
        square roots (for 0.5), and reciprocal powers for negative exponents. It also handles
        cases for negative bases and fractional exponents, ensuring proper sign handling and
        preventing NaNs for invalid operations (e.g., negative base with a non-integer power).

        Args:
            x (Tensor or float): The exponent to which each element of the tensor is raised.
            reverse (bool, optional): If True, computes the power using the formula `x^self` instead of `self^x`.

        Returns:
            Tensor: A tensor containing the result of the element-wise power operation.
        """
        x = self._to_float(x)

        if x.__class__ is not Tensor and not reverse:
            # Handling simple power identities for common exponents
            if x < 0:
                return self.reciprocal().pow(-x)
            if x == 3.0:
                return self * self * self
            if x == 2.0:
                return self * self
            if x == 1.0:
                return self
            if x == 0.5:
                return self.sqrt()

        if not isinstance(x, Tensor) and reverse and x > 0:
            return self.mul(math.log(x)).exp()

        ar = (
            self.abs().log().mul(x).exp()
            if not reverse or isinstance(x, Tensor)
            else self.mul(math.log(abs(x))).exp()
        )

        # Correcting sign for negative bases raised to a power
        sign = (
            (x * math.pi).cos()
            if isinstance(x, Tensor)
            else math.cos(x * math.pi)
            if not reverse
            else (self * math.pi).cos()
        )

        base_sign = (
            (
                self.sign()
                if not reverse
                else x.sign()
                if isinstance(x, Tensor)
                else math.copysign(1, x)
            )
            - 1
        ) / -2

        # Adjust base_sign if the base is 0
        base_sign = base_sign - (
            1.5
            * (
                1
                - (
                    self.sign().abs()
                    if not reverse
                    else x.sign().abs()
                    if isinstance(x, Tensor)
                    else abs(int(bool(x)))
                )
            )
        )

        # Handle NaN injection for negative bases with non-integer powers
        to_nan = (
            ((x - x.trunc()) * 1e10).abs().clip(0, 1)
            if isinstance(x, Tensor)
            else int(bool(x - int(x)))
            if not reverse
            else ((self - self.trunc()) * 1e10).abs().clip(0, 1)
        ) * base_sign

        inject_nan = (
            (((-to_nan) * 2) + 1).log().add(1)
            if isinstance(to_nan, Tensor)
            else 1
            if not to_nan
            else float("nan")
        )

        return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)

    def conv2d(
        self,
        weight,
        bias=None,
        groups=1,
        stride=1,
        dilation=1,
        padding=0,
    ):
        """
        Performs a 2D convolution operation on the input tensor with the given weights.

        This method supports both standard 2D convolution and Winograd's minimal filtering algorithm
        for optimized convolution when using 3x3 filters. It includes functionality for handling
        padding, stride, dilation, and multiple groups in the convolution. The method also supports
        the use of bias in the convolution operation.

        Args:
            weight (Tensor): The weight tensor with shape `(cout, cin, H, W)`, where `cout` is the
                             number of output channels, `cin` is the number of input channels, and
                             `H` and `W` are the height and width of the convolution filter.
            bias (Tensor, optional): The bias tensor to be added to the output, with shape `(cout)`. Default is None.
            groups (int, optional): The number of groups to split the input channels into. Default is 1.
            stride (int or tuple of int, optional): The stride of the convolution. Default is 1.
            dilation (int or tuple of int, optional): The dilation factor of the convolution. Default is 1.
            padding (int, tuple of int, or list of int, optional): The padding to be applied to the input tensor.
                                  If a tuple or list is provided, it should match the dimensions of the kernel.
                                  Default is 0.

        Returns:
            Tensor: The result of the convolution operation, which will have the shape
                    `(bs, cout, oy, ox)` where `bs` is the batch size, `cout` is the number of output channels,
                    and `oy`, `ox` are the output height and width.
        """
        # Extract batch size (bs) and input channel count (cin_) from the input tensor's shape
        (bs, cin_), (cout, cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]

        # Validate the input tensor and weight shapes, ensuring the number of input channels matches
        assert (
            groups * cin == cin_ and len(self.shape) == len(weight.shape)
        ), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"

        # Handle padding argument, making sure it's appropriately shaped
        if isinstance(padding, (tuple, list)):
            assert (
                len(padding) == 2 * len(HW) or len(padding) == len(HW)
            ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"

        # Set padding to appropriate form (list of 2*len(HW) values)
        padding_ = (
            [padding] * 2 * len(HW)
            if isinstance(padding, int)
            else (
                padding
                if len(padding) == 2 * len(HW)
                else [p for p in padding for _ in range(2)][  # pyright: ignore
                    ::-1
                ]  # Adjust padding order
            )
        )

        # Apply padding and pooling operation to the input tensor
        x = self.pad2d(padding_)._pool(
            HW, stride, dilation
        )  # (bs, groups*cin, oy, ox, H, W)

        # Split the output channels by the number of groups
        rcout, oyx = cout // groups, x.shape[2 : -len(HW)]

        # If not using 3x3 filters with stride 1 and dilation 1, use standard convolution
        if not all(x == 3 for x in HW) or stride != 1 or dilation != 1:
            # Reshape the tensor for broadcasting during convolution
            x = (
                x.reshape(bs, groups, cin, 1, *oyx, *HW)
                .expand(bs, groups, cin, rcout, *oyx, *HW)
                .permute(
                    0,
                    1,
                    3,
                    *[4 + i for i in range(len(oyx))],
                    2,
                    *[4 + len(oyx) + i for i in range(len(HW))],
                )
            )

            # Perform convolution operation with the reshaped tensor and weights
            ret = (
                (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))
                .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
                .reshape(bs, cout, *oyx)
            )
            return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

        # If using 3x3 filter and Winograd optimization, proceed with the Winograd minimal filtering algorithm
        def apply_matrix(mat, t, dim=0):
            """
            Applies a matrix transformation recursively to a tensor `t` along the specified dimension.
            """
            return (
                t
                if dim == len(HW)
                else Tensor.stack(
                    [
                        apply_matrix(
                            mat,
                            sum(mm * t[j] for j, mm in enumerate(m) if mm),
                            dim=dim + 1,
                        )
                        for m in mat
                    ]
                )
            )

        # Define Winograd transformation matrices for the 3x3 filter case
        HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
        winograd_Bt = [
            [4, 0, -5, 0, 1, 0],
            [0, -4, -4, 1, 1, 0],
            [0, 4, -4, -1, 1, 0],
            [0, -2, -1, 2, 1, 0],
            [0, 2, -1, -2, 1, 0],
            [0, 4, 0, -5, 0, 1],
        ]
        winograd_G = [
            [1 / 4, 0, 0],
            [-1 / 6, -1 / 6, -1 / 6],
            [-1 / 6, 1 / 6, -1 / 6],
            [1 / 24, 1 / 12, 1 / 6],
            [1 / 24, -1 / 12, 1 / 6],
            [0, 0, 1],
        ]
        winograd_At = [
            [1, 1, 1, 1, 1, 0],
            [0, 1, -1, 2, -2, 0],
            [0, 1, 1, 4, 4, 0],
            [0, 1, -1, 8, -8, 1],
        ]  # applying At in pre-order almost doubles compilation time

        # Apply padding to adjust to 4x4 output tiles (Winograd tiles)
        d = self.pad2d(
            sum(
                [
                    [
                        padding_[i * 2],
                        padding_[i * 2 + 1]
                        + (-(dim + sum(padding_[i * 2 : (i + 1) * 2]) - 2) % 4),
                    ]
                    for i, dim in enumerate(self.shape[-len(HW) :])
                ],
                [],
            )
        )._pool(HWI, HWO)  # (bs, cin_, tyx, HWI) # pyright: ignore
        d = d.permute(
            *range(len(d.shape) - len(HW), len(d.shape)), *range(len(d.shape) - len(HW))
        ).contiguous_backward()  # move HW to the front: # (HWI, bs, cin_, tyx)
        tyx = d.shape[-len(HWI) :]  # dim of tiling

        # Reshape the weight tensor for Winograd transformation
        g = weight.permute(
            *range(len(weight.shape) - len(HW), len(weight.shape)),
            *range(len(weight.shape) - len(HW)),
        )  # move HW to the front

        # Compute the Winograd tiles for the transformation
        gfactors = (
            apply_matrix(winograd_G, g)
            .contiguous()
            .reshape(*HWI, 1, groups, rcout, cin, *([1] * len(tyx)))
        )  # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
        dfactors = (
            apply_matrix(winograd_Bt, d)
            .contiguous()
            .reshape(*HWI, bs, groups, 1, cin, *tyx)
        )  # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)

        # Compute the final result using the transformation matrices
        ret = apply_matrix(
            winograd_At, (gfactors * dfactors).sum(axis=-1 - len(HW))
        )  # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)

        # Reshape and merge the results
        ret = ret.permute(
            [
                *range(len(HW), len(ret.shape) - len(HW)),
                *[i + o for i in range(len(HW)) for o in [len(ret.shape) - len(HW), 0]],
            ]
        )  # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
        ret = ret.reshape(
            bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]
        ).shrink(
            tuple((0, s) for s in [bs, cout, *oyx])
        )  # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final

        # Return the result with bias if provided
        return (
            (
                ret
                if bias is None
                else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))
            )
            .contiguous()
            .contiguous_backward()
        )

    def _pool(
        self,
        k_,
        stride=1,
        dilation=1,
    ):
        """
        Performs pooling operation on the input tensor with the given kernel size, stride, and dilation.

        The method handles both standard pooling and pooling with dilation. It reshapes the input tensor
        based on the given kernel size `k_`, stride, and dilation. It also manages cases where the stride
        and dilation are greater than 1, adjusting the output shape accordingly.

        Args:
            k_ (tuple or list): The size of the pooling kernel (e.g., `(height, width)`).
            stride (int or tuple, optional): The stride of the pooling operation. Default is 1.
            dilation (int or tuple, optional): The dilation factor for the pooling operation. Default is 1.

        Returns:
            Tensor: The result of the pooling operation after reshaping and adjusting with the specified parameters.
        """
        assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
        assert all_int(self.shape) and all_int(
            k_
        ), f"does not support symbolic {self.shape=}, {k_=}"

        # Create stride and dilation pairs based on the kernel size length
        s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))

        # Ensure the kernel size, stride, and dilation match in length
        assert len(k_) == len(s_) and len(k_) == len(
            d_
        ), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"

        # Prefix to handle the leading dimensions of the input shape
        slc_prefix, prefix, i_ = (
            [(0, x) for x in self.shape[0 : -len(k_)]],  # Handle leading dimensions
            self.shape[0 : -len(k_)],  # Shape excluding kernel dimensions
            self.shape[-len(k_) :],  # Kernel-related dimensions
        )

        # If kernel size exceeds stride or dilation is not equal to 1
        if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
            # Compute the output shape and the expansion required for padding
            o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
            e_ = [
                math.ceil(k * (i + d) / i) for k, i, d in zip(k_, i_, d_)
            ]  # Expanding to avoid padding

            # Perform the expansion by reshaping and broadcasting the tensor
            xup = (
                self.reshape(
                    *prefix, *flatten((1, i) for i in i_)
                )  # Reshape the input tensor
                .expand(
                    *prefix, *flatten((e, i) for e, i in zip(e_, i_))
                )  # Expand the tensor
                .reshape(
                    *prefix, *[e * i for e, i in zip(e_, i_)]
                )  # Reshape to expanded size
            )

            # Slide by dilation (adjusts the tensor based on dilation)
            xup = xup.slice(
                slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)]
            )

            # Reshape and handle stride
            xup = xup.reshape(
                *prefix, *flatten((k, i + d) for k, i, d in zip(k_, i_, d_))
            )
            xup = xup.slice(
                slc_prefix
                + flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
            )

            # Reshape to the final form with stride applied and return result
            xup = xup.reshape(
                *prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_))
            )
            xup = xup.slice(
                slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
            )
            xup = xup.reshape(*prefix, *flatten((k, o) for k, o in zip(k_, o_)))

            # Permute dimensions for correct order
            return xup.permute(
                *range(len(prefix)),
                *[len(prefix) + i * 2 + 1 for i in range(len(k_))],  # Output channels
                *[len(prefix) + i * 2 for i in range(len(k_))],  # Kernel dimensions
            )

        # If kernel size and stride match, directly pool with no dilation
        o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
        xup = self.slice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
        xup = xup.reshape(*prefix, *flatten(((o, s) for o, s in zip(o_, s_))))
        xup = xup.slice(slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_)))

        # Permute dimensions for the final output
        return xup.permute(
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],  # Output channels
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],  # Kernel dimensions
        )

    def cat(self, *args, dim=0):
        """
        Concatenates tensors along a specified dimension. The tensors must have the same shape
        except for the dimension along which they are being concatenated.

        Args:
            *args (Tensor): The tensors to concatenate.
            dim (int): The dimension along which to concatenate. Can be negative to count from the end.

        Returns:
            Tensor: A tensor that is the result of concatenating the input tensors along the specified dimension.

        Raises:
            AssertionError: If the input tensors have mismatched shapes or contain zero-dimensional tensors.
        """
        dim = (
            (dim + len(self.shape)) if dim < 0 else dim
        )  # Adjust dimension for negative indexing
        # Check if the shapes of the tensors match except for the concatenation dimension
        assert all(
            len(y.shape) == len(self.shape)
            and all(y.shape[i] == s for i, s in enumerate(self.shape) if i != dim)
            for y in args
        )
        catargs = [self, *args]
        # Check if any tensor has zero-dimensional shape
        assert all(
            t.shape for t in catargs
        ), "zero-dimensional tensor cannot be concatenated"

        shapes = [
            s.shape[dim] for s in catargs
        ]  # Get the sizes of tensors along the concatenation axis
        shape_cumsum = [
            0,
            *accumulate(shapes),
        ]  # Cumulative sum of tensor sizes along the axis

        # Initialize slice indices for padding
        slc = [[(0, 0) for _ in self.shape] for _ in catargs]

        # Compute the slice ranges for each tensor
        for shp, k, s in zip(shapes, shape_cumsum[:-1], slc):
            s[dim] = (k, shape_cumsum[-1] - k - shp)

        # Concatenate the tensors by padding them along the specified dimension
        return reduce(
            Tensor.__add__, [arg.pad(tuple(s)) for arg, s in zip(catargs, slc)]
        )

    def dot(self, w):
        """
        Computes the dot product (or matrix multiplication) of two tensors. This assumes that
        the tensors can be broadcasted in a way that allows matrix multiplication.

        Args:
            w (Tensor): The tensor to multiply with.

        Returns:
            Tensor: The result of the matrix multiplication (dot product).

        Raises:
            AssertionError: If the tensors do not meet the dimensionality or shape requirements for matrix multiplication.
        """
        n1, n2 = len(self.shape), len(w.shape)

        # Check if both tensors are at least 1D
        assert (
            n1 != 0 and n2 != 0
        ), f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"

        # Check if the last dimension of the first tensor matches the second-to-last dimension of the second tensor
        assert (
            self.shape[-1] == w.shape[-min(n2, 2)]
        ), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"

        # Reshape the first tensor for broadcasting compatibility
        x = self.reshape(
            *self.shape[0:-1], *[1] * min(n1 - 1, n2 - 1, 1), self.shape[-1]
        )

        # Reshape the second tensor for broadcasting compatibility and transpose for multiplication
        w = w.reshape(
            *w.shape[0:-2], *[1] * min(n1 - 1, n2 - 1, 1), *w.shape[-min(n2, 2) :]
        ).transpose(-1, -min(n2, 2))

        # Perform element-wise multiplication followed by summation along the last axis
        return (x * w).sum(-1)

    def unsqueeze(self, dim):
        """
        Adds an extra dimension of size 1 at the specified position in the tensor's shape.

        Args:
            dim (int): The dimension where the new axis should be added. Can be negative to count from the end.

        Returns:
            Tensor: A new tensor with the added dimension of size 1.
        """
        if dim < 0:
            dim = len(self.shape) + dim + 1  # Handle negative dimensions
        return self.reshape(
            self.shape[:dim] + (1,) + self.shape[dim:]
        )  # Insert the new dimension

    def numel(self):
        """
        Returns the total number of elements in the tensor.

        Returns:
            int: The number of elements in the tensor.
        """
        return math.prod(
            self.shape
        )  # Calculate product of dimensions to get the number of elements

    def element_size(self) -> int:
        """
        Returns the size of each element in the tensor in bytes (based on the dtype).

        Returns:
            int: The byte size of each element.
        """
        return self.dtype.itemsize  # Return the item size for the dtype

    def nbytes(self) -> int:
        """
        Returns the total number of bytes consumed by the tensor.

        Returns:
            int: The total number of bytes in the tensor.
        """
        return (
            self.numel() * self.element_size()
        )  # Multiply the number of elements by element size

    def is_floating_point(self) -> bool:
        """
        Checks if the tensor's dtype is a floating-point type.

        Returns:
            bool: True if the tensor's dtype is float, False otherwise.
        """
        return Dtypes.is_float(self.dtype)  # Check if dtype is a float

    def _cumsum(self, axis: int = 0, _first_zero=False):
        """
        Computes the cumulative sum of the tensor along a specified axis.

        Args:
            axis (int): The axis along which to compute the cumulative sum.
            _first_zero (bool, optional): Whether to start the cumulative sum with zero. Defaults to False.

        Returns:
            Tensor: The cumulative sum of the tensor along the specified axis.
        """
        return (
            self.transpose(axis, -1)
            .pad2d(
                (self.shape[axis] - int(not _first_zero), 0)
            )  # Padding to handle starting sum
            ._pool((self.shape[axis],))  # Pooling operation for cumulative sum
            .sum(-1)
            .transpose(axis, -1)  # Transpose back to original axis order
        )

    def _reduce(self, fxn, axis=None, keepdim=False):
        """
        General reduction function that applies a specified function (e.g., sum, max) over specified axis.

        Args:
            fxn (function): The reduction function to apply (e.g., mlops.Sum, mlops.Max).
            axis (int or list of int, optional): The axis/axes along which to reduce. Defaults to None.
            keepdim (bool, optional): Whether to retain the reduced dimensions (size 1) in the output. Defaults to False.

        Returns:
            Tensor: The reduced tensor.
        """
        axis_ = (
            list(range(len(self.shape)))
            if axis is None
            else (
                [axis] if isinstance(axis, int) else list(axis)
            )  # Normalize axis format
        )
        axis_ = [
            x if x >= 0 else x + len(self.shape) for x in axis_
        ]  # Handle negative axes
        shape = tuple(
            s for i, s in enumerate(self.shape) if i not in axis_
        )  # Determine new shape

        # Handle edge case where shape has zeros
        if 0 in self.shape and 0 not in shape:
            return Tensor.full(
                tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape,
                {mlops.Sum: 0, mlops.Max: -float("inf")}[
                    fxn
                ],  # Default values for sum or max
            )

        # Apply reduction function
        ret = fxn.apply(
            self,
            new_shape=tuple([1 if i in axis_ else s for i, s in enumerate(self.shape)]),
        )

        return (
            ret if keepdim else ret.reshape(shape=shape)
        )  # Return the reduced tensor, reshape if necessary

    def sum(self, axis=None, keepdim=False):
        """
        Computes the sum of the tensor along the specified axis.

        Args:
            axis (int or list of int, optional): The axis/axes along which to sum. Defaults to None (sum over all axes).
            keepdim (bool, optional): Whether to retain reduced dimensions. Defaults to False.

        Returns:
            Tensor: The sum of the tensor along the specified axis.
        """
        return self._reduce(mlops.Sum, axis, keepdim)  # Call _reduce with sum function

    def max(self, axis=None, keepdim=False):
        """
        Computes the maximum of the tensor along the specified axis.

        Args:
            axis (int or list of int, optional): The axis/axes along which to find the maximum. Defaults to None.
            keepdim (bool, optional): Whether to retain reduced dimensions. Defaults to False.

        Returns:
            Tensor: The maximum of the tensor along the specified axis.
        """
        return self._reduce(mlops.Max, axis, keepdim)  # Call _reduce with max function

    def min(self, axis=None, keepdim=False):
        """
        Computes the minimum of the tensor along the specified axis.

        Args:
            axis (int or list of int, optional): The axis/axes along which to find the minimum. Defaults to None.
            keepdim (bool, optional): Whether to retain reduced dimensions. Defaults to False.

        Returns:
            Tensor: The minimum of the tensor along the specified axis.
        """
        return -(
            (-self).max(axis=axis, keepdim=keepdim)
        )  # Compute min by negating the max of the negative tensor

    def mean(self, axis=None, keepdim=False):
        """
        Computes the mean of the tensor along the specified axis.

        Args:
            axis (int or list of int, optional): The axis/axes along which to compute the mean. Defaults to None.
            keepdim (bool, optional): Whether to retain reduced dimensions. Defaults to False.

        Returns:
            Tensor: The mean of the tensor along the specified axis.
        """
        assert all_int(
            self.shape
        ), "does not support symbolic shape"  # Ensure all dimensions are integers
        out = self.sum(axis=axis, keepdim=keepdim)  # Compute the sum first
        return (
            out.mul(
                math.prod(out.shape) / math.prod(self.shape)
            )  # Scale by the product of dimensions
            if 0 not in self.shape
            else out
        )

    def cumsum(self, axis: int = 0):
        """
        Computes the cumulative sum of the tensor along the specified axis, with optimization for larger tensors.

        Args:
            axis (int): The axis along which to compute the cumulative sum. Defaults to 0.

        Returns:
            Tensor: The cumulative sum of the tensor along the specified axis.
        """
        SPLIT = 256  # Threshold for splitting tensor to optimize performance
        if self.shape[axis] <= SPLIT * 2:
            return self._cumsum(
                axis
            )  # Use the optimized _cumsum method for smaller tensors

        ret = self.transpose(axis, -1).pad2d(
            (
                round_up(self.shape[axis], SPLIT) - self.shape[axis],
                0,
            )  # Pad tensor for optimization
        )
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1] // SPLIT, SPLIT)._cumsum(
            -1
        )  # Reshape and apply cumsum
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[
            ..., :-1
        ]  # Compute base addition for cumsum
        base_add = base_add.unsqueeze(-1).expand(
            *base_add.shape, ret.shape[-1]
        )  # Expand for correct shape

        def fix(x: Tensor):
            return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[
                ..., -self.shape[axis] :
            ].transpose(axis, -1)  # Reshape and fix the tensor shape

        return fix(ret) + fix(base_add)  # Return the fixed cumulative sum

    def _softmax(self, axis):
        """
        Computes the softmax transformation (unstable for numerical stability),
        which is a common operation in classification tasks to convert logits to probabilities.

        Args:
            axis (int): The axis along which the softmax should be computed.

        Returns:
            tuple: A tuple containing:
                - m (Tensor): The input tensor subtracted by the maximum value along the specified axis.
                - e (Tensor): Exponentiated tensor after applying the softmax operation.
                - ss (Tensor): The sum of the exponentiated values along the specified axis.
        """
        m = self - self.max(
            axis=axis, keepdim=True
        )  # Numerical stability: subtracting max before exp
        e = m.exp()  # Apply element-wise exponential
        return (
            m,
            e,
            e.sum(axis=axis, keepdim=True),
        )  # Return intermediate results for log softmax and softmax

    def softmax(self, axis=-1):
        """
        Applies the softmax function along the specified axis to transform logits into probabilities.

        Args:
            axis (int, optional): The axis along which the softmax should be computed. Defaults to -1 (last axis).

        Returns:
            Tensor: The softmax probabilities.
        """
        _, e, ss = self._softmax(axis)  # Call internal softmax for intermediate results
        return e.div(ss)  # Element-wise division to normalize the exponentiated values

    def log_softmax(self, axis=-1):
        """
        Applies the log of the softmax function along the specified axis.

        Args:
            axis (int, optional): The axis along which the log softmax should be computed. Defaults to -1 (last axis).

        Returns:
            Tensor: The log of the softmax probabilities.
        """
        m, _, ss = self._softmax(axis)  # Call internal softmax for intermediate results
        return m - ss.log()  # Compute log-softmax by subtracting log of sum

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
        """
        Applies 2D max pooling on the input tensor using the specified kernel size, stride, and dilation.

        Args:
            kernel_size (tuple of int): The size of the pooling kernel. Defaults to (2, 2).
            stride (tuple of int, optional): The stride of the pooling operation. Defaults to None (same as kernel size).
            dilation (int, optional): The dilation factor for the pooling. Defaults to 1.

        Returns:
            Tensor: The result of the max pooling operation.
        """
        return self._pool(
            make_pair(kernel_size),  # Ensure kernel_size is in pair form
            stride
            if stride is not None
            else kernel_size,  # Use stride or default to kernel size
            dilation,
        ).max(
            axis=tuple(range(0 - len(make_pair(kernel_size)), 0))
        )  # Max pool reduction

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1):
        """
        Computes the sparse categorical cross-entropy loss between the predicted logits and the true labels.

        Args:
            Y (Tensor): The true labels (integers), with shape `(batch_size, ...)`.
            ignore_index (int, optional): A label index to ignore in the loss calculation. Defaults to -1.

        Returns:
            Tensor: The scalar loss value.
        """
        # NOTE: self is a logits input
        loss_mask = Y != ignore_index  # Create a mask to ignore specified labels
        y_counter = (
            Tensor.arange(
                self.shape[-1],
                dtype=Dtypes.int32,
                requires_grad=False,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(
                Y.numel(), self.shape[-1]
            )  # Prepare for comparison with true labels
        )
        # Generate binary mask for labels that match the true labels
        y = (
            (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0)
            * loss_mask.reshape(-1, 1)  # Apply the loss mask
        ).reshape(*Y.shape, self.shape[-1])
        return (
            self.log_softmax().mul(y).sum() / loss_mask.sum()
        )  # Log softmax followed by loss calculation

    def flatten(self, start_dim=0):
        """
        Flattens the input tensor starting from the specified dimension.

        Args:
            start_dim (int, optional): The dimension from which the flattening starts. Defaults to 0.

        Returns:
            Tensor: The reshaped tensor, flattened starting from the `start_dim`.
        """
        return self.reshape(
            shape=self.shape[:start_dim] + (-1,)
        )  # Reshape with flattened dimensions

    def deepwalk(self):
        """
        Traverses the computational graph recursively, starting from this tensor.

        Returns:
            list: A list of all tensors visited during the traversal.
        """

        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if getattr(node, "_ctx", None):
                for i in node._ctx.parents:
                    if i not in visited:
                        _deepwalk(i, visited, nodes)
                nodes.append(node)
            return nodes

        return _deepwalk(self, set(), [])  # Begin deepwalk traversal from this tensor

    def backward(self):
        """
        Computes the gradients of the tensor with respect to the scalar value (this tensor).

        This method assumes that the tensor is a scalar (0-dimensional) and computes the backward pass
        through the computational graph using reverse-mode autodiff.

        Returns:
            Tensor: This tensor, after computing gradients.
        """
        assert (
            self.shape == tuple()
        ), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # Initialize gradient for the scalar with 1
        self.grad = Tensor(1, device=self.device, requires_grad=False)

        for t0 in reversed(self.deepwalk()):  # Traverse the graph in reverse order
            assert t0.grad is not None  # Ensure the tensor has a gradient
            grads = t0._ctx.backward(t0.grad.lazydata)  # Call backward on the context
            grads = [
                Tensor(g, device=self.device, requires_grad=False)
                if g is not None
                else None
                for g in ([grads] if len(t0._ctx.parents) == 1 else grads)
            ]
            for t, g in zip(t0._ctx.parents, grads):
                if g is not None and t.requires_grad:
                    assert (
                        g.shape == t.shape
                    ), f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
                    t.grad = (
                        g if t.grad is None else (t.grad + g)
                    )  # Accumulate gradients if necessary
            del t0._ctx  # Clean up context to free memory
        return self  # Return this tensor after computing gradients
