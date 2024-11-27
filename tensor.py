import numpy as np

import math
from collections import defaultdict
from helpers import DType, Dtypes, all_int, argfix, flatten, make_pair, round_up
from lazy import LazyBuffer
import time
from functools import reduce
from itertools import accumulate
from ops import Device, LoadOps


class Function:
    def __init__(self, device, *tensors):
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
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args, **kwargs) -> LazyBuffer:
        raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn, *x, **kwargs):  # pyright: ignore
        ctx = fxn(x[0].device, *x)
        ret = Tensor(
            ctx.forward(*[t.lazydata for t in x], **kwargs),
            device=ctx.device,
            requires_grad=ctx.requires_grad,
        )
        if ctx.requires_grad and not Tensor.no_grad:
            ret._ctx = ctx  # pyright: ignore
        return ret


import mlops  # noqa [E402]


class Train:
    def __init__(self, val=True):
        self.val = val

    def __enter__(self):
        self.prev, Tensor.training = Tensor.training, self.val

    def __exit__(self):
        Tensor.training = self.prev


class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)
    training = False
    no_grad = False
    default_type = Dtypes.float32

    def __init__(self, data, device=None, dtype=None, requires_grad=None):
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
        device = Device.canonicalize()
        # tensors have gradients, buffers do not
        self.grad = None

        # NOTE: this can be in three states. False and None: no gradient, True: gradient
        # None (the default) will be updated to True if it's put in an optimizer
        self.requires_grad = requires_grad

        # internal variables used for autograd graph construction

        self._ctx = None
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

        # data is a LazyBuffer, but it might be on the wrong device
        if not isinstance(data, LazyBuffer):
            raise RuntimeError(
                f"can't create Tensor from {data!r} with type {type(data)}"
            )
        self.lazydata = data if data.device == device else data.copy_to_device(device)

    @property
    def shape(self):
        return self.lazydata.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    @property
    def device(self):
        return self.lazydata.device

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs):
        dtype = kwargs.pop("dtype", Tensor.default_type)
        return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    _seed: int = int(time.time())

    @staticmethod
    def _loadop(
        op,
        sz,
        device=None,
        dtype=None,
        arg=None,
        **kwargs,
    ):
        assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
        return Tensor(
            LazyBuffer.loadop(
                op,
                (sz,),
                Tensor.default_type if dtype is None else dtype,
                arg,
            ),
            dtype=dtype,
            device=device,
            **kwargs,
        )

    @staticmethod
    def rand(*shape, **kwargs):
        Tensor._seed += 1
        return Tensor._loadop(
            LoadOps.RAND,
            math.prod((shape := argfix(*shape))),
            arg=Tensor._seed,
            **kwargs,
        ).reshape(shape)

    @staticmethod
    def scaled_uniform(*shape, **kwargs):
        return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(
            math.prod(shape) ** -0.5
        )

    def detach(self):
        return Tensor(self.lazydata, device=self.device, requires_grad=False)

    def cast(self, dtype: DType):
        return mlops.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self

    def numpy(self) -> np.ndarray:
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
        if device is None or device == self.device:
            return self
        ret = Tensor(self.lazydata, device)
        if self.grad:
            ret.grad = self.grad.to(device)
        return ret

    def realize(self):
        self.lazydata.schedule()
        return self

    @staticmethod
    def corealize(lst):
        seen = set()
        sched = []
        for t in lst:
            sched += t.lazydata.schedule(seen)

    @staticmethod
    def zeros(*shape, **kwargs):
        return Tensor.full(argfix(*shape), 0, **kwargs)

    def assign(self, x):
        if x.__class__ is not Tensor:
            x = Tensor(x, device=self.device, dtype=self.dtype)
        assert (
            self.shape == x.shape and self.device == x.device
        ), f"assign shape mismatch {self.shape} != {x.shape} or device mismatch {self.device} != {x.device}"
        assert not x.requires_grad  # self requires_grad is okay?
        if self.dtype == x.dtype and self.lazydata.realized is not None:
            x.lazydata.output_buffer = self.lazydata.realized  # pyright: ignore
        self.lazydata = x.lazydata
        return self

    @staticmethod
    def full(shape, fill_value, **kwargs):
        return (
            Tensor(fill_value, **kwargs)
            .reshape([1] * len(new_shape := argfix(shape)))
            .expand(new_shape)
        )

    def full_like(self, fill_value, **kwargs):
        return Tensor.full(
            self.shape,
            fill_value=fill_value,
            dtype=kwargs.pop("dtype", self.dtype),
            device=kwargs.pop("device", self.device),
            **kwargs,
        )

    def ones_like(self, **kwargs):
        return self.full_like(1, **kwargs)

    def contiguous(self):
        return mlops.Contiguous.apply(self)

    def contiguous_backward(self):
        return mlops.ContiguousBackward.apply(self)

    def reshape(self, shape, *args):
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
        return (
            (self < x)
            .detach()
            .where(x, (self > x).detach().where(self, (self + x) / 2))
        )

    def minimum(self, x):
        return -((-self).maximum(-x))

    def where(self, input, other):
        x_, y = self._broadcasted(input)
        x, z = x_._broadcasted(other)
        return mlops.Where.apply(x, *y._broadcasted(z))

    def flip(self, axis, *args):
        return mlops.Flip.apply(
            self,
            axis=[x if x >= 0 else x + len(self.shape) for x in argfix(axis, *args)],
        )

    def permute(self, order, *args):
        return mlops.Permute.apply(self, order=argfix(order, *args))

    def pad(self, arg, value=0.0):
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
        x = self._to_float(x)
        return (
            mlops.Sub.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor or x
            else (-self if reverse else self)
        )

    def div(self, x, reverse=False):
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
        return mlops.Sqrt.apply(self)

    def exp(self):
        return mlops.Exp.apply(self)

    def reciprocal(self):
        return 1.0 / self

    def sin(self):
        return mlops.Sin.apply(self)

    def sign(self):
        return self / (self.abs() + 1e-10)

    def relu(self):
        return mlops.Relu.apply(self)

    def abs(self):
        return self.relu() + (-self).relu()

    def log(self):
        return mlops.Log.apply(self)

    def cos(self):
        return ((math.pi / 2) - self).sin()

    def clip(self, min_, max_):
        return self.maximum(min_).minimum(max_)

    def trunc(self):
        return self.cast(Dtypes.int32).contiguous().cast(self.dtype)

    def pow(self, x, reverse=False):
        x = self._to_float(x)
        if x.__class__ is not Tensor and not reverse:
            # simple pow identities
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
        # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
        sign = (
            (x * math.pi).cos()
            if isinstance(x, Tensor)
            else math.cos(x * math.pi)
            if not reverse
            else (self * math.pi).cos()
        )
        # we only need to correct the sign if the base is negative
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
        # we need 0 to be positive so we need to correct base_sign when the base is 0
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
        # inject nan if the base is negative and the power is not an integer
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

    def matmul(self, x, reverse=False):
        return x.dot(self) if reverse else self.dot(x)

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
        return (self < x) + (self > x)  # type: ignore

    def __eq__(self, x):
        return 1.0 - (self != x)  # type: ignore

    def _to_float(self, x):
        return x

    def neg(self):
        return mlops.Neg.apply(self)

    def add(self, x, reverse=False):
        x = self._to_float(x)
        return (
            mlops.Add.apply(*self._broadcasted(x, reverse))
            if x.__class__ is Tensor or x
            else self
        )

    def expand(self, shape, *args):
        return mlops.Expand.apply(
            self,
            shape=tuple(
                [x if x != -1 else s for s, x in zip(self.shape, argfix(shape, *args))]
            ),
        )

    def shrink(self, arg):
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
        order = list(range(len(self.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    def pad2d(self, padding, value=0):
        slc = [
            (-p0, s + p1)
            for p0, p1, s in zip(padding[::2], padding[1::2], self.shape[::-1])
        ][::-1]
        return self.slice(
            [(0, s) for s in self.shape[: -(len(padding) // 2)]] + slc, value=value
        )

    def conv2d(
        self,
        weight,
        bias=None,
        groups=1,
        stride=1,
        dilation=1,
        padding=0,
    ):
        (bs, cin_), (cout, cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
        assert (
            groups * cin == cin_ and len(self.shape) == len(weight.shape)
        ), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
        if isinstance(padding, (tuple, list)):
            assert (
                len(padding) == 2 * len(HW) or len(padding) == len(HW)
            ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"
        padding_ = (
            [padding] * 2 * len(HW)
            if isinstance(padding, int)
            else (
                padding
                if len(padding) == 2 * len(HW)
                else [p for p in padding for _ in range(2)][::-1]  # pyright: ignore
            )
        )

        # conv2d is a pooling op (with padding)
        x = self.pad2d(padding_)._pool(
            HW, stride, dilation
        )  # (bs, groups*cin, oy, ox, H, W)
        rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
        if not all(x == 3 for x in HW) or stride != 1 or dilation != 1:
            # normal conv
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

            # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
            ret = (
                (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))
                .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
                .reshape(bs, cout, *oyx)
            )
            return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

        # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
        def apply_matrix(mat, t, dim=0):
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

        # todo: stride == dilation
        # use padding to round up to 4x4 output tiles
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

        g = weight.permute(
            *range(len(weight.shape) - len(HW), len(weight.shape)),
            *range(len(weight.shape) - len(HW)),
        )  # move HW to the front

        # compute 6x6 winograd tiles: GgGt, BtdB
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

        ret = apply_matrix(
            winograd_At, (gfactors * dfactors).sum(axis=-1 - len(HW))
        )  # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)

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
        assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
        assert all_int(self.shape) and all_int(
            k_
        ), f"does not support symbolic {self.shape=}, {k_=}"
        s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
        assert len(k_) == len(s_) and len(k_) == len(
            d_
        ), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
        slc_prefix, prefix, i_ = (
            [(0, x) for x in self.shape[0 : -len(k_)]],
            self.shape[0 : -len(k_)],
            self.shape[-len(k_) :],
        )
        if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
            o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
            e_ = [
                math.ceil(k * (i + d) / i) for k, i, d in zip(k_, i_, d_)
            ]  # expands such that we don't need padding
            xup = (
                self.reshape(*prefix, *flatten((1, i) for i in i_))
                .expand(*prefix, *flatten((e, i) for e, i in zip(e_, i_)))
                .reshape(*prefix, *[e * i for e, i in zip(e_, i_)])
            )
            # slide by dilation
            xup = xup.slice(
                slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)]
            )
            xup = xup.reshape(
                *prefix, *flatten((k, i + d) for k, i, d in zip(k_, i_, d_))
            )
            xup = xup.slice(
                slc_prefix
                + flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
            )
            # handle stride, and permute to move reduce to the end
            xup = xup.reshape(
                *prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_))
            )
            xup = xup.slice(
                slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
            )
            xup = xup.reshape(*prefix, *flatten((k, o) for k, o in zip(k_, o_)))
            return xup.permute(
                *range(len(prefix)),
                *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
                *[len(prefix) + i * 2 for i in range(len(k_))],
            )

        o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
        xup = self.slice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
        xup = xup.reshape(*prefix, *flatten(((o, s) for o, s in zip(o_, s_))))
        xup = xup.slice(slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_)))
        return xup.permute(
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
        )

    def __hash__(self):
        return id(self)

    def cat(self, *args, dim=0):
        dim = (dim + len(self.shape)) if dim < 0 else dim
        assert all(
            len(y.shape) == len(self.shape)
            and all(y.shape[i] == s for i, s in enumerate(self.shape) if i != dim)
            for y in args
        )
        catargs = [self, *args]
        assert all(
            t.shape for t in catargs
        ), "zero-dimensional tensor cannot be concatenated"
        shapes = [s.shape[dim] for s in catargs]
        shape_cumsum = [0, *accumulate(shapes)]
        slc = [[(0, 0) for _ in self.shape] for _ in catargs]
        for shp, k, s in zip(shapes, shape_cumsum[:-1], slc):
            s[dim] = (k, shape_cumsum[-1] - k - shp)
        return reduce(
            Tensor.__add__, [arg.pad(tuple(s)) for arg, s in zip(catargs, slc)]
        )

    @staticmethod
    def stack(tensors, dim=0):
        first = tensors[0].unsqueeze(dim)
        unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
        # checks for shapes and number of dimensions delegated to cat
        return first.cat(*unsqueezed_tensors, dim=dim)

    def dot(self, w):
        n1, n2 = len(self.shape), len(w.shape)
        assert (
            n1 != 0 and n2 != 0
        ), f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        assert (
            self.shape[-1] == w.shape[-min(n2, 2)]
        ), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
        x = self.reshape(
            *self.shape[0:-1], *[1] * min(n1 - 1, n2 - 1, 1), self.shape[-1]
        )
        w = w.reshape(
            *w.shape[0:-2], *[1] * min(n1 - 1, n2 - 1, 1), *w.shape[-min(n2, 2) :]
        ).transpose(-1, -min(n2, 2))
        return (x * w).sum(-1)

    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self.shape) + dim + 1
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def __getitem__(
        self, val
    ):  # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz:
                return e if e != -1 else dim_sz - 1
            raise IndexError(
                f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}"
            )

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)
        for i, v in enumerate(orig_slices):
            count[type(v)].append(i)

        if (
            num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])
        ) > len(self.shape):
            raise IndexError(
                f"too many indices for tensor of dimension {len(self.shape)}"
            )
        if len(ellipsis_found := count[type(Ellipsis)]) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
            len(self.shape) - num_slices
        )

        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [
            v
            if isinstance(v, slice)
            else slice(y_ := normalize_int(v, i, dim_sz), y_ + 1)
            if isinstance(v, int)
            else slice(None)
            for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
        ]

        start, stop, strides = (
            zip(*y)
            if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)])
            else ((), (), ())
        )
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
        if any(abs(s) != 1 for s in strides):
            strides = tuple(abs(s) for s in strides)
            # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
            padded_tensor = sliced_tensor.pad(
                tuple(
                    (0, s - (dim_sz % s) if dim_sz % s != 0 else 0)
                    for s, dim_sz in zip(strides, sliced_tensor.shape)
                )
            )
            # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
            reshaped_tensor = padded_tensor.reshape(
                flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides))
            )
            new_shape = reshaped_tensor.shape[::2]
            # Shrink: do [:, 0]
            sliced_tensor = reshaped_tensor.shrink(
                tuple(flatten(((0, sh), (0, 1)) for sh in new_shape))
            )

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
            else:  # s is int or slice or Tensor
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

        if tensors:  # Fancy/tensor indexing
            # normalize idx
            # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
            idx = [
                t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t
                for d, t in zip(dim, tensors)
            ]
            max_dim = max(i.ndim for i in idx)
            # compute sum_dim, arange, and idx
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
            # iteratively fancy index
            for a, i, sd in zip(arange, idx, sum_dim):
                ret = (a == i).mul(ret).sum(sd)
            # special permute case
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

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        if stop is None:
            stop, start = start, 0
        return Tensor.full(
            (math.ceil((stop - start) / step),), step, **kwargs
        ).cumsum() + (start - step)

    def __setitem__(self, s, v):
        return self.__getitem__(s).assign(v)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def nbytes(self) -> int:
        return self.numel() * self.element_size()

    def is_floating_point(self) -> bool:
        return Dtypes.is_float(self.dtype)

    def _cumsum(self, axis: int = 0, _first_zero=False):
        return (
            self.transpose(axis, -1)
            .pad2d((self.shape[axis] - int(not _first_zero), 0))
            ._pool((self.shape[axis],))
            .sum(-1)
            .transpose(axis, -1)
        )

    def _reduce(
        self,
        fxn,
        axis=None,
        keepdim=False,
    ):
        axis_ = (
            list(range(len(self.shape)))
            if axis is None
            else ([axis] if isinstance(axis, int) else list(axis))
        )
        axis_ = [x if x >= 0 else x + len(self.shape) for x in axis_]
        shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)
        if 0 in self.shape and 0 not in shape:
            return Tensor.full(
                tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape,
                {mlops.Sum: 0, mlops.Max: -float("inf")}[fxn],
            )
        ret = fxn.apply(
            self,
            new_shape=tuple([1 if i in axis_ else s for i, s in enumerate(self.shape)]),
        )
        return ret if keepdim else ret.reshape(shape=shape)

    def sum(self, axis=None, keepdim=False):
        return self._reduce(mlops.Sum, axis, keepdim)

    def max(self, axis=None, keepdim=False):
        return self._reduce(mlops.Max, axis, keepdim)

    def min(self, axis=None, keepdim=False):
        return -((-self).max(axis=axis, keepdim=keepdim))

    def mean(self, axis=None, keepdim=False):
        assert all_int(self.shape), "does not support symbolic shape"
        out = self.sum(axis=axis, keepdim=keepdim)
        return (
            out.mul(math.prod(out.shape) / math.prod(self.shape))
            if 0 not in self.shape
            else out
        )

    def cumsum(self, axis: int = 0):
        # TODO: someday the optimizer will find this on it's own
        # for now this is a two stage cumsum
        SPLIT = 256
        if self.shape[axis] <= SPLIT * 2:
            return self._cumsum(axis)
        ret = self.transpose(axis, -1).pad2d(
            (round_up(self.shape[axis], SPLIT) - self.shape[axis], 0)
        )
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1] // SPLIT, SPLIT)._cumsum(-1)
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
        base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])

        def fix(x: Tensor):
            return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[
                ..., -self.shape[axis] :
            ].transpose(axis, -1)

        return fix(ret) + fix(base_add)

    def _softmax(self, axis):
        m = self - self.max(axis=axis, keepdim=True)
        e = m.exp()
        return m, e, e.sum(axis=axis, keepdim=True)

    def softmax(self, axis=-1):
        _, e, ss = self._softmax(axis)
        return e.div(ss)

    def log_softmax(self, axis=-1):
        m, _, ss = self._softmax(axis)
        return m - ss.log()

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
        return self._pool(
            make_pair(kernel_size),
            stride if stride is not None else kernel_size,
            dilation,
        ).max(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1):
        # NOTE: self is a logits input
        loss_mask = Y != ignore_index
        y_counter = (
            Tensor.arange(
                self.shape[-1],
                dtype=Dtypes.int32,
                requires_grad=False,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(Y.numel(), self.shape[-1])
        )
        y = (
            (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0)
            * loss_mask.reshape(-1, 1)
        ).reshape(*Y.shape, self.shape[-1])
        return self.log_softmax().mul(y).sum() / loss_mask.sum()

    def flatten(self, start_dim=0):
        return self.reshape(shape=self.shape[:start_dim] + (-1,))

    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if getattr(node, "_ctx", None):
                for i in node._ctx.parents:
                    if i not in visited:
                        _deepwalk(i, visited, nodes)
                nodes.append(node)
            return nodes

        return _deepwalk(self, set(), [])

    def backward(self):
        assert (
            self.shape == tuple()
        ), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
        self.grad = Tensor(1, device=self.device, requires_grad=False)

        for t0 in reversed(self.deepwalk()):
            assert t0.grad is not None
            grads = t0._ctx.backward(t0.grad.lazydata)
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
                    t.grad = g if t.grad is None else (t.grad + g)
            del t0._ctx
        return self
