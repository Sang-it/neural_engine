from typing import Union, List, Tuple, Optional, Any
import numpy as np

from helpers import DType, Dtypes
from ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps


class RawCPUBuffer:
    """
    A simple wrapper for raw CPU buffer (NumPy array).

    Attributes:
        x (np.ndarray): The underlying NumPy array
    """

    def __init__(self, x: np.ndarray):
        """
        Initialize RawCPUBuffer with a NumPy array.

        Args:
            x (np.ndarray): NumPy array to wrap
        """
        self.x = x

    def toCPU(self) -> np.ndarray:
        """
        Return the underlying NumPy array.

        Returns:
            np.ndarray: The wrapped array
        """
        return self.x


class LazyBuffer:
    """
    A lazy evaluation buffer that wraps NumPy arrays with deferred operations.

    Attributes:
        device (str): Device identifier, defaults to "CPU"
    """

    device = "CPU"

    def __init__(self, buf: np.ndarray):
        """
        Initialize LazyBuffer with a NumPy array.

        Args:
            buf (np.ndarray): NumPy array to wrap
        """
        self._np = buf

    @property
    def base(self) -> "LazyBuffer":
        """
        Return the base buffer (self in this implementation).

        Returns:
            LazyBuffer: The current buffer
        """
        return self

    @property
    def dtype(self) -> DType:
        """
        Get the dtype of the underlying NumPy array.

        Returns:
            DType: The data type of the array
        """
        return Dtypes.from_np(self._np.dtype)

    @property
    def realized(self) -> RawCPUBuffer:
        """
        Convert to a RawCPUBuffer.

        Returns:
            RawCPUBuffer: Wrapped NumPy array
        """
        return RawCPUBuffer(self._np)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the underlying array.

        Returns:
            Tuple[int, ...]: Array dimensions
        """
        return self._np.shape

    def copy_to_device(self, _: str) -> "LazyBuffer":
        """
        No-op device copy (always returns self for CPU).

        Args:
            _ (str): Target device (ignored)

        Returns:
            LazyBuffer: Current buffer
        """
        return self

    @staticmethod
    def fromCPU(x: np.ndarray) -> "LazyBuffer":
        """
        Create a LazyBuffer from a NumPy array.

        Args:
            x (np.ndarray): Input NumPy array

        Returns:
            LazyBuffer: Wrapped array
        """
        return LazyBuffer(x)

    @staticmethod
    def loadop(
        op: LoadOps, shape: Tuple[int, ...], dtype: DType, arg: Optional[Any] = None
    ) -> "LazyBuffer":
        """
        Create a LazyBuffer using various load operations.

        Args:
            op (LoadOps): Load operation type
            shape (Tuple[int, ...]): Shape of the array
            dtype (DType): Data type of the array
            arg (Optional[Any], optional): Additional argument for load operation

        Returns:
            LazyBuffer: Created buffer

        Raises:
            NotImplementedError: For unsupported load operations
        """
        if op == LoadOps.RAND:
            return LazyBuffer(
                np.random.default_rng(arg).random(size=shape, dtype=dtype.np)
            )
        elif op == LoadOps.CONST:
            return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:
            return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else:
            raise NotImplementedError(f"Unsupported load operation: {op}")

    def contiguous(self) -> "LazyBuffer":
        """
        No-op contiguous method (returns self).

        Returns:
            LazyBuffer: Current buffer
        """
        return self

    def schedule(self, _: Optional[Any] = None) -> List[Any]:
        """
        No-op scheduling method.

        Returns:
            List[Any]: Empty list
        """
        return []

    def const(self, x: Any) -> "LazyBuffer":
        """
        Create a buffer filled with a constant value.

        Args:
            x (Any): Constant value to fill

        Returns:
            LazyBuffer: New buffer with constant values
        """
        return LazyBuffer(np.full_like(self._np, x))

    def cast(self, dtype: DType, bitcast: bool = False) -> "LazyBuffer":
        """
        Cast the buffer to a different data type.

        Args:
            dtype (DType): Target data type
            bitcast (bool, optional): Whether to use bitcast. Defaults to False.

        Returns:
            LazyBuffer: Casted buffer
        """
        return LazyBuffer(
            self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np)
        )

    def e(
        self, op: Union[UnaryOps, BinaryOps, TernaryOps], *srcs: "LazyBuffer"
    ) -> "LazyBuffer":
        """
        Execute unary, binary, and ternary operations.

        Args:
            op (Union[UnaryOps, BinaryOps, TernaryOps]): Operation to perform
            *srcs (LazyBuffer): Source buffers for the operation

        Returns:
            LazyBuffer: Result of the operation

        Raises:
            NotImplementedError: For unsupported operations
        """
        # Unary operations
        if op == UnaryOps.NEG:
            ret = -self._np
        elif op == UnaryOps.EXP2:
            ret = np.exp2(self._np)
        elif op == UnaryOps.LOG2:
            ret = np.log2(self._np)
        elif op == UnaryOps.SIN:
            ret = np.sin(self._np)
        elif op == UnaryOps.SQRT:
            ret = np.sqrt(self._np)

        # Binary operations
        elif op == BinaryOps.ADD:
            ret = self._np + srcs[0]._np
        elif op == BinaryOps.SUB:
            ret = self._np - srcs[0]._np
        elif op == BinaryOps.MUL:
            ret = self._np * srcs[0]._np
        elif op == BinaryOps.DIV:
            ret = self._np / srcs[0]._np
        elif op == BinaryOps.MAX:
            ret = np.maximum(self._np, srcs[0]._np)
        elif op == BinaryOps.CMPLT:
            ret = self._np < srcs[0]._np

        # Ternary operations
        elif op == TernaryOps.WHERE:
            ret = np.where(self._np, srcs[0]._np, srcs[1]._np)
        else:
            raise NotImplementedError(f"Unsupported operation: {op}")

        # Determine output dtype
        out_dtype = (
            self.dtype.np
            if len(srcs) == 0
            else max(self.dtype, *[x.dtype for x in srcs]).np
        )
        return LazyBuffer(ret.astype(out_dtype, copy=False))

    def r(self, op: ReduceOps, new_shape: Tuple[int, ...]) -> "LazyBuffer":
        """
        Perform reduction operations.

        Args:
            op (ReduceOps): Reduction operation
            new_shape (Tuple[int, ...]): Target shape after reduction

        Returns:
            LazyBuffer: Reduced buffer

        Raises:
            AssertionError: If shapes are incompatible
            NotImplementedError: For unsupported reduction operations
        """
        assert len(self.shape) == len(
            new_shape
        ), "Reduce shapes must have same dimensions"

        # Determine reduction axes
        axis = tuple(i for i, (a, b) in enumerate(zip(self.shape, new_shape)) if a != b)

        if op == ReduceOps.SUM:
            return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
        elif op == ReduceOps.MAX:
            return LazyBuffer(self._np.max(axis, keepdims=True))

    def reshape(self, arg: Tuple[int, ...]) -> "LazyBuffer":
        """
        Reshape the buffer.

        Args:
            arg (Tuple[int, ...]): New shape

        Returns:
            LazyBuffer: Reshaped buffer
        """
        return LazyBuffer(self._np.reshape(arg))

    def expand(self, arg: Tuple[int, ...]) -> "LazyBuffer":
        """
        Broadcast the buffer to a new shape.

        Args:
            arg (Tuple[int, ...]): Target shape

        Returns:
            LazyBuffer: Broadcasted buffer
        """
        return LazyBuffer(np.broadcast_to(self._np, arg))

    def shrink(self, arg: Tuple[Tuple[int, int], ...]) -> "LazyBuffer":
        """
        Slice the buffer using the provided ranges.

        Args:
            arg (List[Tuple[int, int]]): Slicing ranges for each dimension

        Returns:
            LazyBuffer: Sliced buffer
        """
        return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)])

    def permute(self, arg: Tuple[int, ...]) -> "LazyBuffer":
        """
        Transpose the buffer.

        Args:
            arg (Tuple[int, ...]): New axis order

        Returns:
            LazyBuffer: Transposed buffer
        """
        return LazyBuffer(self._np.transpose(arg))

    def pad(self, arg: Any) -> "LazyBuffer":
        """
        Pad the buffer.

        Args:
            arg (Any): Padding specification

        Returns:
            LazyBuffer: Padded buffer
        """
        return LazyBuffer(np.pad(self._np, arg))

    def stride(self, arg: Tuple[int, ...]) -> "LazyBuffer":
        """
        Create a strided view of the buffer.

        Args:
            arg (Tuple[int, ...]): Stride values for each dimension

        Returns:
            LazyBuffer: Strided buffer
        """
        return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)])
