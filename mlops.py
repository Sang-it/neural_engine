import math

from typing import List, Tuple
from helpers import argsort, DType
from ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tensor import Function
from lazy import LazyBuffer


class Contiguous(Function):
    """
    Function to ensure a LazyBuffer is contiguous.

    This function passes through the input if it's already contiguous,
    and creates a contiguous copy if needed.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Ensure the input LazyBuffer is contiguous.

        Args:
            x (LazyBuffer): Input buffer

        Returns:
            LazyBuffer: Contiguous buffer
        """
        return x.contiguous()

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Gradient pass-through for contiguity operation.

        Args:
            grad_output (LazyBuffer): Gradient from subsequent layers

        Returns:
            LazyBuffer: Unchanged gradient
        """
        return grad_output


class ContiguousBackward(Function):
    """
    Alternate contiguous function that applies contiguity in the backward pass.

    Useful in scenarios where contiguity is needed during gradient computation.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Pass through the input without modification.

        Args:
            x (LazyBuffer): Input buffer

        Returns:
            LazyBuffer: Unmodified input buffer
        """
        return x

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Ensure the gradient is contiguous.

        Args:
            grad_output (LazyBuffer): Gradient from subsequent layers

        Returns:
            LazyBuffer: Contiguous gradient
        """
        return grad_output.contiguous()


class Cast(Function):
    """
    Function to cast a LazyBuffer to a different data type.

    Supports both type conversion and bitcast operations.
    """

    def forward(self, x: LazyBuffer, dtype: DType, bitcast: bool = False) -> LazyBuffer:
        """
        Cast the input buffer to a new data type.

        Args:
            x (LazyBuffer): Input buffer
            dtype (DType): Target data type
            bitcast (bool, optional): Whether to use bitcast. Defaults to False.

        Returns:
            LazyBuffer: Casted buffer
        """
        # Store original dtype and bitcast method for backward pass
        self.input_dtype = x.dtype
        self.bitcast = bitcast

        return x.cast(dtype, bitcast)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Cast the gradient back to the original input data type.

        Args:
            grad_output (LazyBuffer): Gradient from subsequent layers

        Returns:
            LazyBuffer: Gradient casted to original input type
        """
        return grad_output.cast(self.input_dtype, self.bitcast)


class Zero(Function):
    """
    Function that replaces the input with zeros.

    Useful for creating zero-initialized tensors or zero gradients.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Replace the input tensor with zeros of the same shape.

        Args:
            x (LazyBuffer): Input tensor

        Returns:
            LazyBuffer: Tensor filled with zeros
        """
        return x.const(0)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        """
        Create a zero gradient of the same shape.

        Args:
            grad (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient filled with zeros
        """
        return grad.const(0)


class Neg(Function):
    """
    Function to negate a tensor.

    Applies the unary negation operation to the input.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Negate the input tensor.

        Args:
            x (LazyBuffer): Input tensor

        Returns:
            LazyBuffer: Negated tensor
        """
        return x.e(UnaryOps.NEG)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        """
        Negate the gradient during backpropagation.

        Args:
            grad (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Negated gradient
        """
        return grad.e(UnaryOps.NEG)


class Sin(Function):
    """
    Function to compute sine of a tensor.

    Implements forward and backward passes for sine operation.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Compute sine of the input tensor.

        Args:
            x (LazyBuffer): Input tensor

        Returns:
            LazyBuffer: Sine of the input tensor
        """
        self.x = x
        return x.e(UnaryOps.SIN)

    def backward(self, grad: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient for sine operation using cosine derivative.

        Args:
            grad (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient multiplied by cosine of input
        """
        return (
            self.x.const(math.pi / 2)
            .e(BinaryOps.SUB, self.x)
            .e(UnaryOps.SIN)
            .e(BinaryOps.MUL, grad)
        )


class Relu(Function):
    """
    Rectified Linear Unit (ReLU) activation function.

    Replaces negative values with zero, leaves positive values unchanged.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Apply ReLU activation to the input tensor.

        Args:
            x (LazyBuffer): Input tensor

        Returns:
            LazyBuffer: ReLU activated tensor
        """
        self.ret = x.e(BinaryOps.MAX, x.const(0))
        return self.ret

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient for ReLU operation.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient masked by positive values
        """
        return (
            self.ret.const(0).e(BinaryOps.CMPLT, self.ret).e(BinaryOps.MUL, grad_output)
        )


class Log(Function):
    """
    Natural logarithm function for tensors.

    Computes logarithm using base-2 logarithm and scaling.
    """

    def forward(self, x: LazyBuffer) -> LazyBuffer:
        """
        Compute natural logarithm of the input tensor.

        Args:
            x (LazyBuffer): Input tensor

        Returns:
            LazyBuffer: Natural logarithm of input
        """
        self.x = x
        return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient for logarithm operation.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient divided by input
        """
        return grad_output.e(BinaryOps.DIV, self.x)


class Exp(Function):
    def forward(self, x: LazyBuffer):
        self.ret = x.e(BinaryOps.MUL, x.const(1 / math.log(2))).e(UnaryOps.EXP2)
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        return self.ret.e(BinaryOps.MUL, grad_output)


class Sqrt(Function):
    def forward(self, x: LazyBuffer):
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        return grad_output.e(
            BinaryOps.DIV, self.ret.e(BinaryOps.MUL, self.ret.const(2))
        )


# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
    def forward(self, x: LazyBuffer):
        self.ret = x.const(1).e(
            BinaryOps.DIV,
            x.const(1).e(
                BinaryOps.ADD,
                x.e(BinaryOps.MUL, x.const(-1 / math.log(2))).e(UnaryOps.EXP2),
            ),
        )
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        return self.ret.e(
            BinaryOps.MUL, self.ret.const(1).e(BinaryOps.SUB, self.ret)
        ).e(BinaryOps.MUL, grad_output)


# ************* binary ops *************


class Less(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.e(BinaryOps.CMPLT, y)


class Add(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.e(BinaryOps.ADD, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output if self.needs_input_grad[
            0
        ] else None, grad_output if self.needs_input_grad[1] else None


class Sub(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        return x.e(BinaryOps.SUB, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output if self.needs_input_grad[0] else None, grad_output.e(
            UnaryOps.NEG
        ) if self.needs_input_grad[1] else None


class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        self.x, self.y = x, y
        return x.e(BinaryOps.MUL, y)

    def backward(self, grad_output: LazyBuffer):
        return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[
            0
        ] else None, self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[
            1
        ] else None


class Div(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer):
        self.x, self.y = x, y
        return x.e(BinaryOps.DIV, y)

    def backward(self, grad_output: LazyBuffer):
        return grad_output.e(BinaryOps.DIV, self.y) if self.needs_input_grad[
            0
        ] else None, grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(
            BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y)
        ) if self.needs_input_grad[1] else None


# ************* ternary ops *************


class Where(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer, z: LazyBuffer):
        self.x = x
        return x.e(TernaryOps.WHERE, y, z)

    def backward(self, grad_output: LazyBuffer):
        return (
            None,
            self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0))
            if self.needs_input_grad[1]
            else None,
            self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output)
            if self.needs_input_grad[2]
            else None,
        )


# ************* reduce ops *************


class Sum(Function):
    def forward(self, x: LazyBuffer, new_shape):
        self.input_shape = x.shape
        return x.r(ReduceOps.SUM, new_shape)

    def backward(self, grad_output: LazyBuffer):
        return grad_output.expand(self.input_shape)


class Max(Function):
    def forward(self, x: LazyBuffer, new_shape):
        self.x, self.ret = x, x.r(ReduceOps.MAX, new_shape)
        return self.ret

    def backward(self, grad_output: LazyBuffer):
        # 1s in locations where the max was chosen (can be two locations)
        max_is_1s = self.x.const(1.0).e(
            BinaryOps.SUB, self.x.e(BinaryOps.CMPLT, self.ret.expand(self.x.shape))
        )
        div = max_is_1s.r(ReduceOps.SUM, grad_output.shape).expand(self.x.shape)
        return max_is_1s.e(BinaryOps.DIV, div).e(
            BinaryOps.MUL, grad_output.expand(self.x.shape)
        )


# ************* movement ops *************


# NOTE: this is sum in reverse
class Expand(Function):
    """
    Tensor expansion operation.

    Expands a tensor to a specified shape, potentially broadcasting along dimensions.
    """

    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        """
        Expand the input tensor to the specified shape.

        Args:
            x (LazyBuffer): Input tensor to be expanded
            shape (Tuple[int, ...]): Target shape to expand to

        Returns:
            LazyBuffer: Expanded tensor
        """
        # Store the original input shape for backward pass
        self.input_shape = x.shape
        return x.expand(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient by reducing the expanded tensor.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient reduced to the original input shape
        """
        return grad_output.r(ReduceOps.SUM, self.input_shape)


class Reshape(Function):
    """
    Tensor reshaping operation.

    Changes the shape of a tensor without modifying its data.
    """

    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        """
        Reshape the input tensor to the specified shape.

        Args:
            x (LazyBuffer): Input tensor to be reshaped
            shape (Tuple[int, ...]): Target shape to reshape to

        Returns:
            LazyBuffer: Reshaped tensor
        """
        # Store the original input shape for backward pass
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Restore the gradient to the original input shape.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient reshaped to the original input shape
        """
        return grad_output.reshape(self.input_shape)


class Permute(Function):
    """
    Tensor permutation operation.

    Reorders the dimensions of a tensor according to the specified order.
    """

    def forward(self, x: LazyBuffer, order: Tuple[int, ...]) -> LazyBuffer:
        """
        Permute the dimensions of the input tensor.

        Args:
            x (LazyBuffer): Input tensor to be permuted
            order (Tuple[int, ...]): New order of dimensions

        Returns:
            LazyBuffer: Permuted tensor
        """
        # Store the input dimension order for backward pass
        self.input_order = order
        return x.permute(order)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Restore the gradient to the original dimension order.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient permuted back to the original order
        """
        return grad_output.permute(argsort(self.input_order))


class Pad(Function):
    """
    Padding operation for tensors.

    Adds padding to the input tensor along specified dimensions.
    """

    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[int, int], ...]) -> LazyBuffer:
        """
        Apply padding to the input tensor.

        Args:
            x (LazyBuffer): Input tensor to be padded
            arg (List[Tuple[int, int]]): Padding specification for each dimension

        Returns:
            LazyBuffer: Padded tensor
        """
        # Store the new tensor shape after padding
        self.narg = tuple([(p[0], s + p[0]) for s, p in zip(x.shape, arg)])
        return x.pad(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient by shrinking the padded tensor.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient with padding removed
        """
        return grad_output.shrink(self.narg)


class Shrink(Function):
    """
    Shrinking operation for tensors.

    Reduces the size of a tensor along specified dimensions.
    """

    def forward(self, x: LazyBuffer, arg: Tuple[Tuple[int, int]]) -> LazyBuffer:
        """
        Shrink the input tensor.

        Args:
            x (LazyBuffer): Input tensor to be shrunk
            arg (List[Tuple[int, int]]): Shrinking specification for each dimension

        Returns:
            LazyBuffer: Shrunk tensor
        """
        # Store the new tensor shape after shrinking
        self.narg = tuple([(p[0], s - p[1]) for s, p in zip(x.shape, arg)])
        return x.shrink(arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient by padding the shrunk tensor.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient with padding added

        Raises:
            AssertionError: If symbolic shrink is used (non-integer padding)
        """
        # Ensure all padding values are integers
        assert all(
            isinstance(x[0], int) and isinstance(x[1], int) for x in self.narg
        ), "Symbolic shrink does not support backward pass"

        return grad_output.pad(self.narg)


class Flip(Function):
    """
    Tensor flipping operation.

    Reverses the order of elements along specified axes.
    """

    def forward(self, x: LazyBuffer, axis: Tuple[int, List[int]]) -> LazyBuffer:
        """
        Flip the input tensor along specified axes.

        Args:
            x (LazyBuffer): Input tensor to be flipped
            axis (Union[int, List[int]]): Axis or list of axes to flip

        Returns:
            LazyBuffer: Flipped tensor
        """
        # Compute stride for each dimension based on flip axes
        self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
        return x.stride(self.arg)

    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        """
        Compute gradient with the same flipping applied.

        Args:
            grad_output (LazyBuffer): Incoming gradient

        Returns:
            LazyBuffer: Gradient with same stride as forward pass
        """
        return grad_output.stride(self.arg)
