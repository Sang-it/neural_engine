from enum import Enum, auto
from typing import List


class UnaryOps(Enum):
    """
    Enumeration of unary operations supported in the computational graph.

    Unary operations are single-input mathematical or type transformation operations.
    """

    NOOP = auto()  # No operation (identity)
    EXP2 = auto()  # Exponential base 2 (2^x)
    LOG2 = auto()  # Logarithm base 2 (log2(x))
    CAST = auto()  # Type conversion
    SIN = auto()  # Sine function
    SQRT = auto()  # Square root
    RECIP = auto()  # Reciprocal (1/x)
    NEG = auto()  # Negation (-x)


class BinaryOps(Enum):
    """
    Enumeration of binary operations supported in the computational graph.

    Binary operations are two-input mathematical or comparison operations.
    """

    ADD = auto()  # Addition
    SUB = auto()  # Subtraction
    MUL = auto()  # Multiplication
    DIV = auto()  # Division
    MAX = auto()  # Maximum of two values
    MOD = auto()  # Modulo operation
    CMPLT = auto()  # Compare less than


class ReduceOps(Enum):
    """
    Enumeration of reduction operations for multi-dimensional arrays.

    Reduction operations collapse dimensions while preserving specific properties.
    """

    SUM = auto()  # Sum reduction
    MAX = auto()  # Maximum reduction


class TernaryOps(Enum):
    """
    Enumeration of ternary operations supported in the computational graph.

    Ternary operations involve three inputs.
    """

    MULACC = auto()  # Multiply-accumulate
    WHERE = auto()  # Conditional selection


class MovementOps(Enum):
    """
    Enumeration of array movement and transformation operations.

    These operations change the shape, layout, or structure of arrays without
    changing their underlying data.
    """

    RESHAPE = auto()  # Change array shape
    PERMUTE = auto()  # Reorder array dimensions
    EXPAND = auto()  # Broadcast array
    PAD = auto()  # Add padding to array
    SHRINK = auto()  # Reduce array size
    STRIDE = auto()  # Create strided view of array


class LoadOps(Enum):
    """
    Enumeration of array loading and initialization operations.

    These operations create new arrays with specific characteristics.
    """

    EMPTY = auto()  # Create uninitialized array
    RAND = auto()  # Create random array
    CONST = auto()  # Create constant-filled array
    FROM = auto()  # Create from existing data
    CONTIGUOUS = auto()  # Ensure contiguous memory layout
    CUSTOM = auto()  # Custom loading method


class Device:
    """
    Utility class for device management and canonicalization.

    Currently supports only CPU device, but designed to be extensible.
    """

    DEFAULT = "CPU"
    _buffers: List[str] = ["CPU"]

    @staticmethod
    def canonicalize() -> str:
        """
        Canonicalize the device specification.

        Returns:
            str: Canonicalized device name (always "CPU" in this implementation)
        """
        return "CPU"
