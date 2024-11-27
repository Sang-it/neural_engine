import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, List, Any, Iterator


@dataclass(frozen=True, order=True)
class DType:
    """
    Represents a data type with priority, size, name, and NumPy type.

    Attributes:
        priority (int): Comparison priority of the data type
        itemsize (int): Size of the item in bytes
        name (str): Human-readable name of the data type
        np (type): Corresponding NumPy type
        sz (int, optional): Additional size parameter. Defaults to 1.
    """

    priority: int
    itemsize: int
    name: str
    np: type
    sz: int = 1


class Dtypes:
    """
    Collection of predefined data types with utility methods.
    """

    # Integer types
    int8 = DType(1, 1, "char", np.int8)
    int16 = DType(3, 2, "short", np.int16)
    int32 = DType(5, 4, "int", np.int32)
    int64 = DType(7, 8, "long", np.int64)

    # Unsigned integer types
    uint8 = DType(2, 1, "unsigned char", np.uint8)
    uint16 = DType(4, 2, "unsigned short", np.uint16)
    uint32 = DType(6, 4, "unsigned int", np.uint32)
    uint64 = DType(8, 8, "unsigned long", np.uint64)

    # Float types
    bool = DType(0, 1, "bool", np.bool_)
    float16 = DType(9, 2, "half", np.float16)
    float32 = DType(10, 4, "float", np.float32)
    float64 = DType(11, 8, "double", np.float64)

    # Alias definitions
    half = float16
    float = float32
    double = float64

    @staticmethod
    def is_int(x: DType) -> bool:  # pyright: ignore
        """
        Check if the given dtype is an integer type.

        Args:
            x (DType): Data type to check

        Returns:
            bool: True if the type is an integer, False otherwise
        """
        return x in (
            Dtypes.int8,
            Dtypes.int16,
            Dtypes.int32,
            Dtypes.int64,
            Dtypes.uint8,
            Dtypes.uint16,
            Dtypes.uint32,
            Dtypes.uint64,
        )

    @staticmethod
    def is_float(x: bool) -> bool:  # pyright: ignore
        """
        Check if the given dtype is a float type.

        Args:
            x (DType): Data type to check

        Returns:
            bool: True if the type is a float, False otherwise
        """
        return x in (Dtypes.float16, Dtypes.float32, Dtypes.float64)

    @staticmethod
    def is_unsigned(x: DType) -> bool:  # pyright: ignore
        """
        Check if the given dtype is an unsigned integer type.

        Args:
            x (DType): Data type to check

        Returns:
            bool: True if the type is unsigned, False otherwise
        """
        return x in (Dtypes.uint8, Dtypes.uint16, Dtypes.uint32, Dtypes.uint64)

    @staticmethod
    def from_np(x) -> DType:
        """
        Convert a NumPy dtype to the corresponding DType.

        Args:
            x: NumPy dtype or type

        Returns:
            DType: Corresponding DType instance
        """
        return DTYPES_DICT[np.dtype(x).name]


# Create a dictionary of all defined dtypes
DTYPES_DICT = {
    k: v
    for k, v in Dtypes.__dict__.items()
    if not k.startswith("__") and not callable(v) and v.__class__ is not staticmethod
}


def dedup(x: List[Any]) -> List[Any]:
    """
    Remove duplicates from a list while preserving order.

    Args:
        x (List[Any]): Input list

    Returns:
        List[Any]: List with duplicates removed
    """
    return list(dict.fromkeys(x))


def argfix(*x: Union[Tuple[Any], Any]) -> Tuple[Any, ...]:
    """
    Standardize input arguments, converting single tuple/list to its elements.

    Args:
        *x: Variable number of arguments

    Returns:
        Tuple[Any, ...]: Processed arguments
    """
    return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x


def make_pair(x: Union[int, Any], cnt: int = 2) -> Tuple[Any, ...]:
    """
    Create a tuple of a given length with the same value.

    Args:
        x (Union[int, Any]): Value to repeat
        cnt (int, optional): Number of repetitions. Defaults to 2.

    Returns:
        Tuple[Any, ...]: Tuple with repeated values
    """
    return (x,) * cnt if isinstance(x, int) else x


def flatten(lst: Iterator) -> List[Any]:
    """
    Flatten a list of lists into a single list.

    Args:
        lst (List[List[Any]]): List of lists

    Returns:
        List[Any]: Flattened list
    """
    return [item for sublist in lst for item in sublist]


def argsort(x: Any) -> Any:
    """
    Return indices that would sort the input sequence.

    Args:
        x: Input sequence

    Returns:
        Any: Sorted indices of the same type as input
    """
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


def all_int(t: Tuple[Any, ...]) -> bool:
    """
    Check if all elements in a tuple/sequence are integers.

    Args:
        t (Tuple[Any, ...]): Input sequence

    Returns:
        bool: True if all elements are integers, False otherwise
    """
    return all(isinstance(s, int) for s in t)


def round_up(num: int, amt: int) -> int:
    """
    Round up a number to the nearest multiple of a given amount.

    Args:
        num (int): Number to round up
        amt (int): Rounding increment

    Returns:
        int: Rounded up number
    """
    return (num + amt - 1) // amt * amt
