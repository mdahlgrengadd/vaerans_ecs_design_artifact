"""Arena allocator and TensorRef for zero-copy memory management.

The Arena provides a contiguous memory buffer with bump allocation strategy.
TensorRefs are lightweight handles (offset, shape, dtype) pointing into the arena,
enabling zero-copy tensor operations without data duplication.

Key Features:
- Zero-copy: Components store TensorRefs, not arrays
- Aligned allocation: Respects dtype alignment requirements
- Generation counter: Detects stale TensorRefs after arena reset
- Subrefs: Create views into existing allocations for batching

Example:
    >>> arena = Arena(size_bytes=1024)
    >>> ref = arena.alloc_tensor((10, 10), np.float32)
    >>> arr = arena.view(ref)
    >>> arr[:] = 1.0  # Modify in-place
    >>> arena.reset()  # Clear arena for reuse
    >>> # arena.view(ref)  # Would raise ValueError: stale ref
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorRef:
    """Lightweight handle pointing to tensor data in an Arena.

    Attributes:
        offset: Byte offset into arena buffer
        shape: Tensor dimensions
        dtype: NumPy data type
        strides: Byte strides for each dimension (enables subrefs)
        generation: Arena generation counter (for staleness detection)
    """

    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    strides: tuple[int, ...]
    generation: int

    def __post_init__(self) -> None:
        """Validate TensorRef fields."""
        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")
        if len(self.shape) != len(self.strides):
            raise ValueError(
                f"shape and strides must have same length: "
                f"shape={self.shape}, strides={self.strides}"
            )
        if self.generation < 0:
            raise ValueError(f"generation must be non-negative, got {self.generation}")

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        """Total number of bytes (including stride gaps)."""
        if self.size == 0:
            return 0
        # Calculate span: from first to last element
        last_offset = sum((s - 1) * st for s, st in zip(self.shape, self.strides))
        return last_offset + self.dtype.itemsize

    def subref(self, slices: tuple[slice | int, ...]) -> TensorRef:
        """Create a view into this TensorRef (for batching).

        Args:
            slices: Tuple of slices or integers for indexing

        Returns:
            New TensorRef pointing to the sliced region

        Example:
            >>> # Allocate batch of 4 images
            >>> ref = arena.alloc_tensor((4, 3, 256, 256), np.float32)
            >>> # Get subref for second image
            >>> img1_ref = ref.subref((1, slice(None), slice(None), slice(None)))
        """
        # Normalize slices
        normalized: list[slice | int] = []
        for i, s in enumerate(slices):
            if isinstance(s, int):
                normalized.append(s)
            elif isinstance(s, slice):
                normalized.append(s)
            else:
                raise TypeError(f"Invalid slice type: {type(s)}")

        # Pad with full slices if needed
        while len(normalized) < len(self.shape):
            normalized.append(slice(None))

        # Calculate new offset, shape, strides
        new_offset = self.offset
        new_shape: list[int] = []
        new_strides: list[int] = []

        for i, (s, size, stride) in enumerate(zip(normalized, self.shape, self.strides)):
            if isinstance(s, int):
                # Integer index: skip this dimension
                if s < 0:
                    s = size + s
                if not (0 <= s < size):
                    raise IndexError(f"Index {s} out of bounds for dimension {i} with size {size}")
                new_offset += s * stride
            elif isinstance(s, slice):
                start, stop, step = s.indices(size)
                if step != 1:
                    raise NotImplementedError("Strided slices not supported")
                new_shape.append(stop - start)
                new_strides.append(stride)
                new_offset += start * stride

        return TensorRef(
            offset=new_offset,
            shape=tuple(new_shape),
            dtype=self.dtype,
            strides=tuple(new_strides),
            generation=self.generation,
        )


class Arena:
    """Contiguous memory allocator with bump allocation strategy.

    The Arena manages a pre-allocated bytearray buffer and allocates tensors
    sequentially. All allocations are aligned to dtype requirements.

    Attributes:
        size: Total arena size in bytes
        offset: Current allocation offset (bump pointer)
        generation: Incremented on reset() to invalidate old TensorRefs

    Example:
        >>> arena = Arena(size_bytes=1024)
        >>> ref1 = arena.alloc_tensor((10,), np.float32)
        >>> ref2 = arena.alloc_tensor((5, 5), np.uint8)
        >>> print(f"Allocated {arena.offset} / {arena.size} bytes")
    """

    def __init__(self, size_bytes: int):
        """Create arena with specified size.

        Args:
            size_bytes: Total size in bytes (should be >> expected usage)
        """
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be positive, got {size_bytes}")

        self._buffer = bytearray(size_bytes)
        self._size = size_bytes
        self._offset = 0
        self._generation = 0

    @property
    def size(self) -> int:
        """Total arena size in bytes."""
        return self._size

    @property
    def offset(self) -> int:
        """Current allocation offset (bytes used)."""
        return self._offset

    @property
    def generation(self) -> int:
        """Current generation counter."""
        return self._generation

    @property
    def available(self) -> int:
        """Remaining bytes available for allocation."""
        return self._size - self._offset

    def reset(self) -> None:
        """Reset arena for reuse. Invalidates all existing TensorRefs.

        After reset, attempting to view old TensorRefs will raise ValueError.
        """
        self._offset = 0
        self._generation += 1

    def alloc_tensor(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | type | str,
    ) -> TensorRef:
        """Allocate a tensor in the arena.

        Args:
            shape: Tensor dimensions
            dtype: NumPy data type

        Returns:
            TensorRef handle to the allocated tensor

        Raises:
            ValueError: If allocation would exceed arena size
        """
        # Normalize dtype
        dt = np.dtype(dtype)

        # Calculate required bytes
        size = int(np.prod(shape))
        nbytes = size * dt.itemsize

        # Align offset to dtype alignment (typically 1, 4, or 8 bytes)
        alignment = dt.alignment
        aligned_offset = (self._offset + alignment - 1) // alignment * alignment

        # Check if we have space
        end_offset = aligned_offset + nbytes
        if end_offset > self._size:
            raise ValueError(
                f"Arena out of memory: need {nbytes} bytes at offset {aligned_offset}, "
                f"but arena size is {self._size} (available: {self.available})"
            )

        # Calculate default strides (C-contiguous)
        strides = []
        stride = dt.itemsize
        for dim_size in reversed(shape):
            strides.append(stride)
            stride *= dim_size
        strides.reverse()

        # Create TensorRef
        ref = TensorRef(
            offset=aligned_offset,
            shape=shape,
            dtype=dt,
            strides=tuple(strides),
            generation=self._generation,
        )

        # Update offset
        self._offset = end_offset

        return ref

    def view(self, ref: TensorRef) -> np.ndarray:
        """Get a NumPy array view of a TensorRef.

        Args:
            ref: TensorRef to view

        Returns:
            NumPy array backed by arena memory (zero-copy)

        Raises:
            ValueError: If TensorRef is stale (from previous generation)
        """
        # Validate generation
        if ref.generation != self._generation:
            raise ValueError(
                f"Stale TensorRef: arena was reset (current generation {self._generation}, "
                f"ref is from generation {ref.generation})"
            )

        # Validate bounds
        end_offset = ref.offset + ref.nbytes
        if end_offset > self._size:
            raise ValueError(
                f"TensorRef out of bounds: offset={ref.offset}, nbytes={ref.nbytes}, "
                f"arena size={self._size}"
            )

        # Create NumPy array view
        # Use ndarray constructor with buffer protocol
        arr = np.ndarray(
            shape=ref.shape,
            dtype=ref.dtype,
            buffer=self._buffer,
            offset=ref.offset,
            strides=ref.strides,
        )

        return arr

    def copy_tensor(self, arr: np.ndarray) -> TensorRef:
        """Allocate tensor and copy data from array.

        Args:
            arr: NumPy array to copy

        Returns:
            TensorRef pointing to the copied data
        """
        ref = self.alloc_tensor(arr.shape, arr.dtype)
        self.view(ref)[:] = arr
        return ref

    def __repr__(self) -> str:
        return (
            f"Arena(size={self._size}, offset={self._offset}, "
            f"generation={self._generation}, available={self.available})"
        )
