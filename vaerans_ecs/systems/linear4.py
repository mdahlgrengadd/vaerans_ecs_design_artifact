"""Generic 4-channel linear transform system."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pathlib import Path

from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World


class LinearTransform4(System):
    """Apply a 4x4 linear transform to 4-channel tensors.

    This system is component-agnostic: you provide input/output component
    types and the attribute names that store the TensorRef.
    """

    def __init__(
        self,
        forward: np.ndarray,
        inverse: np.ndarray | None = None,
        *,
        input_component: type,
        input_attr: str,
        output_component: type,
        output_attr: str,
        orthonormal: bool = False,
        output_dtype: np.dtype | None = None,
        mode: Literal["encode", "decode", "forward", "inverse"] = "forward",
    ) -> None:
        super().__init__(mode=mode)
        self.input_component = input_component
        self.output_component = output_component
        self.input_attr = input_attr
        self.output_attr = output_attr
        self.output_dtype = output_dtype

        self._forward = self._validate_matrix(forward)
        if inverse is None:
            if orthonormal:
                self._inverse = self._forward.T
            else:
                self._inverse = np.linalg.inv(self._forward)
        else:
            self._inverse = self._validate_matrix(inverse)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        input_component: type,
        input_attr: str,
        output_component: type,
        output_attr: str,
        forward_key: str = "klt_forward",
        inverse_key: str = "klt_inverse",
        orthonormal: bool = True,
        output_dtype: np.dtype | None = None,
        mode: Literal["encode", "decode", "forward", "inverse"] = "forward",
    ) -> "LinearTransform4":
        """Create a LinearTransform4 from matrices stored in a .npz file."""
        data = np.load(str(path))
        forward = data[forward_key]
        inverse = data[inverse_key] if inverse_key in data else None
        return cls(
            forward,
            inverse=inverse,
            input_component=input_component,
            input_attr=input_attr,
            output_component=output_component,
            output_attr=output_attr,
            orthonormal=orthonormal,
            output_dtype=output_dtype,
            mode=mode,
        )


def linear4_pair_from_npz(
    path: str | Path,
    *,
    input_component: type,
    input_attr: str,
    output_component: type,
    output_attr: str,
    forward_key: str = "klt_forward",
    inverse_key: str = "klt_inverse",
    orthonormal: bool = True,
    output_dtype: np.dtype | None = None,
) -> tuple[LinearTransform4, LinearTransform4]:
    """Load matrices from .npz and return (forward, inverse) systems."""
    data = np.load(str(path))
    forward = data[forward_key]
    inverse = data[inverse_key] if inverse_key in data and inverse_key else None
    forward_sys = LinearTransform4(
        forward,
        inverse=inverse,
        input_component=input_component,
        input_attr=input_attr,
        output_component=output_component,
        output_attr=output_attr,
        orthonormal=orthonormal,
        output_dtype=output_dtype,
        mode="forward",
    )
    inverse_sys = LinearTransform4(
        forward,
        inverse=inverse,
        input_component=input_component,
        input_attr=input_attr,
        output_component=output_component,
        output_attr=output_attr,
        orthonormal=orthonormal,
        output_dtype=output_dtype,
        mode="inverse",
    )
    return forward_sys, inverse_sys

    def required_components(self) -> list[type]:
        if self.mode in ("encode", "forward"):
            return [self.input_component]
        return [self.output_component]

    def produced_components(self) -> list[type]:
        if self.mode in ("encode", "forward"):
            return [self.output_component]
        return [self.input_component]

    def run(self, world: World, eids: list[int]) -> None:
        if self.mode in ("encode", "forward"):
            self._apply(world, eids, self.input_component, self.input_attr,
                        self.output_component, self.output_attr, self._forward)
        else:
            self._apply(world, eids, self.output_component, self.output_attr,
                        self.input_component, self.input_attr, self._inverse)

    @staticmethod
    def _validate_matrix(mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float32)
        if arr.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {arr.shape}")
        return arr

    def _apply(
        self,
        world: World,
        eids: list[int],
        in_component: type,
        in_attr: str,
        out_component: type,
        out_attr: str,
        matrix: np.ndarray,
    ) -> None:
        for eid in eids:
            comp = world.get_component(eid, in_component)
            data = world.arena.view(getattr(comp, in_attr))
            if data.ndim != 3 or data.shape[0] != 4:
                raise ValueError(f"Expected (4, H, W) tensor, got {data.shape}")

            data_f = data.astype(np.float32, copy=False)
            transformed = np.tensordot(matrix, data_f, axes=([1], [0]))

            if self.output_dtype is None:
                if np.issubdtype(data.dtype, np.integer):
                    out_dtype = np.float32
                else:
                    out_dtype = data.dtype
            else:
                out_dtype = self.output_dtype

            out_ref = world.arena.copy_tensor(transformed.astype(out_dtype, copy=False))
            world.add_component(eid, out_component(**{out_attr: out_ref}))
