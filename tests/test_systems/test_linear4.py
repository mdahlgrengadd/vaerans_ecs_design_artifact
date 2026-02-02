"""Tests for LinearTransform4 system and KLT utilities."""

from __future__ import annotations

import numpy as np

from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.world import World
from vaerans_ecs.eval.linear4 import (
    covariance_4ch_from_arrays,
    klt_from_covariance,
)
from vaerans_ecs.systems.linear4 import LinearTransform4


def _make_latent(data: np.ndarray) -> tuple[World, int]:
    world = World()
    eid = world.new_entity()
    ref = world.arena.copy_tensor(data.astype(np.float32))
    world.add_component(eid, Latent4(z=ref))
    return world, eid


def test_linear4_identity_roundtrip() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 8, 8)).astype(np.float32)
    world, eid = _make_latent(data)

    identity = np.eye(4, dtype=np.float32)
    fwd = LinearTransform4(
        identity,
        input_component=Latent4,
        input_attr="z",
        output_component=YUVW4,
        output_attr="t",
        orthonormal=True,
        mode="forward",
    )
    fwd.run(world, [eid])

    yuvw = world.get_component(eid, YUVW4)
    out = world.arena.view(yuvw.t)
    assert np.allclose(out, data, atol=1e-6)

    inv = LinearTransform4(
        identity,
        input_component=Latent4,
        input_attr="z",
        output_component=YUVW4,
        output_attr="t",
        orthonormal=True,
        mode="inverse",
    )
    inv.run(world, [eid])

    latent = world.get_component(eid, Latent4)
    recovered = world.arena.view(latent.z)
    assert np.allclose(recovered, data, atol=1e-6)


def test_linear4_orthonormal_roundtrip() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 4)).astype(np.float32)
    q, _ = np.linalg.qr(a)

    data = rng.standard_normal((4, 8, 8)).astype(np.float32)
    world, eid = _make_latent(data)

    fwd = LinearTransform4(
        q,
        input_component=Latent4,
        input_attr="z",
        output_component=YUVW4,
        output_attr="t",
        orthonormal=True,
        mode="forward",
    )
    inv = LinearTransform4(
        q,
        input_component=Latent4,
        input_attr="z",
        output_component=YUVW4,
        output_attr="t",
        orthonormal=True,
        mode="inverse",
    )

    fwd.run(world, [eid])
    inv.run(world, [eid])

    recovered = world.arena.view(world.get_component(eid, Latent4).z)
    assert np.allclose(recovered, data, atol=1e-5)


def test_klt_reduces_correlation() -> None:
    rng = np.random.default_rng(2)
    h, w = 64, 64
    s1 = rng.standard_normal((h, w))
    s2 = rng.standard_normal((h, w))
    s3 = rng.standard_normal((h, w))
    s4 = rng.standard_normal((h, w))

    c0 = s1 + 0.5 * s2
    c1 = s1 + 0.5 * s3
    c2 = s1 + 0.5 * s4
    c3 = 0.8 * s1 + 0.2 * s2
    data = np.stack([c0, c1, c2, c3], axis=0).astype(np.float32)

    cov = covariance_4ch_from_arrays([data])
    forward, inverse, _ = klt_from_covariance(cov)

    before = np.corrcoef(data.reshape(4, -1))
    transformed = np.tensordot(forward, data, axes=([1], [0]))
    after = np.corrcoef(transformed.reshape(4, -1))

    mask = ~np.eye(4, dtype=bool)
    before_mean = float(np.mean(np.abs(before[mask])))
    after_mean = float(np.mean(np.abs(after[mask])))

    assert after_mean < before_mean * 0.6
