"""Tests for Hadamard4 transform system."""

import numpy as np
import pytest

from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4


class TestHadamard4System:
    """Tests for Hadamard4 system."""

    def test_system_creation_encode(self) -> None:
        """Test Hadamard4 creation in encode mode."""
        system = Hadamard4(mode="encode")
        assert system.mode == "encode"

    def test_system_creation_decode(self) -> None:
        """Test Hadamard4 creation in decode mode."""
        system = Hadamard4(mode="decode")
        assert system.mode == "decode"

    def test_required_components_encode(self) -> None:
        """Test required components in encode mode."""
        system = Hadamard4(mode="encode")
        assert system.required_components() == [Latent4]

    def test_required_components_decode(self) -> None:
        """Test required components in decode mode."""
        system = Hadamard4(mode="decode")
        assert system.required_components() == [YUVW4]

    def test_produced_components_encode(self) -> None:
        """Test produced components in encode mode."""
        system = Hadamard4(mode="encode")
        assert system.produced_components() == [YUVW4]

    def test_produced_components_decode(self) -> None:
        """Test produced components in decode mode."""
        system = Hadamard4(mode="decode")
        assert system.produced_components() == [Latent4]

    def test_forward_transform(self) -> None:
        """Test forward Hadamard transform."""
        world = World()
        eid = world.new_entity()

        # Create latent tensor
        z_ref = world.arena.alloc_tensor((4, 8, 8), np.float32)
        z = world.arena.view(z_ref)

        # Fill with known values
        z[:] = 1.0
        world.add_component(eid, Latent4(z=z_ref))

        # Apply forward transform
        system = Hadamard4(mode="encode")
        assert system.can_run(world, eid)
        system.run(world, [eid])

        # Verify output
        assert world.has_component(eid, YUVW4)
        yuvw = world.get_component(eid, YUVW4)
        t = world.arena.view(yuvw.t)

        # With input all 1s, H4 @ [1,1,1,1]^T = [2, 0, 0, 0]^T (due to normalization)
        assert t.shape == (4, 8, 8)
        assert np.allclose(t[0, :, :], 2.0)  # Y channel ≈ 2
        assert np.allclose(t[1:, :, :], 0.0)  # U, V, W channels ≈ 0

    def test_inverse_transform(self) -> None:
        """Test inverse Hadamard transform (recovery)."""
        world = World()
        eid = world.new_entity()

        # Create transformed tensor
        t_ref = world.arena.alloc_tensor((4, 8, 8), np.float32)
        t = world.arena.view(t_ref)

        # Set transformed values (from forward pass with input all 1s -> [2, 0, 0, 0])
        t[0, :, :] = 2.0  # Y channel
        t[1:, :, :] = 0.0  # U, V, W channels
        world.add_component(eid, YUVW4(t=t_ref))

        # Apply inverse transform
        system = Hadamard4(mode="decode")
        assert system.can_run(world, eid)
        system.run(world, [eid])

        # Verify recovery
        assert world.has_component(eid, Latent4)
        latent = world.get_component(eid, Latent4)
        z = world.arena.view(latent.z)

        # Should recover original values (all 1s)
        assert z.shape == (4, 8, 8)
        assert np.allclose(z, 1.0, atol=1e-6)

    def test_round_trip(self) -> None:
        """Test round-trip encode-decode recovery."""
        world = World()

        # Create initial latent tensor with random values
        original_data = np.random.randn(4, 16, 16).astype(np.float32)

        eid1 = world.new_entity()
        z_ref = world.arena.copy_tensor(original_data)
        world.add_component(eid1, Latent4(z=z_ref))

        # Forward transform
        encode_system = Hadamard4(mode="encode")
        encode_system.run(world, [eid1])

        # Extract transformed data
        yuvw = world.get_component(eid1, YUVW4)
        transformed_data = world.arena.view(yuvw.t).copy()

        # Create new entity with transformed data
        eid2 = world.new_entity()
        t_ref = world.arena.copy_tensor(transformed_data)
        world.add_component(eid2, YUVW4(t=t_ref))

        # Inverse transform
        decode_system = Hadamard4(mode="decode")
        decode_system.run(world, [eid2])

        # Verify recovery
        latent = world.get_component(eid2, Latent4)
        recovered_data = world.arena.view(latent.z)

        # Should match original within numerical precision
        assert np.allclose(recovered_data, original_data, atol=1e-5)

    def test_orthogonality(self) -> None:
        """Test that Hadamard matrix is orthogonal (H^T @ H = I)."""
        system = Hadamard4()
        H = system._H4

        # Check orthogonality
        product = H.T @ H
        identity = np.eye(4)

        assert np.allclose(product, identity, atol=1e-6)

    def test_multiple_entities(self) -> None:
        """Test processing multiple entities in one call."""
        world = World()

        eids = []
        original_data = []

        # Create 5 entities with different latent data
        for i in range(5):
            eid = world.new_entity()
            data = np.full((4, 8, 8), float(i), dtype=np.float32)
            ref = world.arena.copy_tensor(data)
            world.add_component(eid, Latent4(z=ref))
            eids.append(eid)
            original_data.append(data)

        # Process all at once
        system = Hadamard4(mode="encode")
        system.run(world, eids)

        # Verify each entity was transformed
        for i, eid in enumerate(eids):
            assert world.has_component(eid, YUVW4)
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)

            # With input all i, Y channel should be 2*i (due to normalization), others 0
            assert np.allclose(t[0, :, :], float(i) * 2.0, atol=1e-6)
            assert np.allclose(t[1:, :, :], 0.0, atol=1e-6)

    def test_different_shapes(self) -> None:
        """Test Hadamard with different spatial dimensions."""
        for h in [4, 8, 16, 32]:
            world = World()
            eid = world.new_entity()

            # Create latent with different shape
            z_ref = world.arena.alloc_tensor((4, h, h), np.float32)
            z = world.arena.view(z_ref)
            z[:] = 1.0
            world.add_component(eid, Latent4(z=z_ref))

            # Transform
            system = Hadamard4(mode="encode")
            system.run(world, [eid])

            # Verify shape preservation
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)
            assert t.shape == (4, h, h)

            # Verify correctness (with normalization, input all 1s gives [2, 0, 0, 0])
            assert np.allclose(t[0, :, :], 2.0)
            assert np.allclose(t[1:, :, :], 0.0)

    def test_batch_method(self) -> None:
        """Test vectorized batch transform method."""
        system = Hadamard4()

        # Create test data
        data = np.ones((4, 8, 8), dtype=np.float32)

        # Test batch transform
        result = system._hadamard_batch(data, system._H4)

        assert result.shape == data.shape
        # With input all 1s, output should be [2, 0, 0, 0] (with normalization)
        assert np.allclose(result[0, :, :], 2.0)
        assert np.allclose(result[1:, :, :], 0.0)

    def test_precision(self) -> None:
        """Test numerical precision of transforms."""
        world = World()
        eid = world.new_entity()

        # Create latent with diverse values
        data = np.random.randn(4, 16, 16).astype(np.float32)
        z_ref = world.arena.copy_tensor(data)
        world.add_component(eid, Latent4(z=z_ref))

        # Forward transform
        encode_system = Hadamard4(mode="encode")
        encode_system.run(world, [eid])
        yuvw = world.get_component(eid, YUVW4)
        transformed = world.arena.view(yuvw.t).copy()

        # Verify orthogonality property: ||Hx|| = ||x||
        original_norm = np.linalg.norm(data)
        transformed_norm = np.linalg.norm(transformed)

        # Norms should be equal (up to numerical precision)
        assert np.allclose(original_norm, transformed_norm, atol=1e-4)

    def test_mode_forward_alias(self) -> None:
        """Test that 'forward' mode works like 'encode'."""
        world = World()
        eid = world.new_entity()

        z_ref = world.arena.alloc_tensor((4, 8, 8), np.float32)
        z = world.arena.view(z_ref)
        z[:] = 1.0
        world.add_component(eid, Latent4(z=z_ref))

        system = Hadamard4(mode="forward")
        assert system.mode == "forward"
        assert system.required_components() == [Latent4]
        assert system.produced_components() == [YUVW4]

    def test_mode_inverse_alias(self) -> None:
        """Test that 'inverse' mode works like 'decode'."""
        world = World()
        eid = world.new_entity()

        t_ref = world.arena.alloc_tensor((4, 8, 8), np.float32)
        t = world.arena.view(t_ref)
        t[0, :, :] = 1.0
        t[1:, :, :] = 0.0
        world.add_component(eid, YUVW4(t=t_ref))

        system = Hadamard4(mode="inverse")
        assert system.mode == "inverse"
        assert system.required_components() == [YUVW4]
        assert system.produced_components() == [Latent4]
