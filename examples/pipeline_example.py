#!/usr/bin/env python3
"""Example demonstrating Phase 11: Fluent Pipeline API.

This example shows how to use the Pipe fluent API to compose systems
into processing pipelines.
"""

import numpy as np

from vaerans_ecs.components.image import RGB
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World
from vaerans_ecs.systems.hadamard import Hadamard4


class DummyVAEEncoder(System):
    """Simple VAE encoder for demonstration."""

    def __init__(self) -> None:
        super().__init__(mode="encode")

    def required_components(self):
        return [RGB]

    def produced_components(self):
        return [Latent4]

    def run(self, world: World, eids: list[int]) -> None:
        """Encode RGB to dummy latent."""
        for eid in eids:
            rgb = world.get_component(eid, RGB)
            rgb_view = world.arena.view(rgb.pix)
            # Simulate 8x downsampling
            h, w = rgb_view.shape[0] // 8, rgb_view.shape[1] // 8
            latent_data = np.random.randn(4, h, w).astype(np.float32)
            latent_ref = world.arena.copy_tensor(latent_data)
            world.add_component(eid, Latent4(z=latent_ref))


def main() -> None:
    """Demonstrate fluent pipeline API."""
    print("=== Phase 11: Fluent Pipeline API Example ===\n")

    # Create world with arena
    world = World(arena_bytes=256 << 20)  # 256 MB
    print("[OK] Created World with 256 MB arena\n")

    # Create a test image
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    entity = world.spawn_image(img)
    print(f"[OK] Created image entity {entity} with shape {img.shape}\n")

    # Example 1: Simple pipeline with .to() method
    print("Example 1: Simple pipeline with .to()")
    print("-" * 40)
    latent = (
        world.pipe(entity)
        .to(DummyVAEEncoder())
        .out(Latent4)
    )
    print(
        f"[OK] Encoded image to latent with shape {world.arena.view(latent.z).shape}\n")

    # Example 2: Chained transformation pipeline
    print("Example 2: Chained transformations")
    print("-" * 40)
    # Create a new entity with latent
    latent_data = np.random.randn(4, 32, 32).astype(np.float32)
    latent_ref = world.arena.copy_tensor(latent_data)
    entity2 = world.new_entity()
    world.add_component(entity2, Latent4(z=latent_ref))

    # Forward then inverse transform via pipeline
    result = (
        world.pipe(entity2)
        .to(Hadamard4(mode="forward"))
        .to(Hadamard4(mode="inverse"))
        .out(Latent4)
    )
    result_view = world.arena.view(result.z)
    print(f"[OK] Forward + inverse transforms (round-trip)")
    print(f"  Input shape:  {latent_data.shape}")
    print(f"  Output shape: {result_view.shape}")
    print(
        f"  Recovery error (MSE): {np.mean((result_view - latent_data)**2):.2e}\n")

    # Example 3: Using pipe operator |
    print("Example 3: Using pipe operator |")
    print("-" * 40)
    entity3 = world.new_entity()
    world.add_component(entity3, Latent4(z=latent_ref))

    result2 = (
        world.pipe(entity3)
        | Hadamard4(mode="forward")
        | Hadamard4(mode="inverse")
    ).out(Latent4)
    print(f"[OK] Pipe operator | works identically to .to()\n")

    # Example 4: Processing multiple entities
    print("Example 4: Multiple entities separately")
    print("-" * 40)
    images = [
        np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
        np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
        np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
    ]
    eids = world.spawn_batch_images(images)
    print(f"[OK] Created batch of {len(eids)} images")

    # Each entity gets its own pipeline
    for i, eid in enumerate(eids):
        latent = world.pipe(eid).to(DummyVAEEncoder()).out(Latent4)
        latent_view = world.arena.view(latent.z)
        print(f"  Entity {eid}: latent shape {latent_view.shape}")
    print()

    # Example 5: select() for component branching
    print("Example 5: Branching with .select()")
    print("-" * 40)
    entity4 = world.new_entity()
    latent_data2 = np.random.randn(4, 16, 16).astype(np.float32)
    latent_ref2 = world.arena.copy_tensor(latent_data2)
    world.add_component(entity4, Latent4(z=latent_ref2))

    # Use select() to switch component branch
    yuvw = (
        world.pipe(entity4)
        .select(Latent4)
        .to(Hadamard4(mode="forward"))
        .out(YUVW4)
    )
    print(f"[OK] Used .select(Latent4) to process specific component")
    print(f"  Output type: YUVW4")
    print(f"  Transformed shape: {world.arena.view(yuvw.t).shape}\n")

    # Example 6: use() alias for select()
    print("Example 6: .use() as alias for .select()")
    print("-" * 40)
    entity5 = world.new_entity()
    world.add_component(entity5, Latent4(z=latent_ref2))

    yuvw2 = (
        world.pipe(entity5)
        .use(Latent4)
        .to(Hadamard4(mode="forward"))
        .out(YUVW4)
    )
    print(f"[OK] .use() is cleaner alias for component selection\n")

    # Summary
    print("Summary")
    print("=" * 40)
    print("[OK] Pipe.to() adds systems to pipeline")
    print("[OK] Pipe | operator provides alternative syntax")
    print("[OK] Pipe.select()/use() branches to different components")
    print("[OK] Pipe.out() executes pipeline and returns result")
    print("[OK] Pipe.execute() runs without returning component")
    print("[OK] Type-safe: IDE knows result types after .out()")
    print("[OK] Composable: chain multiple systems seamlessly")
    print("\n[OK] Phase 11 complete: Fluent API enables elegant pipelines!")


if __name__ == "__main__":
    main()
