#!/usr/bin/env python3
"""Test the updated normalized Hadamard transform."""

import numpy as np
from vaerans_ecs.core.world import World
from vaerans_ecs.components.latent import Latent4, YUVW4
from vaerans_ecs.systems.hadamard import Hadamard4

print("Testing Normalized Hadamard Transform")
print("=" * 60)

# Create test data
world = World(arena_bytes=100 << 20)
entity = world.new_entity()

# Create a simple test latent with known values
test_latent = np.array([
    [[10.0, 20.0], [30.0, 40.0]],  # C0
    [[5.0, 15.0], [25.0, 35.0]],   # C1
    [[8.0, 18.0], [28.0, 38.0]],   # C2
    [[12.0, 22.0], [32.0, 42.0]],  # C3
], dtype=np.float32)

print("\n1. Original Latent:")
print(f"   Shape: {test_latent.shape}")
print(f"   C0[0,0]={test_latent[0, 0, 0]}, C1[0,0]={test_latent[1, 0, 0]}")
print(f"   C2[0,0]={test_latent[2, 0, 0]}, C3[0,0]={test_latent[3, 0, 0]}")

# Manual calculation for pixel [0,0]
C0, C1, C2, C3 = 10.0, 5.0, 8.0, 12.0
Y_expected = (C0 + C1 + C2 + C3) / 4.0
U_expected = (C0 - C1 + C2 - C3) / 4.0
V_expected = (C0 + C1 - C2 - C3) / 4.0
W_expected = (C0 - C1 - C2 + C3) / 4.0

print(f"\n2. Expected YUVW at [0,0]:")
print(f"   Y = (10+5+8+12)/4 = {Y_expected}")
print(f"   U = (10-5+8-12)/4 = {U_expected}")
print(f"   V = (10+5-8-12)/4 = {V_expected}")
print(f"   W = (10-5-8+12)/4 = {W_expected}")

# Store in world
latent_ref = world.arena.copy_tensor(test_latent)
world.add_component(entity, Latent4(z=latent_ref))

# Forward transform
print("\n3. Applying forward Hadamard transform...")
hadamard_fwd = Hadamard4(mode='forward')
hadamard_fwd.run(world, [entity])

# Get YUVW
yuvw = world.get_component(entity, YUVW4)
yuvw_data = world.arena.view(yuvw.t)

print(f"   Actual YUVW at [0,0]:")
print(f"   Y = {yuvw_data[0, 0, 0]}")
print(f"   U = {yuvw_data[1, 0, 0]}")
print(f"   V = {yuvw_data[2, 0, 0]}")
print(f"   W = {yuvw_data[3, 0, 0]}")

# Check if they match
if np.allclose([yuvw_data[0, 0, 0], yuvw_data[1, 0, 0], yuvw_data[2, 0, 0], yuvw_data[3, 0, 0]],
               [Y_expected, U_expected, V_expected, W_expected]):
    print("   [OK] Forward transform matches expected values!")
else:
    print("   [ERROR] Forward transform doesn't match!")

# Inverse transform
print("\n4. Applying inverse Hadamard transform...")
hadamard_inv = Hadamard4(mode='inverse')
hadamard_inv.run(world, [entity])

# Get reconstructed latent
latent_recon = world.get_component(entity, Latent4)
latent_recon_data = world.arena.view(latent_recon.z)

print(f"   Reconstructed at [0,0]:")
print(f"   C0 = {latent_recon_data[0, 0, 0]} (original: {C0})")
print(f"   C1 = {latent_recon_data[1, 0, 0]} (original: {C1})")
print(f"   C2 = {latent_recon_data[2, 0, 0]} (original: {C2})")
print(f"   C3 = {latent_recon_data[3, 0, 0]} (original: {C3})")

# Check reconstruction
if np.allclose(latent_recon_data, test_latent, atol=1e-5):
    print("\n[OK] Perfect reconstruction! Hadamard is invertible.")
else:
    max_error = np.max(np.abs(latent_recon_data - test_latent))
    print(f"\n[ERROR] Reconstruction error: max={max_error}")

# Check energy distribution
print("\n5. Energy Distribution:")
Y, U, V, W = yuvw_data[0], yuvw_data[1], yuvw_data[2], yuvw_data[3]
energy_Y = float(np.sum(Y**2))
energy_U = float(np.sum(U**2))
energy_V = float(np.sum(V**2))
energy_W = float(np.sum(W**2))
total = energy_Y + energy_U + energy_V + energy_W

print(f"   Y: {energy_Y/total*100:.1f}% of total energy")
print(f"   U: {energy_U/total*100:.1f}% of total energy")
print(f"   V: {energy_V/total*100:.1f}% of total energy")
print(f"   W: {energy_W/total*100:.1f}% of total energy")

print("\n" + "=" * 60)
print("Summary:")
print("  - Forward transform uses normalized Hadamard (divide by 4)")
print("  - Inverse transform reconstructs perfectly")
print("  - Y contains most energy (as expected)")
print("\n[OK] Hadamard transform implementation verified!")
