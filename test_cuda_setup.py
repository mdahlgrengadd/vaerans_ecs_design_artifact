#!/usr/bin/env python3
"""Quick test to verify ONNX Runtime CUDA setup."""

import sys

print("Testing ONNX Runtime Setup")
print("=" * 60)

# Check onnxruntime
try:
    import onnxruntime as ort
    print(f"[OK] onnxruntime installed: v{ort.__version__}")

    available_providers = ort.get_available_providers()
    print(f"\n[INFO] Available providers:")
    for provider in available_providers:
        marker = "[GPU]" if "CUDA" in provider or "Tensorrt" in provider else "[CPU]"
        print(f"   {marker} {provider}")

    if "CUDAExecutionProvider" in available_providers:
        print("\n[OK] CUDA provider is available!")
        print("     Your VAE models will run on GPU")
    else:
        print("\n[WARN] CUDA provider NOT available")
        print("       To enable GPU acceleration:")
        print("       1. Uninstall CPU version: pip uninstall onnxruntime")
        print("       2. Install GPU version: pip install onnxruntime-gpu")
        print("       3. Ensure CUDA is installed on your system")

except ImportError:
    print("[ERROR] onnxruntime not installed")
    print("        Install with: pip install onnxruntime-gpu")
    sys.exit(1)

# Check CUDA availability
print("\n" + "=" * 60)
print("Testing actual VAE loading...")
print("=" * 60 + "\n")

try:
    from vaerans_ecs.systems.vae import OnnxVAEEncode
    from vaerans_ecs.core.world import World
    import numpy as np

    # Create a simple test
    world = World(arena_bytes=100 << 20)
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    eid = world.spawn_image(test_image)

    # Try to create encoder (will trigger session load)
    print("Loading VAE encoder...")
    encoder = OnnxVAEEncode(model='test-vae', mode='encode')

    print("\nAttempting encode operation...")
    encoder.run(world, [eid])

    print("\n[OK] Encoder test successful!")
    print("     Check the output above to see which provider was used.")

except Exception as e:
    print(f"\n[WARN] Test failed: {e}")
    print("       This is expected if ONNX models are not present.")

print("\n" + "=" * 60)
print("Configuration:")
print("=" * 60)
print("Check your vaerans_ecs.toml for:")
print("  [models]")
print("  execution_provider = \"CUDAExecutionProvider\"")
print("\nThe SDK will automatically fall back to CPU if CUDA is unavailable.")
