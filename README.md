# VAE+ANS Image Compression SDK

A developer-friendly image compression SDK using Variational Autoencoder (VAE) encoding and Asymmetric Numeral Systems (ANS) entropy coding, built with an Entity-Component-System (ECS) architecture.

## Features

- **Learned Compression**: VAE-based image encoding for better perceptual quality
- **Efficient Entropy Coding**: ANS for near-optimal compression
- **Zero-Copy Memory**: Arena allocation with TensorRef handles for minimal overhead
- **Flexible Pipeline**: Fluent API with method chaining and branching
- **Batch Processing**: Automatic batching of operations with identical shapes
- **Type Safety**: Pydantic models with full mypy support

## Installation

```bash
pip install vaerans_ecs
```

Or install from source:

```bash
git clone https://github.com/yourorg/vaerans_ecs.git
cd vaerans_ecs
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from vaerans_ecs import compress, decompress

# Load or create an image
img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

# Compress
compressed = compress(img, model='sdxl-vae', quality=50)
print(f"Compressed size: {len(compressed)} bytes")

# Decompress
reconstructed = decompress(compressed)
print(f"Reconstructed shape: {reconstructed.shape}")
```

## Configuration

Copy `vaerans_ecs.toml.example` to `vaerans_ecs.toml` and configure your ONNX model paths:

```toml
[models.sdxl-vae]
encoder = "/path/to/sdxl_vae_encoder.onnx"
decoder = "/path/to/sdxl_vae_decoder.onnx"
```

Or set the `VAERANS_CONFIG` environment variable:

```bash
export VAERANS_CONFIG=/path/to/your/config.toml
```

## Advanced Usage

### Custom Pipeline

```python
from vaerans_ecs import World
from vaerans_ecs.systems import (
    OnnxVAEEncode, Hadamard4, WaveletCDF53,
    QuantizeU8, ANSEncode
)
from vaerans_ecs.components import ANSBitstream

world = World()
entity = world.spawn_image(img)

# Build custom compression pipeline
bitstream = (
    world.pipe(entity)
    .to(OnnxVAEEncode(model='sdxl-vae', mode='encode'))
    .to(Hadamard4(mode='forward'))
    .to(WaveletCDF53(levels=4, mode='forward'))
    .to(QuantizeU8(mode='encode', quality=50))
    .to(ANSEncode())
    .out(ANSBitstream)
)
```

### Batch Processing

```python
# Process multiple images efficiently
images = [img1, img2, img3]
eids = world.spawn_batch_images(images)

# Pipeline automatically batches operations
for eid in eids:
    result = world.pipe(eid).to(OnnxVAEEncode()).out(Latent4)
```

## Architecture

The SDK is built on five architectural layers:

1. **Application & API**: High-level `compress()` / `decompress()` functions
2. **Pipeline & Scheduling**: System sequencing, batching, and execution
3. **Systems & Transforms**: VAE, Hadamard, Wavelet, Quantizers, ANS coders
4. **ECS Components & Memory**: Component definitions and entity management
5. **Arena & Hardware**: Zero-copy memory allocation

See [SOFTWARE_DESIGN.md](SOFTWARE_DESIGN.md) for detailed architecture documentation.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=vaerans_ecs --cov-report=html

# Type checking
mypy vaerans_ecs

# Linting
ruff check vaerans_ecs
black vaerans_ecs
```

## Implementation Status

This SDK is currently under development. Implementation progress:

- [x] Phase 0: Project Setup
- [ ] Phase 1: Arena and TensorRef
- [ ] Phase 2: World and Entity Management
- [ ] Phase 3: Basic Components
- [ ] Phase 4: System Base Class
- [ ] Phase 5: Simple Transform Systems
- [ ] Phase 6-8: External Transform Systems
- [ ] Phase 9: Bitstream Serialization
- [ ] Phase 10: ONNX VAE Systems
- [ ] Phase 11: Pipeline and Scheduling
- [ ] Phase 12: Metrics Systems
- [ ] Phase 13: High-Level API
- [ ] Phase 14-16: Quality and Polish

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use this SDK in your research, please cite:

```bibtex
@software{vaerans_ecs,
  title = {VAE+ANS Image Compression SDK},
  author = {VAE+ANS Team},
  year = {2026},
  url = {https://github.com/yourorg/vaerans_ecs}
}
```
