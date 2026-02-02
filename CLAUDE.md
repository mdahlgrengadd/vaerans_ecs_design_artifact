# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **design artifact** for a VAE+ANS image compression SDK, not an implemented codebase. It contains comprehensive architectural documentation for building a developer-experience-first image compression SDK using an Entity-Component-System (ECS) architecture with zero-copy memory management.

## Key Architectural Concepts

### Core Design Principles

1. **Zero-Copy Memory Management**: All tensor data resides in a contiguous Arena (pre-allocated bytearray). Components store only lightweight TensorRef handles (offset, shape, dtype) pointing into the arena—no data copying between systems.

2. **Entity-Component-System (ECS) Pattern**:
   - **Entities**: Integer IDs representing compression jobs
   - **Components**: Data containers (RGB, Latent4, WaveletPyr, Symbols, Bitstream, etc.) wrapping TensorRefs
   - **Systems**: Transformations (VAE encode/decode, Hadamard, Wavelet, Quantization, ANS encoding) that operate on components

3. **Batch-First Execution**: Multiple entities with identical shapes are grouped and processed in a single ONNX inference call or transform operation.

4. **Fluent API**: Pipelines built using method chaining (`.to()`) or pipe operators (`|`), with branching via `.select()` and `.fork()`.

### Compression Pipeline

The design describes this transformation chain:
```
RGB Image → VAE Encode → Hadamard Transform → Wavelet Decomposition →
Quantization → ANS Entropy Coding → Bitstream
```

Decoding reverses the process. Optional branches support blurring, residual coding, and quality metrics (PSNR/SSIM).

### Five-Layer Architecture

1. **Application & API**: `World`, `compress()`, `decompress()`, fluent `Pipe` interface
2. **Pipeline & Scheduling**: System sequencing, batching, branching, trace logging
3. **Systems & Transforms**: VAE (ONNX), Hadamard, Wavelet, Quantizers, ANS coders, metrics
4. **ECS Components & Memory**: Component definitions and World entity management
5. **Arena & Hardware**: Zero-copy memory model, ONNX Runtime provider selection

## File Structure (Proposed)

The design document specifies this directory structure for the SDK implementation:

```
vaerans_ecs/
  __init__.py
  api.py                # High-level compress()/decompress() and Pipe
  core/
    arena.py            # Arena allocator and TensorRef
    world.py            # World, entities, component stores
    pipeline.py         # Pipe and system scheduling
    context.py          # Device/dtype policies, tracing
  components/
    image.py            # RGB, ReconRGB, BlurRGB
    latent.py           # Latent4, YUVW4
    wavelet.py          # WaveletPyr, packing
    quant.py            # SymbolsU8, QuantParams
    entropy.py          # ANSBitstream, probability tables
    residual.py         # Residual tensors
  systems/
    vae.py              # OnnxVAEEncode/Decode
    hadamard.py         # Hadamard4 transform
    wavelet.py          # WaveletCDF53/Haar
    quantize.py         # QuantizeU8/Dequantize
    ans.py              # ANS encode/decode (wraps constriction)
    blur.py             # GaussianBlur
    residual.py         # Residual/AddResidual
    metrics.py          # PSNR, SSIM, MS-SSIM
  eval/
    report.py           # Report generation
  viz/
    plots.py            # RD curves, visualizations
```

## External Dependencies (Design Spec)

The SDK is designed to integrate:
- **ONNX Runtime**: VAE model inference (SOFTWARE_DESIGN.md:9, 154)
- **constriction**: Optimized rANS entropy coding library (SOFTWARE_DESIGN.md:12, 97)
- **PyWavelets**: Wavelet transforms (CDF 5/3, Haar)
- **scikit-image**: PSNR and SSIM metrics (SOFTWARE_DESIGN.md:107)
- **NumPy**: Array views into arena memory

## Key Design References

- **Zero-copy memory**: SOFTWARE_DESIGN.md:15-16, 101-104, 127-130
- **VAE encoding/decoding**: SOFTWARE_DESIGN.md:9, 90-91, 154
- **ECS architecture**: SOFTWARE_DESIGN.md:7, 132-161
- **ANS entropy coding**: SOFTWARE_DESIGN.md:12, 97, 158
- **Fluent API examples**: SOFTWARE_DESIGN.md:30-86
- **Quality metrics (PSNR/SSIM)**: SOFTWARE_DESIGN.md:20, 105-107, 161
- **Batching strategy**: SOFTWARE_DESIGN.md:17, 102-104
- **Extensibility patterns**: SOFTWARE_DESIGN.md:255-261

## Working with This Design

When implementing features from this design:

1. **Understand the ECS flow**: Systems operate on components attached to entities. Check which components a system requires and produces.

2. **Respect the Arena model**: All tensors must be allocated via `Arena.alloc_tensor()`. Components hold TensorRefs, not raw arrays.

3. **Follow the layered architecture**: Don't mix concerns across layers. Systems shouldn't manage scheduling; the pipeline shouldn't implement transforms.

4. **Support batching**: Systems should handle lists of entity IDs and group by shape before processing.

5. **Maintain fluent API**: Pipeline operations should be chainable and support branching via `.select()`, `.fork()`, `.use()`.

6. **Reference the design doc**: Line numbers in SOFTWARE_DESIGN.md provide precise specifications for algorithms, data structures, and API patterns.

## Diagrams

- `diagrams/layered_design.png`: Five-layer architecture visualization
- `diagrams/component_diagram.png`: ECS relationships (Arena, World, Components, Systems)
