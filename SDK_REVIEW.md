# SDK Review, Verification & Validation Report

**Date:** 2026-02-03
**Scope:** Full review of the VAE+ANS Image Compression SDK design artifact and implementation
**Test Environment:** Python 3.11, all dependencies installed from pyproject.toml

---

## Executive Summary

The SDK implements a VAE+ANS image compression pipeline using an ECS (Entity-Component-System) architecture with zero-copy arena memory. The core architectural concepts are sound and the design document is comprehensive. However, the current implementation has **10 failing tests** (out of 271), several **critical code-level bugs**, and notable **design-implementation gaps** that must be addressed before the SDK can be considered production-ready.

**Test Results:** 231 passed, 10 failed, 30 skipped (8 warnings)
**Coverage:** 77% overall

---

## 1. Critical Bugs

### 1.1 LinearTransform4 Indentation Error (systems/linear4.py:122-138)

The `required_components()`, `produced_components()`, and `run()` methods are **nested inside the `linear4_pair_from_npz()` function** rather than being class methods of `LinearTransform4`. This is an indentation error that makes the class non-instantiable:

```
TypeError: Can't instantiate abstract class LinearTransform4
    with abstract methods produced_components, required_components, run
```

The methods at lines 122-175 need to be dedented one level so they belong to the class body, not the function body. This causes 2 test failures.

### 1.2 Hadamard Test Expectations Mismatched with Implementation (7 test failures)

The Hadamard4 system uses `/4` normalization for the forward transform:

```python
# Forward: Y = (C0 + C1 + C2 + C3) / 4
# Inverse: C0 = Y + U + V + W
```

The round-trip `inverse(forward(x)) = x` is **mathematically correct** with this convention. However, the test suite (`tests/test_systems/test_hadamard.py`) was written for a `/2` normalization convention (norm-preserving orthonormal Hadamard). Specific issues:

- **test_forward_transform**: Expects Y=2.0 when all channels=1.0 (i.e. /2 norm), but implementation gives Y=1.0 (/4 norm)
- **test_inverse_transform**: Expects reconstructed=1.0 from Y=1/4, but gets 2.0
- **test_orthogonality**: References `system._H4` attribute that doesn't exist in the implementation
- **test_batch_method**: References `system._hadamard_batch` method that doesn't exist
- **test_precision**: Checks norm preservation `||Hx|| = ||x||` which only holds for /2 normalization

The implementation is internally consistent but the tests are stale and reflect a prior version of the code.

### 1.3 Per-Band Quantization is Non-Functional (systems/quantize.py:229-257)

The `_quantize()` method computes per-band scales and offsets but then **only uses `scales[0]` and `offsets[0]`** regardless of the `per_band` flag:

```python
# Line 246-250: Even when per_band=True, only scales[0] is used
scale = scales[0] if len(scales) > 0 else 1.0
offset_val = offsets[0] if len(offsets) > 0 else 0.0
quantized = ((packed - offset_val) * scale).clip(0, 255).astype(np.uint8)
```

The `_compute_quant_params()` method correctly calculates per-band parameters, but they are never applied per-band during the actual quantize/dequantize step. This means higher bands with different dynamic ranges will be quantized incorrectly, degrading compression quality.

### 1.4 Serialization Header Size Docstring Mismatch (core/serialization.py:6-17)

The module docstring says "Header: 16 bytes" with "Reserved: 6 bytes", but the actual struct format `<4sHII` produces 14 bytes (4+2+4+4), matching `HEADER_SIZE = 14`. The `get_serialized_size()` function also has an off-by-default estimate (hardcoded 150 bytes for metadata), causing 1 test failure:
```
assert 2134 == 2157  # off by 23 bytes
```

---

## 2. Architectural Issues

### 2.1 Decode Pipeline Has Circular Dependencies

Two critical systems cannot function in a standalone decode pipeline:

- **ANSDecode** (`systems/ans.py:152-165`) requires the original `SymbolsU8` component to exist in order to know `n_symbols`. In a real decode scenario (deserializing from file), SymbolsU8 doesn't exist yet. The symbol count should be stored in `ANSBitstream` or in the serialization metadata.

- **QuantizeU8 inverse** (`systems/quantize.py:135-146`) requires the original `WaveletPyr` to exist for its index structure. In a decode-from-file scenario, this is unavailable. The wavelet index structure needs to be serialized alongside the bitstream.

These are fundamental issues that make the encode-serialize-deserialize-decode path impossible without both the original and compressed data present simultaneously.

### 2.2 compress() Bypasses the Full Pipeline (api.py:87-130)

The high-level `compress()` function does **not** use the full pipeline (Wavelet, Quantize, ANS). Instead it:
1. Runs VAE encode + optional Hadamard
2. Serializes the raw float32 latent as bytes
3. Uses dummy QuantParams (scale=1.0, offset=0.0)
4. Uses a uniform probability table (1/256 per symbol)

This means `compress()` provides essentially zero actual compression -- the "compressed" output is larger than the latent representation. The function header and docstring describe full compression, creating a false impression.

### 2.3 Component Base Class Duplication

Each component module independently redefines `Component(BaseModel)`:

- `components/image.py:8` - defines Component
- `components/latent.py:8` - defines Component
- `components/wavelet.py:8` - defines Component
- `components/quant.py:8` - defines Component
- `components/entropy.py:8` - defines Component
- `components/residual.py:8` - defines Component

And `core/world.py:29` defines `Component = BaseModel` yet again. There is no single canonical base class. While all definitions are functionally equivalent, this violates DRY and makes it fragile -- if one definition diverges (e.g., one adds a custom validator), the type system would break since `world.py`'s `Component` is a different class than `image.py`'s `Component`.

### 2.4 Package __init__.py Exports Nothing

`vaerans_ecs/__init__.py` has all public API exports commented out:
```python
# from vaerans_ecs.api import compress, decompress
# from vaerans_ecs.core.world import World
# from vaerans_ecs.core.arena import Arena, TensorRef
```

Users cannot do `from vaerans_ecs import World` or `from vaerans_ecs import compress` as shown in the quickstart examples in SOFTWARE_DESIGN.md.

---

## 3. Design-Implementation Gaps

### 3.1 Missing Systems from Design Spec

The SOFTWARE_DESIGN.md specifies these systems which have no implementation:

| Specified | Status |
|-----------|--------|
| `systems/blur.py` (GaussianBlur) | Not implemented |
| `systems/residual.py` (Residual / AddResidual) | Not implemented |
| `core/context.py` (Device/dtype policies, tracing) | Not implemented |
| `eval/report.py` (HTML/JSON report generation) | Not implemented |
| `viz/plots.py` (RD curves, histograms) | Not implemented |

### 3.2 Extra Systems Not in Design Spec

The implementation includes systems not specified in the design document:

- `systems/linear4.py` (LinearTransform4) -- generic 4x4 linear transform (partly broken, see 1.1)
- `eval/hadamard.py` -- Hadamard evaluation utilities
- `eval/linear4.py` -- Linear4 evaluation utilities

These are reasonable extensions but represent design drift.

### 3.3 File Tree Divergence

The design doc's proposed file tree differs from the actual implementation:

| Design Spec | Implementation |
|-------------|---------------|
| `eval/report.py` | `eval/hadamard.py`, `eval/linear4.py` |
| `viz/plots.py` | `viz/__init__.py` (empty) |
| `core/context.py` | Missing |
| `systems/blur.py` | Missing |
| `systems/residual.py` | Missing |

### 3.4 API Surface Divergence

The design doc shows:
```python
w.pipe(entity).to(QuantizeU8(per_band=True)).to(ANSEncode()).out('Bitstream')
```

The implementation uses **component types** for `.out()` and `.select()`, not string keys:
```python
pipeline.out(YUVW4)  # type, not string 'YUVW4'
```

This is actually an improvement over the design doc (better type safety), but the design doc examples are misleading.

---

## 4. Code Quality Issues

### 4.1 Batching Only Partially Implemented

The design emphasizes "batch-first execution", but most systems process entities in a for-loop:
- `Hadamard4.run()` -- iterates per entity
- `WaveletCDF53.run()` -- iterates per entity
- `QuantizeU8.run()` -- iterates per entity
- `ANSEncode.run()` / `ANSDecode.run()` -- iterates per entity

Only `OnnxVAEEncode` and `OnnxVAEDecode` actually batch by shape and run a single inference call. True batching would require vectorized NumPy operations across entities.

### 4.2 Zero-Copy Is Partially Violated

The design emphasizes zero-copy, but several systems call `copy_tensor()` to create new arena allocations:
- `Hadamard4` allocates new output tensors via `alloc_tensor` (acceptable for output)
- `OnnxVAEEncode._encode_batch()` calls `.astype(np.float32)` creating a copy outside the arena
- `WaveletCDF53._run_forward()` calls `pywt.wavedec2()` which allocates outside the arena
- `ANSEncode.run()` calls `encoder.get_compressed()` returning data outside the arena

The zero-copy model holds for passing data between systems (via TensorRef), but intermediate computations frequently allocate outside the arena. This is an inherent limitation when wrapping libraries (PyWavelets, constriction, ONNX Runtime) that manage their own memory.

### 4.3 VAE Systems Use print() for Logging

`systems/vae.py:241,474` uses `print()` statements for model loading status:
```python
print(f"VAE Encoder loaded: {os.path.basename(model_path)} using {provider}")
```

These should use Python's `logging` module for proper log level control.

### 4.4 systems/__init__.py Is Empty

The systems package doesn't re-export its classes, so users must use verbose imports:
```python
from vaerans_ecs.systems.vae import OnnxVAEEncode  # required
# vs the design doc's implied:
from vaerans_ecs.systems import OnnxVAEEncode  # doesn't work
```

---

## 5. Diagram Validation

### 5.1 Layered Architecture Diagram (diagrams/layered_design.png)

The diagram correctly shows the five layers:
1. Application & API
2. Pipeline & Scheduling
3. Systems & Transforms
4. ECS Components & Memory
5. Arena & Hardware

This matches the design doc (SOFTWARE_DESIGN.md:110-119) and the implementation structure.

### 5.2 Component Diagram (diagrams/component_diagram.png)

The diagram shows:
- **Components:** RGB, Latent4, YUVW4, WaveletPyr, Symbols, Bitstream, Residual
- **Systems:** OnnxVAEEncode, OnnxVAEDecode, Hadamard4, Wavelet, QuantizeU8, ANSEncode, ANSDecode, Blur, Residual, AddResidual
- **Core:** Entity, World, Arena with correct relationships

Issues:
- "Symbols" in diagram should be "SymbolsU8" to match the implementation
- Blur, Residual, and AddResidual systems are shown but not implemented
- Missing: MetricPSNR, MetricSSIM, MetricMSSSIM, MetricMSE (implemented but not shown)
- Missing: ReconRGB, BlurRGB, QuantParams components (implemented but not shown)

---

## 6. Dependency and Configuration Validation

### 6.1 pyproject.toml

Dependencies are correctly specified and versions are compatible:
- `numpy>=1.21` -- OK
- `pydantic>=2.0` -- OK (using v2 API)
- `onnxruntime>=1.15` -- OK
- `constriction>=0.3` -- OK (using stream API)
- `PyWavelets>=1.4` -- OK
- `scikit-image>=0.19` -- OK (using metrics subpackage)
- `tomli>=2.0; python_version<'3.11'` -- OK (with tomllib fallback)

### 6.2 Config File (vaerans_ecs.toml.example)

Well-structured with model paths, execution provider, memory settings, compression defaults, and evaluation settings. The SDXL VAE configuration includes `input_range`, `output_range`, `latent_scale`, and `decoder_expects_scaled_latent` -- all properly handled in vae.py.

---

## 7. Integration Test Assessment

The integration test (`tests/integration/test_full_pipeline.py`) validates the core non-VAE pipeline:

```
Latent4 -> Hadamard -> Wavelet -> Quantize -> ANS -> Decode (reverse)
```

This test **passes** and demonstrates that the lossy pipeline produces reasonable reconstruction quality (MSE < 0.1 at quality=70), and that higher quality reduces error monotonically. The full VAE integration test is skipped (requires real ONNX models).

---

## 8. Summary of Findings

### Severity Classification

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1.1 | LinearTransform4 indentation bug | **Critical** | Code defect |
| 1.2 | Hadamard test/impl normalization mismatch | **High** | Test defect |
| 1.3 | Per-band quantization non-functional | **High** | Code defect |
| 1.4 | Serialization header docstring error | **Medium** | Doc defect |
| 2.1 | Decode pipeline circular dependencies | **Critical** | Architecture |
| 2.2 | compress() bypasses full pipeline | **High** | Architecture |
| 2.3 | Component base class duplication | **Medium** | Code quality |
| 2.4 | __init__.py exports nothing | **Medium** | Usability |
| 3.1 | Missing systems from design spec | **Medium** | Completeness |
| 3.2 | Extra systems not in spec | **Low** | Design drift |
| 3.3 | File tree divergence | **Low** | Documentation |
| 3.4 | API surface divergence (strings vs types) | **Low** | Documentation |
| 4.1 | Batching only partial | **Medium** | Performance |
| 4.2 | Zero-copy partially violated | **Low** | Performance |
| 4.3 | print() instead of logging | **Low** | Code quality |
| 4.4 | Empty systems __init__.py | **Low** | Usability |

### What Works Well

1. **Arena + TensorRef zero-copy model** -- well-implemented with generation counters for staleness detection
2. **ECS pattern** -- clean separation of entities, components, and systems
3. **World component management** -- type-safe query, add, get, remove operations
4. **Fluent Pipe API** -- `.to()` chaining with `.out()` execution and dependency checking
5. **VAE system** -- robust TOML config loading, lazy session init, batching by shape, input/output range normalization
6. **Wavelet system** -- correct packing/unpacking of PyWavelets coefficient structures
7. **ANS encoding** -- correct use of constriction's stack ANS coder with add-1 smoothing
8. **Quality metrics** -- PSNR, SSIM, MSE, MS-SSIM all correctly delegated to scikit-image
9. **Serialization format** -- versioned binary format with magic number and JSON metadata
10. **Test coverage** -- 231 passing tests covering core infrastructure thoroughly

### Recommended Priorities

1. Fix LinearTransform4 indentation (1.1) -- trivial fix, 2 tests restored
2. Align Hadamard tests with /4 normalization or change implementation to /2 (1.2) -- 7 tests restored
3. Fix per-band quantization to actually apply per-band parameters (1.3)
4. Store n_symbols in ANSBitstream and wavelet index in serialization (2.1)
5. Implement full pipeline in compress() (2.2)
6. Unify Component base class (2.3)
7. Enable __init__.py exports (2.4)
