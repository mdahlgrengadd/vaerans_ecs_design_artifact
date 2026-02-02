# Implementation Plan: VAE+ANS Image Compression SDK

## Executive Summary

This plan addresses:
1. **Review and revision of SOFTWARE_DESIGN.md**: Fix citations, inconsistencies, and gaps
2. **Architecture improvements**: Adopt Pydantic for type-safe components
3. **API standardization**: Use mode parameters and type references throughout
4. **15-phase implementation**: From core Arena to full API with tests

**Key Architectural Decisions** (confirmed with user):
- ✅ Pydantic BaseModel for all components (type safety)
- ✅ Mode parameter for encode/decode operations (cleaner API)
- ✅ Local ONNX files via configuration (no auto-download)
- ✅ Bump allocator with reset() (simpler, efficient)
- ✅ Type-safe component references (IDE support)

**Estimated Scope**: 15 implementation phases, ~3000-4000 lines of production code, ~2000-3000 lines of tests

## Review Summary

### Design Document Strengths
- Clear 5-layer architecture with good separation of concerns
- Well-defined ECS pattern for extensibility
- Zero-copy Arena memory model for efficiency
- Comprehensive component and system catalog
- Good API examples demonstrating fluent interface

### Issues Identified in Design Document

#### Critical Issues
1. **Broken Citations**: Document contains unresolved citation markers (【326889321455777†L228-L244】) that should be removed or properly formatted
2. **API Inconsistencies**:
   - Line 58: `QuantizeU8.dequantize()` vs Line 48: `QuantizeU8(per_band=True)` - unclear if static or instance methods
   - Lines 59-60: `.inverse()` pattern inconsistent with forward direction
3. **Missing System Interface Definition**: How do systems declare their input/output component dependencies?

#### Important Gaps
4. **Scheduler Algorithm**: No detail on how the scheduler resolves dependencies and batches entities
5. **Error Handling Strategy**: Not mentioned anywhere
6. **Memory Management Edge Cases**:
   - What happens when arena is full?
   - Can memory be freed/recycled?
   - Fragmentation handling?
7. **Thread Safety**: Is World thread-safe? Can multiple threads compress simultaneously?
8. **Component Naming**: `.out('Bitstream')` uses strings - how does this integrate with type system?

#### Documentation Gaps
9. **ONNX Model Management**: Where models come from, versioning, compatibility
10. **Quantization Details**: How scale/zero-point parameters are chosen
11. **ANS Probability Modeling**: How histograms are computed, context modeling
12. **Testing Strategy**: Completely missing
13. **Python Requirements**: Version requirements, complete dependency list
14. **Performance Considerations**: Expected throughput, memory usage, benchmarks

## Proposed Revisions to SOFTWARE_DESIGN.md

### 1. Remove Broken Citations
**Problem**: Document contains unresolved citation markers like 【326889321455777†L228-L244】
**Fix**: Remove all citation markers and rewrite as clean prose. If citations are needed, use proper references section.

### 2. Update to Pydantic-Based Architecture
**Major Change**: Switch from dataclasses to Pydantic BaseModel for all components

**Benefits**:
- Runtime type validation
- IDE autocomplete and type checking
- Automatic JSON serialization
- Better error messages

**Changes needed**:
- Update all component definitions in Class Overview (lines 187-195)
- Update code examples (lines 32-64, 70-84)
- Add `pydantic>=2.0` to dependencies (line 27)
- Explain Pydantic configuration in Component Architecture section (lines 137-149)

### 3. Standardize API with Mode Parameter
**Problem**: Inconsistent API for inverse operations (lines 58-60)

**Fix**: Use mode parameter pattern throughout:
```python
# Encoding
.to(QuantizeU8(mode='encode', per_band=True))
.to(Hadamard4(mode='forward'))
.to(WaveletCDF53(levels=4, mode='forward'))

# Decoding
.to(QuantizeU8(mode='decode', per_band=True))
.to(Hadamard4(mode='inverse'))
.to(WaveletCDF53(levels=4, mode='inverse'))
```

**Changes needed**:
- Update Quick Start examples (lines 32-64)
- Update System class definition (lines 197-209)
- Update all system descriptions (lines 154-161)

### 4. Make Component References Type-Safe
**Problem**: `.out('Bitstream')` uses strings (line 50)

**Fix**: Use component type references:
```python
# Before (string-based)
.out('Bitstream')
.use('Bitstream')

# After (type-safe)
.out(ANSBitstream)
.use(ANSBitstream)
```

**Benefits**:
- IDE autocomplete
- Compile-time type checking with mypy
- Refactoring-safe (rename component → all uses update)
- No typos in component names

**Changes needed**:
- Update all code examples (lines 32-84)
- Update Pipeline API description (lines 163-166, 210-218)

### 5. Add Configuration-Based Model Loading
**New Content**: Explain model loading from local configuration

Add to "Model & Algorithm Overview" section (after line 107):

```markdown
#### ONNX Model Configuration

VAE models are loaded from local files specified in configuration:

- Users create `vaerans_ecs.toml` config file with model paths
- Example config:
  ```toml
  [models.sdxl-vae]
  encoder = "/path/to/vae_encoder.onnx"
  decoder = "/path/to/vae_decoder.onnx"
  ```
- Models are loaded from absolute or relative paths
- No automatic downloads or network dependencies
- Clear and explicit model management

Dependencies: `tomli` (Python <3.11) or `tomllib` (Python >=3.11)
```

### 6. Add Implementation Details Section
**New Section** after "Class Overview" (after line 218):

```markdown
## Implementation Details

### System Mode Pattern

Systems support bidirectional operations via mode parameter:

```python
class System(ABC):
    def __init__(self, mode: Literal['encode', 'decode', 'forward', 'inverse'] = 'encode'):
        self.mode = mode
```

The mode determines required and produced components:
- Encode mode: RGB → Latent4
- Decode mode: Latent4 → ReconRGB

### Component Type Registry

World maintains type-indexed component stores:

```python
self._components: dict[type[Component], dict[int, Component]] = {}
```

This enables type-safe queries:
```python
latent: Latent4 = world.get_component(entity, Latent4)
```

### Arena Memory Model

Arena uses bump allocation with reset:
- Simple offset pointer allocation
- Proper alignment for numpy dtypes
- `.reset()` method for batch reuse
- OOM raises MemoryError with context

Memory is NOT freed per-allocation. Process all entities, then reset.

### Error Handling

Fast-fail strategy with clear messages:
- Component missing → KeyError with component type name
- Arena OOM → MemoryError with space requested/available
- System dependency unsatisfied → RuntimeError with required components
- Invalid Pydantic data → ValidationError with field details

### Thread Safety

World instances are NOT thread-safe. For parallel processing:
- Create separate World per thread
- Each World has its own Arena
- No shared state between Worlds
```

### 7. Add Testing Strategy Section
**New Section** after "Implementation Details":

```markdown
## Testing Strategy

### Unit Tests
- Each component type: creation, validation, field constraints
- Arena: allocation, alignment, OOM, reset
- Systems: forward/inverse correctness, batching
- Pipeline: dependency resolution, execution order

### Integration Tests
- Full encode-decode round-trip
- Multi-entity batch processing
- Branching pipelines (residual coding)
- Metric computation

### Test Data
- Fixture images (various sizes, content types)
- Golden outputs for deterministic systems
- Known PSNR/SSIM values for validation

### Performance Benchmarks
- Throughput (images/second) vs batch size
- Memory usage profiling
- Comparison with reference implementations
```

### 8. Add Requirements Section
**New Section** before "Quick Start" (before line 22):

```markdown
### Requirements

**Python**: 3.9 or higher (required for type hint features)

**Core Dependencies**:
- numpy >= 1.21
- pydantic >= 2.0
- onnxruntime >= 1.15
- tomli >= 2.0 (Python <3.11 only)

**Compression**:
- constriction >= 0.3
- pywavelets >= 1.4

**Metrics**:
- scikit-image >= 0.19

**Development**:
- pytest >= 7.0
- pytest-cov >= 4.0
- mypy >= 1.0
- black, ruff

**Optional (GPU)**:
- onnxruntime-gpu >= 1.15
- CUDA toolkit 11.x or 12.x
```

### 9. Fix Terminology Inconsistencies
- Line 141: "colours‑space" → "colorspace"
- Lines 19, 20, 69: Inconsistent component naming (e.g., "Bitstream" vs "ANSBitstream")
  → Standardize all component names to match class names

### 10. Update File Tree
Update file tree (lines 222-251) to include:
```
vaerans_ecs/
  ...
  core/
    ...
    serialization.py    # Bitstream serialization/deserialization
  ...
tests/
  test_arena.py
  test_world.py
  test_components.py
  test_systems/
    test_vae.py
    test_hadamard.py
    ...
  test_pipeline.py
  test_api.py
  integration/
    test_roundtrip.py
    test_batch.py
examples/
  basic_compression.py
  batch_processing.py
  custom_pipeline.py
benchmarks/
  throughput.py
  memory_profiling.py
```

## Implementation Plan

### Phase 0: Project Setup (Foundation)
**Goal**: Create project structure and development environment

**Tasks**:
1. Create directory structure as specified
2. Setup `pyproject.toml` with dependencies:
   - Core: numpy, onnxruntime, pydantic>=2.0, tomli (for Python <3.11)
   - Compression: constriction, pywavelets
   - Metrics: scikit-image
   - Dev: pytest, black, mypy, ruff, pytest-cov
3. Initialize git repository
4. Create basic `__init__.py` files
5. Setup pytest and mypy configuration
6. Configure mypy for strict type checking with Pydantic plugin
7. Create example `vaerans_ecs.toml.example` config file:
   ```toml
   # Example configuration for vaerans_ecs
   # Copy to vaerans_ecs.toml and update paths

   [models.sdxl-vae]
   encoder = "/path/to/sdxl_vae_encoder.onnx"
   decoder = "/path/to/sdxl_vae_decoder.onnx"

   [models.custom-vae]
   encoder = "./models/my_encoder.onnx"
   decoder = "./models/my_decoder.onnx"
   ```

**Validation**: `pytest` runs (even with no tests), `mypy` passes, project imports successfully

---

### Phase 1: Core Memory Infrastructure
**Goal**: Implement Arena and TensorRef with zero-copy semantics

**Files to Create**:
- `vaerans_ecs/core/arena.py`
- `tests/test_arena.py`

**Implementation Details**:

**Arena Class**:
```python
class Arena:
    def __init__(self, size_bytes: int):
        self._buffer = bytearray(size_bytes)
        self._offset = 0
        self._size = size_bytes
        self._allocations = []  # track for debugging

    def alloc_tensor(self, shape: tuple, dtype: np.dtype) -> TensorRef:
        # Calculate size in bytes
        itemsize = np.dtype(dtype).itemsize
        size = int(np.prod(shape)) * itemsize

        # Calculate aligned offset (align to itemsize)
        alignment = max(itemsize, 8)
        aligned_offset = (self._offset + alignment - 1) // alignment * alignment

        # Check for OOM
        if aligned_offset + size > self._size:
            raise MemoryError(f"Arena out of memory: need {size} bytes, have {self._size - aligned_offset}")

        # Create TensorRef
        strides = _compute_strides(shape, itemsize)
        ref = TensorRef(offset=aligned_offset, shape=shape, dtype=dtype, strides=strides)

        # Update offset
        self._offset = aligned_offset + size
        self._allocations.append(ref)

        return ref

    def view(self, ref: TensorRef) -> np.ndarray:
        # Return numpy view into buffer (zero-copy)
        return np.ndarray(
            shape=ref.shape,
            dtype=ref.dtype,
            buffer=self._buffer,
            offset=ref.offset,
            strides=ref.strides
        )

    def reset(self):
        """Reset arena to reuse memory. Invalidates all existing TensorRefs."""
        self._offset = 0
        self._allocations.clear()
```

**TensorRef Class**:
```python
@dataclass(frozen=True)
class TensorRef:
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype
    strides: tuple[int, ...]

    def subref_axis0(self, index: int) -> TensorRef:
        # Slice along first dimension
```

**Tests**:
- Basic allocation and viewing
- Alignment for different dtypes
- Out of memory handling
- Subrefs for batching

**Validation**: All arena tests pass, memory views work correctly

---

### Phase 2: World and Entity Management
**Goal**: Implement World with entity and component stores

**Files to Create**:
- `vaerans_ecs/core/world.py`
- `tests/test_world.py`

**Implementation Details**:

**World Class**:
```python
from typing import TypeVar, Generic

T = TypeVar('T', bound=Component)

class World:
    def __init__(self, arena_bytes: int = 512 << 20):
        self.arena = Arena(arena_bytes)
        self._next_eid = 0
        self._components: dict[type[Component], dict[int, Component]] = {}
        self.metadata: dict[int, dict[str, Any]] = {}

    def new_entity(self) -> int:
        """Create a new entity and return its ID."""
        eid = self._next_eid
        self._next_eid += 1
        self.metadata[eid] = {}
        return eid

    def spawn_image(self, img: np.ndarray) -> int:
        """
        Ingest an RGB image into the world.

        Args:
            img: RGB image as (H, W, 3) uint8 array

        Returns:
            Entity ID
        """
        # Validate input
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {img.shape}")

        # Create entity
        eid = self.new_entity()

        # Allocate in arena
        pix_ref = self.arena.alloc_tensor(img.shape, np.uint8)
        self.arena.view(pix_ref)[:] = img

        # Create and attach RGB component
        rgb = RGB(pix=pix_ref, colorspace='sRGB')
        self.add_component(eid, rgb)

        return eid

    def add_component(self, eid: int, component: Component) -> None:
        """Attach a component to an entity."""
        comp_type = type(component)
        if comp_type not in self._components:
            self._components[comp_type] = {}
        self._components[comp_type][eid] = component

    def get_component(self, eid: int, comp_type: type[T]) -> T:
        """
        Retrieve a component from an entity.

        Raises:
            KeyError: If entity doesn't have the component
        """
        return self._components[comp_type][eid]  # type: ignore

    def has_component(self, eid: int, comp_type: type[Component]) -> bool:
        """Check if entity has a component type."""
        return eid in self._components.get(comp_type, {})

    def query(self, *comp_types: type[Component]) -> list[int]:
        """
        Query for all entities that have ALL specified component types.

        Args:
            comp_types: Component types to query for

        Returns:
            List of entity IDs
        """
        if not comp_types:
            return []

        # Start with entities that have the first component
        candidates = set(self._components.get(comp_types[0], {}).keys())

        # Intersect with entities that have remaining components
        for comp_type in comp_types[1:]:
            candidates &= set(self._components.get(comp_type, {}).keys())

        return sorted(candidates)

    def pipe(self, entity: int) -> 'Pipe':
        """Create a pipeline for the given entity."""
        return Pipe(self, entity)
```

**Tests**:
- Entity creation
- Component attachment/retrieval
- Multiple entities with different components
- spawn_image with various sizes

**Validation**: World can manage entities and components correctly

---

### Phase 3: Basic Components
**Goal**: Implement core component types

**Files to Create**:
- `vaerans_ecs/components/image.py` (RGB, ReconRGB, BlurRGB)
- `vaerans_ecs/components/latent.py` (Latent4, YUVW4)
- `vaerans_ecs/components/wavelet.py` (WaveletPyr)
- `vaerans_ecs/components/quant.py` (SymbolsU8, QuantParams)
- `vaerans_ecs/components/entropy.py` (ANSBitstream)
- `vaerans_ecs/components/residual.py` (Residual)
- `tests/test_components.py`

**Implementation Details**:
Each component is a Pydantic BaseModel wrapping TensorRefs:

```python
from pydantic import BaseModel, Field

class Component(BaseModel):
    """Base class for all ECS components."""
    model_config = {'arbitrary_types_allowed': True}

class RGB(Component):
    pix: TensorRef
    colorspace: str = 'sRGB'

class Latent4(Component):
    z: TensorRef

class YUVW4(Component):
    t: TensorRef

class WaveletPyr(Component):
    packed: TensorRef
    index: TensorRef
    levels: int = Field(ge=1, le=10)
    wavelet: str = Field(default='bior2.2')

class SymbolsU8(Component):
    q: TensorRef
    qp: 'QuantParams'

class QuantParams(BaseModel):
    scale: np.ndarray
    zero: np.ndarray
    model_config = {'arbitrary_types_allowed': True}

class ANSBitstream(Component):
    probs: np.ndarray
    words_u32: TensorRef
    n_symbols: int = Field(ge=0)

class Residual(Component):
    r: TensorRef

class ReconRGB(Component):
    pix: TensorRef
    colorspace: str = 'sRGB'

class BlurRGB(Component):
    pix: TensorRef
    colorspace: str = 'sRGB'
```

Benefits of Pydantic:
- Runtime validation of component fields
- Type checking with mypy
- Automatic JSON serialization for checkpointing
- Clear error messages when invalid data is provided
- Documentation via Field descriptions

**Tests**:
- Component creation
- TensorRef wrapping
- Metadata preservation

**Validation**: All component types can be created and store references correctly

---

### Phase 4: System Base Class and Registry
**Goal**: Define System interface with dependency declaration

**Files to Create**:
- `vaerans_ecs/core/system.py`
- `tests/test_system.py`

**Implementation Details**:

```python
from abc import ABC, abstractmethod
from typing import Literal

class System(ABC):
    """Base class for all systems."""

    def __init__(self, mode: Literal['encode', 'decode', 'forward', 'inverse'] = 'encode'):
        """
        Initialize system with mode.

        Args:
            mode: Operation mode. 'encode'/'forward' for forward direction,
                  'decode'/'inverse' for reverse direction.
        """
        self.mode = mode

    @abstractmethod
    def required_components(self) -> list[type[Component]]:
        """Return list of component types this system needs."""
        pass

    @abstractmethod
    def produced_components(self) -> list[type[Component]]:
        """Return list of component types this system produces."""
        pass

    @abstractmethod
    def run(self, world: World, eids: list[int]) -> None:
        """Execute system on given entities."""
        pass

    def can_run(self, world: World, eid: int) -> bool:
        """Check if entity has all required components."""
        return all(world.has_component(eid, ct) for ct in self.required_components())

# Example usage:
encoder = QuantizeU8(mode='encode', per_band=True)
decoder = QuantizeU8(mode='decode', per_band=True)
```

**Tests**:
- Mock system implementation
- Dependency checking
- can_run validation

**Validation**: System interface is clear and testable

---

### Phase 5: Simple Transform Systems
**Goal**: Implement systems without external dependencies

**Files to Create**:
- `vaerans_ecs/systems/hadamard.py` (Hadamard4, Hadamard4Inverse)
- `vaerans_ecs/systems/blur.py` (GaussianBlur)
- `vaerans_ecs/systems/residual.py` (ResidualCompute, ResidualAdd)
- `tests/test_hadamard.py`
- `tests/test_blur.py`
- `tests/test_residual.py`

**Implementation Details**:

**Hadamard4 System**:
```python
class Hadamard4(System):
    def required_components(self):
        return [Latent4]

    def produced_components(self):
        return [YUVW4]

    def run(self, world: World, eids: list[int]):
        for eid in eids:
            latent = world.get_component(eid, Latent4)
            # Get view from arena
            z = world.arena.view(latent.z)
            # Allocate output
            t_ref = world.arena.alloc_tensor(z.shape, z.dtype)
            t = world.arena.view(t_ref)
            # Apply 4x4 Hadamard on channel dimension
            self._hadamard_transform(z, t)
            # Store result
            world.add_component(eid, YUVW4(t_ref))

    def _hadamard_transform(self, input, output):
        # Implement orthonormal Hadamard matrix multiply
        # H4 = [[1,1,1,1], [1,1,-1,-1], [1,-1,-1,1], [1,-1,1,-1]] / 2
        pass
```

**Tests**:
- Forward and inverse transforms
- Orthonormality (inverse should recover original)
- Various tensor shapes

**Validation**: Transforms work correctly, inverse recovers original data

---

### Phase 6: Wavelet Transform Systems
**Goal**: Integrate PyWavelets for multi-level decomposition

**Files to Create**:
- `vaerans_ecs/systems/wavelet.py` (WaveletCDF53, WaveletCDF53Inverse, WaveletHaar, WaveletHaarInverse)
- `tests/test_wavelet.py`

**Implementation Details**:

```python
import pywt

class WaveletCDF53(System):
    def __init__(self, levels: int = 4):
        self.levels = levels
        self.wavelet = 'bior2.2'  # CDF 5/3

    def required_components(self):
        return [YUVW4]

    def produced_components(self):
        return [WaveletPyr]

    def run(self, world: World, eids: list[int]):
        for eid in eids:
            yuvw = world.get_component(eid, YUVW4)
            t = world.arena.view(yuvw.t)

            # Apply wavelet decomposition per channel
            coeffs_list = []
            for c in range(t.shape[0]):
                coeffs = pywt.wavedec2(t[c], self.wavelet, level=self.levels)
                coeffs_list.append(coeffs)

            # Pack coefficients into arena
            packed_ref, index_ref = self._pack_coefficients(world.arena, coeffs_list)

            world.add_component(eid, WaveletPyr(
                packed=packed_ref,
                index=index_ref,
                levels=self.levels,
                wavelet=self.wavelet
            ))

    def _pack_coefficients(self, arena, coeffs_list):
        # Pack into contiguous buffer with index
        pass
```

**Tests**:
- Multi-level decomposition
- Reconstruction accuracy
- Different wavelet types (CDF 5/3, Haar)

**Validation**: Wavelet transform is invertible with minimal numerical error

---

### Phase 7: Quantization Systems
**Goal**: Implement quantization with per-band scaling

**Files to Create**:
- `vaerans_ecs/systems/quantize.py` (QuantizeU8, DequantizeU8)
- `tests/test_quantize.py`

**Implementation Details**:

```python
class QuantizeU8(System):
    def __init__(self, per_band: bool = True):
        self.per_band = per_band

    def required_components(self):
        return [WaveletPyr]

    def produced_components(self):
        return [SymbolsU8]

    def run(self, world: World, eids: list[int]):
        for eid in eids:
            pyr = world.get_component(eid, WaveletPyr)
            packed = world.arena.view(pyr.packed)
            index = world.arena.view(pyr.index)

            # Compute scale and zero point per band
            if self.per_band:
                scales, zeros = self._compute_per_band_params(packed, index)
            else:
                scales, zeros = self._compute_global_params(packed)

            # Quantize
            q_ref = world.arena.alloc_tensor(packed.shape, np.uint8)
            q = world.arena.view(q_ref)
            self._quantize(packed, q, scales, zeros)

            world.add_component(eid, SymbolsU8(
                q=q_ref,
                qp=QuantParams(scale=scales, zero=zeros)
            ))

    def _compute_per_band_params(self, data, index):
        # For each band, compute min/max and derive scale/zero
        pass
```

**Tests**:
- Quantization to different bit depths
- Per-band vs global quantization
- Dequantization error bounds

**Validation**: Quantization preserves value range, dequantization reconstructs approximately

---

### Phase 8: ANS Entropy Coding
**Goal**: Integrate constriction library for rANS coding

**Files to Create**:
- `vaerans_ecs/systems/ans.py` (ANSEncode, ANSDecode)
- `tests/test_ans.py`

**Implementation Details**:

```python
import constriction

class ANSEncode(System):
    def required_components(self):
        return [SymbolsU8]

    def produced_components(self):
        return [ANSBitstream]

    def run(self, world: World, eids: list[int]):
        for eid in eids:
            symbols_comp = world.get_component(eid, SymbolsU8)
            symbols = world.arena.view(symbols_comp.q).ravel()

            # Build histogram
            hist, bins = np.histogram(symbols, bins=256, range=(0, 256))
            probs = (hist + 1) / (hist.sum() + 256)  # Add-1 smoothing

            # Encode with ANS
            encoder = constriction.stream.stack.AnsCoder()
            model = constriction.stream.model.Categorical(probs)
            encoder.encode_reverse(symbols, model)
            words = encoder.get_compressed()

            # Store in arena
            words_ref = world.arena.alloc_tensor((len(words),), np.uint32)
            world.arena.view(words_ref)[:] = words

            world.add_component(eid, ANSBitstream(
                probs=probs,
                words_u32=words_ref,
                n_symbols=len(symbols)
            ))
```

**Tests**:
- Encode/decode round-trip
- Compression ratio on test data
- Probability model accuracy

**Validation**: ANS encoding is reversible, achieves expected compression rates

---

### Phase 9: Bitstream Serialization
**Goal**: Implement serialization/deserialization for compressed data

**Files to Create**:
- `vaerans_ecs/core/serialization.py`
- `tests/test_serialization.py`

**Implementation Details**:

```python
import struct
import json
from typing import Tuple

# File format:
# [Header: 16 bytes]
#   - Magic: 4 bytes ('VAE\x00')
#   - Version: 2 bytes (major, minor)
#   - Metadata length: 4 bytes
#   - Reserved: 6 bytes
# [Metadata: variable JSON]
#   - model name
#   - wavelet levels
#   - image shape
#   - quantization params
# [Probability table: variable]
# [ANS bitstream: variable]

def serialize_bitstream(
    bitstream: ANSBitstream,
    model: str,
    levels: int,
    image_shape: tuple[int, int, int],
    quant_params: QuantParams
) -> bytes:
    """Serialize ANSBitstream component to bytes."""
    # Build metadata
    metadata = {
        'model': model,
        'wavelet_levels': levels,
        'image_shape': image_shape,
        'scale': quant_params.scale.tolist(),
        'zero': quant_params.zero.tolist(),
        'n_symbols': bitstream.n_symbols
    }
    metadata_bytes = json.dumps(metadata).encode('utf-8')

    # Build header
    header = struct.pack(
        '<4sHIH',
        b'VAE\x00',  # Magic
        1, 0,  # Version 1.0
        len(metadata_bytes),  # Metadata length
        0  # Reserved
    )

    # Serialize probability table
    probs_bytes = bitstream.probs.astype(np.float32).tobytes()

    # Serialize ANS words
    words_bytes = bitstream.words_u32.tobytes()  # Already in arena as uint32

    return header + metadata_bytes + probs_bytes + words_bytes

def deserialize_bitstream(data: bytes) -> Tuple[ANSBitstream, dict]:
    """
    Deserialize bytes to ANSBitstream component and metadata.

    Returns:
        (bitstream_component, metadata_dict)
    """
    # Parse header
    magic, ver_major, ver_minor, meta_len, _ = struct.unpack('<4sHIH', data[:16])

    if magic != b'VAE\x00':
        raise ValueError(f"Invalid file format: expected 'VAE\\x00', got {magic}")

    if ver_major != 1:
        raise ValueError(f"Unsupported version: {ver_major}.{ver_minor}")

    # Parse metadata
    metadata = json.loads(data[16:16+meta_len].decode('utf-8'))

    # Parse probability table
    offset = 16 + meta_len
    probs = np.frombuffer(data[offset:offset+256*4], dtype=np.float32)
    offset += 256 * 4

    # Parse ANS words
    words = np.frombuffer(data[offset:], dtype=np.uint32)

    # Create ANSBitstream (words are copied here, not in arena yet)
    bitstream = ANSBitstream(
        probs=probs,
        words_u32=words,  # Will need to be copied to arena
        n_symbols=metadata['n_symbols']
    )

    return bitstream, metadata
```

**Tests**:
- Round-trip serialization
- Version checking
- Corrupt data handling
- Large bitstreams

**Validation**: Bitstreams can be serialized and deserialized correctly

---

### Phase 10: ONNX VAE Systems
**Goal**: Integrate ONNX Runtime for VAE encode/decode

**Files to Create**:
- `vaerans_ecs/systems/vae.py` (OnnxVAEEncode, OnnxVAEDecode)
- `vaerans_ecs/models/` (directory for model files)
- `tests/test_vae.py`

**Implementation Details**:

```python
import onnxruntime as ort
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

class OnnxVAEEncode(System):
    def __init__(
        self,
        model: str = 'sdxl-vae',
        mode: Literal['encode', 'decode'] = 'encode',
        provider: str = 'CPUExecutionProvider',
        config_path: str | Path | None = None
    ):
        super().__init__(mode=mode)
        self.model_name = model
        self.model_path = self._load_model(model, mode, config_path)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=[provider]
        )

    def _load_model(self, model: str, mode: str, config_path: str | Path | None) -> str:
        """
        Load ONNX model from configuration file.

        Args:
            model: Model identifier (e.g., 'sdxl-vae') or direct ONNX file path
            mode: 'encode' or 'decode'
            config_path: Path to config file, or None to search defaults

        Returns:
            Path to ONNX model file

        Config file format (vaerans_ecs.toml):
        ```toml
        [models.sdxl-vae]
        encoder = "/path/to/vae_encoder.onnx"
        decoder = "/path/to/vae_decoder.onnx"

        [models.custom-vae]
        encoder = "./models/custom_encoder.onnx"
        decoder = "./models/custom_decoder.onnx"
        ```
        """
        # If model is a direct path to ONNX file, use it
        model_path = Path(model)
        if model_path.exists() and model_path.suffix == '.onnx':
            return str(model_path)

        # Otherwise, look up in config file
        config = self._load_config(config_path)

        if 'models' not in config or model not in config['models']:
            raise ValueError(
                f"Model '{model}' not found in config. "
                f"Available models: {list(config.get('models', {}).keys())} "
                f"or provide a direct path to .onnx file."
            )

        model_config = config['models'][model]
        key = 'encoder' if mode == 'encode' else 'decoder'

        if key not in model_config:
            raise ValueError(
                f"Model '{model}' config missing '{key}' path. "
                f"Found keys: {list(model_config.keys())}"
            )

        # Resolve path (can be absolute or relative to config file)
        onnx_path = Path(model_config[key])
        if not onnx_path.is_absolute() and config_path:
            # Resolve relative to config file location
            onnx_path = (Path(config_path).parent / onnx_path).resolve()

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_path}"
            )

        return str(onnx_path)

    def _load_config(self, config_path: str | Path | None) -> dict:
        """Load configuration from TOML file."""
        # Search for config in standard locations
        search_paths = []
        if config_path:
            search_paths.append(Path(config_path))
        else:
            search_paths.extend([
                Path.cwd() / 'vaerans_ecs.toml',
                Path.cwd() / '.vaerans_ecs.toml',
                Path.home() / '.config' / 'vaerans_ecs' / 'config.toml',
            ])

        for path in search_paths:
            if path.exists():
                with open(path, 'rb') as f:
                    return tomllib.load(f)

        raise FileNotFoundError(
            f"No config file found. Searched: {[str(p) for p in search_paths]}. "
            f"Create a vaerans_ecs.toml file with model paths."
        )

    def required_components(self) -> list[type[Component]]:
        if self.mode == 'encode':
            return [RGB]
        else:
            return [Latent4]

    def produced_components(self) -> list[type[Component]]:
        if self.mode == 'encode':
            return [Latent4]
        else:
            return [ReconRGB]

    def run(self, world: World, eids: list[int]):
        # Group entities by image shape for batching
        shape_groups = self._group_by_shape(world, eids)

        for shape, entity_ids in shape_groups.items():
            # Build batch input
            batch_input = self._build_batch(world, entity_ids)

            # Run inference
            outputs = self.session.run(['latent'], {'input': batch_input})

            # Split and store results
            self._store_results(world, entity_ids, outputs[0])

    def _group_by_shape(self, world, eids):
        groups = {}
        for eid in eids:
            rgb = world.get_component(eid, RGB)
            shape = world.arena.view(rgb.pix).shape
            if shape not in groups:
                groups[shape] = []
            groups[shape].append(eid)
        return groups
```

**Tests**:
- Requires actual ONNX model or mock
- Test with dummy model first
- Batching logic

**Validation**: VAE encode produces expected latent shape, decode reconstructs image

---

### Phase 11: Pipeline and Scheduling
**Goal**: Implement Pipe with fluent API and dependency resolution

**Files to Create**:
- `vaerans_ecs/core/pipeline.py`
- `tests/test_pipeline.py`

**Implementation Details**:

```python
from typing import TypeVar, Generic, overload

T = TypeVar('T', bound=Component)

class Pipe:
    """Type-safe fluent pipeline builder."""

    def __init__(self, world: World, entity: int):
        self.world = world
        self.entities = [entity]
        self.systems: list[System] = []
        self._current_component_type: type[Component] | None = None

    def to(self, system: System) -> 'Pipe':
        """Add system to pipeline."""
        self.systems.append(system)
        return self

    def __or__(self, system: System) -> 'Pipe':
        """Pipe operator for chaining."""
        return self.to(system)

    def select(self, component_type: type[T]) -> 'Pipe':
        """
        Switch to a different component branch.

        This allows building non-linear pipelines by selecting which
        component to operate on next.

        Args:
            component_type: The component type to select

        Returns:
            Self for chaining
        """
        self._current_component_type = component_type
        return self

    def use(self, component_type: type[T]) -> 'Pipe':
        """Alias for select."""
        return self.select(component_type)

    def out(self, component_type: type[T]) -> T:
        """
        Execute pipeline and return component.

        Args:
            component_type: The component type to retrieve

        Returns:
            The component instance

        Raises:
            KeyError: If entity doesn't have the requested component
        """
        self.execute()
        return self.world.get_component(self.entities[0], component_type)

    def execute(self) -> None:
        """Run all systems in order."""
        for system in self.systems:
            # Check which entities can run this system
            runnable = [eid for eid in self.entities if system.can_run(self.world, eid)]

            if not runnable:
                # Get component names for better error message
                required = [ct.__name__ for ct in system.required_components()]
                raise RuntimeError(
                    f"System {type(system).__name__} cannot run: "
                    f"entities missing required components {required}"
                )

            # Run system on runnable entities
            system.run(self.world, runnable)

# Example usage with type safety:
bitstream: ANSBitstream = (
    world.pipe(entity)
    .to(OnnxVAEEncode(model='sdxl-vae'))
    .to(Hadamard4(mode='forward'))
    .to(WaveletCDF53(levels=4, mode='forward'))
    .to(QuantizeU8(mode='encode', per_band=True))
    .to(ANSEncode())
    .out(ANSBitstream)  # Type-safe: returns ANSBitstream
)

# IDE knows bitstream is ANSBitstream, provides autocomplete
print(bitstream.n_symbols)
```

**Tests**:
- Pipeline construction
- System execution order
- Dependency resolution
- Branching with select/use

**Validation**: Pipelines execute systems in correct order, handle dependencies

---

### Phase 12: Metrics Systems
**Goal**: Implement PSNR, SSIM, MS-SSIM metrics

**Files to Create**:
- `vaerans_ecs/systems/metrics.py`
- `tests/test_metrics.py`

**Implementation Details**:

```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class MetricPSNR(System):
    def __init__(self, src_comp: type = RGB, recon_comp: type = ReconRGB):
        self.src_comp = src_comp
        self.recon_comp = recon_comp

    def required_components(self):
        return [self.src_comp, self.recon_comp]

    def produced_components(self):
        return []  # Produces metadata, not components

    def run(self, world: World, eids: list[int]):
        for eid in eids:
            src = world.arena.view(world.get_component(eid, self.src_comp).pix)
            recon = world.arena.view(world.get_component(eid, self.recon_comp).pix)

            psnr = peak_signal_noise_ratio(src, recon)

            # Store in entity metadata
            if not hasattr(world, 'metadata'):
                world.metadata = {}
            if eid not in world.metadata:
                world.metadata[eid] = {}
            world.metadata[eid]['psnr'] = psnr
```

**Tests**:
- PSNR computation
- SSIM computation
- MS-SSIM multi-scale

**Validation**: Metrics match expected values for known test cases

---

### Phase 13: High-Level API
**Goal**: Implement compress() and decompress() convenience functions

**Files to Create**:
- `vaerans_ecs/api.py`
- `tests/test_api.py`

**Implementation Details**:

```python
def compress(
    image: np.ndarray,
    model: str = 'sdxl-vae',
    quality: int = 50,
    wavelet_levels: int = 4,
    **kwargs
) -> bytes:
    """
    Compress an RGB image to bytes.

    Args:
        image: RGB image as (H, W, 3) uint8 array
        model: VAE model name or path
        quality: Quality level (0-100, higher = better quality)
        wavelet_levels: Number of wavelet decomposition levels

    Returns:
        Compressed bitstream as bytes
    """
    world = World()
    entity = world.spawn_image(image)

    # Build encode pipeline with mode parameters
    bitstream: ANSBitstream = (
        world.pipe(entity)
        .to(OnnxVAEEncode(model=model, mode='encode'))
        .to(Hadamard4(mode='forward'))
        .to(WaveletCDF53(levels=wavelet_levels, mode='forward'))
        .to(QuantizeU8(mode='encode', per_band=True, quality=quality))
        .to(ANSEncode())
        .out(ANSBitstream)  # Type-safe
    )

    # Serialize bitstream component to bytes
    return serialize_bitstream(bitstream, model=model, levels=wavelet_levels)

def decompress(data: bytes, **kwargs) -> np.ndarray:
    """
    Decompress bytes to RGB image.

    Args:
        data: Compressed bitstream bytes

    Returns:
        Reconstructed RGB image as (H, W, 3) uint8 array
    """
    world = World()
    entity = world.new_entity()

    # Deserialize bitstream and metadata
    bitstream, metadata = deserialize_bitstream(data)
    world.add_component(entity, bitstream)

    # Build decode pipeline using metadata
    recon: ReconRGB = (
        world.pipe(entity)
        .to(ANSDecode())
        .to(QuantizeU8(mode='decode', per_band=True))
        .to(WaveletCDF53(levels=metadata['wavelet_levels'], mode='inverse'))
        .to(Hadamard4(mode='inverse'))
        .to(OnnxVAEEncode(model=metadata['model'], mode='decode'))
        .out(ReconRGB)  # Type-safe
    )

    # Return numpy array view
    return world.arena.view(recon.pix)
```

**Tests**:
- End-to-end compress/decompress
- Various image sizes
- Quality settings

**Validation**: Complete pipeline works, images can be compressed and decompressed

---

### Phase 14: Context and Tracing
**Goal**: Add instrumentation for performance monitoring

**Files to Create**:
- `vaerans_ecs/core/context.py`
- `tests/test_context.py`

**Implementation Details**:

```python
@dataclass
class ExecutionContext:
    device: str = 'cpu'
    dtype: str = 'float32'
    trace: bool = False
    trace_data: list = field(default_factory=list)

class TracedSystem(System):
    """Wrapper that adds tracing to any system."""

    def __init__(self, inner: System, context: ExecutionContext):
        self.inner = inner
        self.context = context

    def run(self, world: World, eids: list[int]):
        if self.context.trace:
            start = time.perf_counter()
            self.inner.run(world, eids)
            elapsed = time.perf_counter() - start
            self.context.trace_data.append({
                'system': type(self.inner).__name__,
                'entities': len(eids),
                'time_ms': elapsed * 1000
            })
        else:
            self.inner.run(world, eids)
```

**Validation**: Tracing captures timing and metadata without affecting results

---

### Phase 15: Evaluation and Visualization
**Goal**: Implement report generation and RD curve plotting

**Files to Create**:
- `vaerans_ecs/eval/report.py`
- `vaerans_ecs/viz/plots.py`
- `tests/test_eval.py`

**Implementation Details**:

```python
def generate_report(world: World, eids: list[int], format: str = 'html') -> str:
    """Generate evaluation report from world metadata."""
    metrics = []
    for eid in eids:
        if hasattr(world, 'metadata') and eid in world.metadata:
            metrics.append(world.metadata[eid])

    # Create HTML/JSON report
    if format == 'html':
        return generate_html_report(metrics)
    elif format == 'json':
        return json.dumps(metrics)

def plot_rd_curve(bitrates: list[float], psnrs: list[float], **kwargs):
    """Plot rate-distortion curve."""
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(bitrates, psnrs, 'o-')
    plt.xlabel('Bitrate (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve')
    plt.grid(True)
    return plt.gcf()
```

**Validation**: Reports are generated correctly, plots are readable

---

### Phase 16: Testing and Documentation
**Goal**: Comprehensive test coverage and usage documentation

**Tasks**:
1. Achieve >80% test coverage across all modules
2. Add integration tests for full pipelines
3. Create example scripts in `examples/` directory
4. Write API documentation (docstrings)
5. Generate documentation with Sphinx
6. Add performance benchmarks

**Files to Create**:
- `examples/basic_compression.py`
- `examples/batch_processing.py`
- `examples/custom_pipeline.py`
- `docs/` directory with Sphinx config
- `benchmarks/` directory with performance tests

**Validation**:
- All tests pass
- Documentation builds successfully
- Examples run without errors

---

## Design Decisions (CONFIRMED)

1. **Inverse System Pattern**: ✅ Mode parameter in constructor
   - Example: `QuantizeU8(mode='encode')` and `QuantizeU8(mode='decode')`
   - Shared configuration, runtime mode switching
   - Cleaner than static methods, fewer classes than separate types

2. **Component References**: ✅ Type-safe with Pydantic
   - Use Pydantic BaseModel for all components
   - `.out(ANSBitstream)` instead of `.out('Bitstream')`
   - Full IDE support, runtime type checking
   - Component registry keyed by type

3. **Arena Memory**: ✅ Bump allocator with reset()
   - Simple bump pointer allocation
   - `.reset()` method for batch reuse
   - No per-allocation freeing
   - Optimal for batch processing patterns

4. **ONNX Models**: ✅ Local files via configuration
   - Models specified in config file (`vaerans_ecs.toml` or similar)
   - Users provide absolute/relative paths to ONNX encoder/decoder files
   - No automatic downloads, no network dependencies
   - Simple and explicit model management

5. **Thread Safety**: Single-threaded per World (decided during implementation)
   - Each World is single-threaded
   - Users can create multiple Worlds for parallel processing
   - Simpler implementation, explicit parallelism

6. **Error Strategy**: Exceptions immediately (decided during implementation)
   - Fast-fail for easier debugging
   - Clear error messages with context
   - Optional: future batch processing could collect errors

## Implementation Order Rationale

The phase order follows dependency chain:
1. **Phases 1-3**: Core infrastructure (no external deps) - Arena, World, Components
2. **Phases 4-5**: System abstraction and simple transforms (testable without models)
3. **Phases 6-8**: External integrations (PyWavelets, constriction)
4. **Phase 9**: Bitstream serialization (required for API)
5. **Phase 10**: Most complex external integration (ONNX + config loading)
6. **Phase 11**: Orchestration layer (Pipeline, needs all systems)
7. **Phases 12-15**: Quality of life features (metrics, API, tracing, visualization)
8. **Phase 16**: Polish and documentation

Each phase produces working, tested code that can be demonstrated independently.

**Critical Path**: Phases 1-3 → 4 → 10 → 11 → 13 (minimal end-to-end pipeline)
**Parallel Work**: Phases 5-8 can be developed concurrently after Phase 4

---

## Verification Strategy

After each phase:
1. **Unit tests pass**: `pytest tests/test_<module>.py -v`
2. **Type checking passes**: `mypy vaerans_ecs/<module>.py --strict`
3. **Code quality**: `ruff check vaerans_ecs/<module>.py`

After Phase 13 (High-Level API):
**End-to-End Test**:
```python
import numpy as np
from vaerans_ecs import compress, decompress

# Create test image
img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

# Compress and decompress
compressed = compress(img, model='sdxl-vae', quality=50)
reconstructed = decompress(compressed)

# Verify
assert reconstructed.shape == img.shape
assert reconstructed.dtype == np.uint8
print(f"Compression ratio: {img.nbytes / len(compressed):.2f}x")
print(f"Size: {len(compressed)} bytes")
```

After Phase 16 (Full Implementation):
**Integration Tests**:
```bash
# Run full test suite
pytest tests/ -v --cov=vaerans_ecs --cov-report=html

# Run examples
python examples/basic_compression.py
python examples/batch_processing.py
python examples/custom_pipeline.py

# Run benchmarks
python benchmarks/throughput.py
python benchmarks/memory_profiling.py

# Build documentation
cd docs && make html
```

---

## Next Steps

### Immediate Actions (Before Implementation)
1. **Revise SOFTWARE_DESIGN.md** according to proposed changes
2. **Get user approval** on revised design document
3. **Create project repository** with initial structure

### Implementation Approach
Two options:

**Option A: Sequential Implementation**
- Implement phases 0-16 in order
- Most dependencies satisfied
- Slower to first demo

**Option B: Critical Path First**
- Phases 0, 1, 2, 3, 4, 10, 11, 13 (minimal pipeline)
- Quick validation of architecture
- Backfill remaining phases
- Recommended for this project

### Risk Mitigation

**Technical Risks**:
1. **ONNX Model Availability**: Fallback to mock models for testing
2. **Arena Alignment**: Thorough testing on different platforms
3. **Pydantic Performance**: Benchmark vs plain dataclasses if needed
4. **ANS Compression**: Validate constriction library integration early

**Project Risks**:
1. **Scope Creep**: Stick to design, resist adding features
2. **Testing Gaps**: Maintain >80% coverage from Phase 1
3. **API Changes**: Lock API after Phase 11, only bug fixes after

### Success Criteria

✅ All 16 phases completed with tests passing
✅ End-to-end compression/decompression works
✅ PSNR matches expected quality levels
✅ Compression ratios competitive with baseline
✅ Type checking passes with mypy --strict
✅ Documentation complete and examples run
✅ Code coverage >80%

---

## Summary

This plan transforms the SOFTWARE_DESIGN.md from a conceptual document into a production-ready implementation with:

1. **Type-safe architecture** using Pydantic
2. **Clean API** with mode parameters and type references
3. **Comprehensive testing** at every phase
4. **Flexible model loading** via configuration files
5. **Performance focus** via zero-copy Arena

The 16-phase approach ensures each component is solid before building the next layer, with continuous validation throughout.
