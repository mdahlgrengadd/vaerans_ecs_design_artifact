# DX‑Friendly VAE+ANS Image Compression SDK – Software Design Document

## README

### Overview

The proposed SDK compresses and decompresses RGB images using modern latent‑space models and classical transforms.  It exposes a *Developer Experience (DX)–first* API with fluent method chaining and pipe operators.  Under the hood it uses an **Entity‑Component‑System (ECS)** architecture backed by a zero‑copy memory **Arena**; this design minimises memory allocations and copies by storing all tensor data in a contiguous arena and moving only lightweight references.  The high‑level steps are:

- **Variational Autoencoder (VAE) encoding/decoding** – A VAE uses an encoder network to project the image into a continuous, probabilistic latent representation and a decoder to reconstruct it【326889321455777†L228-L244】.  This SDK wraps an ONNX‑exported encoder/decoder and executes them using ONNX Runtime.
- **Transforms** – After encoding the latent, optional transforms such as *Hadamard* rotation (4×4 orthonormal transform), *wavelet* decomposition (e.g. CDF 5/3), and blurring are applied.  These help decorrelate channels and concentrate energy into a few coefficients.
- **Quantisation** – Each sub‑band is linearly scaled into integer symbols.  Per‑channel scale/zero point parameters are stored as side‑information.
- **Entropy coding (rANS)** – Symbols are encoded into a bitstream using a range‑asymmetric numeral system.  The `constriction` library provides optimised entropy coders for Python and Rust【583281890738763†L10-L17】; its stream codes achieve bit‑rates less than 0.1 % above the theoretical minimum while being faster than arithmetic coding【583281890738763†L108-L112】.

### Design goals

1. **Zero‑copy memory management.**  Data resides in an arena; each component points to a `TensorRef` (offset, shape, dtype) rather than a separate NumPy array.  Zero‑copy mechanisms avoid unnecessary user‑space ↔ kernel‑space copies and reduce CPU overhead【580669371589968†L22-L33】.
2. **Batch‑first execution.**  Multiple entities/jobs can be encoded, transformed and decoded simultaneously.  Systems group jobs with identical shapes and run a single ONNX inference per group.
3. **Open/Closed modularity.**  New transforms, coders and metrics can be added without modifying existing code.  Systems declare their input/output component types and are scheduled automatically.
4. **DX‑friendly fluent API.**  Pipelines can be constructed using `|` for piping or `.to()` for method chaining.  Branching (`.select()` and `.fork()`) allows optional operations such as blurring or residual coding.
5. **Evaluation built in.**  Quality metrics such as PSNR and SSIM are wrapped from scikit‑image; the PSNR function computes the peak signal‑to‑noise ratio【623168233359239†L489-L512】 and SSIM computes the mean structural similarity index【623168233359239†L558-L612】.

### Quick start

Install dependencies and the SDK (once published on PyPI):

```sh
pip install vaerans_ecs onnxruntime constriction pywavelets scikit-image
```

Encoding and decoding a single image:

```python
from vaerans_ecs import World
from vaerans_ecs.systems import OnnxVAEEncode, OnnxVAEDecode, Hadamard4, WaveletCDF53, QuantizeU8, ANSEncode

# create a world with a 512 MB arena
w = World(arena_bytes=512<<20)

# ingest an RGB image (H×W×3 uint8) into the world and get an entity id
entity = w.spawn_image(img)

# build a pipeline with chained systems
bitstream = (
    w.pipe(entity)
     .to(OnnxVAEEncode(model='sdxl-vae'))
     .to(Hadamard4())
     .to(WaveletCDF53(levels=4))
     .to(QuantizeU8(per_band=True))
     .to(ANSEncode())
     .out('Bitstream')
)

# decode
decoded = (
    w.pipe(entity)
     .use('Bitstream')
     .to(ANSDecode())
     .to(QuantizeU8.dequantize())
     .to(WaveletCDF53.inverse(levels=4))
     .to(Hadamard4.inverse())
     .to(OnnxVAEDecode(model='sdxl-vae'))
     .out('ReconRGB')
)
```

### Optional operations

The ECS design allows optional branches.  For example, to blur reconstructions and compute residuals:

```python
p = w.pipe(entity)
# encode as above …

# decode to reconstruction
p.select('Latent4').to(OnnxVAEDecode(), out='ReconRGB')

# blur reconstruction
p.select('ReconRGB').to(GaussianBlur(sigma=1.5), out='BlurRGB')

# compute residual between original and reconstruction
p.to(Residual(src='RGB', recon='ReconRGB', dtype='i16'), out='ResidualRGB')

# compress residual stream similarly (optional)
```

The `Residual` component can be entropy‑coded as a second stream and added back with `AddResidual` during decoding.

## Model & Algorithm Overview

A variational autoencoder consists of an encoder that learns to isolate latent variables from the input and a decoder that reconstructs the input from those latent variables【326889321455777†L228-L244】.  Unlike typical autoencoders, VAEs encode a *continuous, probabilistic* latent space【326889321455777†L240-L244】, enabling generation of new data similar to the training distribution.  In this SDK, the VAE compresses the image into a four‑channel latent tensor.  

After encoding, the latent is passed through a sequence of linear transforms:

1. **Hadamard transform.**  A fast orthonormal rotation that mixes the four latent channels and packs energy into fewer channels.  This transform can be inverted exactly.
2. **Wavelet decomposition.**  Multi‑level decomposition using the CDF 5/3 or Haar wavelet.  The wavelet transform produces LL (low‑low), LH, HL and HH sub‑bands at each level, concentrating high‑frequency information into high‑band sub‑bands.
3. **Quantisation.**  Each sub‑band is linearly mapped to integer symbols.  Per‑band scale and zero‑point parameters are stored alongside the bitstream so the symbols can be dequantised later.
4. **Entropy coding with rANS.**  Symbols are compressed into a bitstream using the range‑asymmetric numeral system; the `constriction` library provides highly optimised ANS coders with a clear API【583281890738763†L10-L17】.  The library’s stream codes achieve bit rates within ~0.1 % of the theoretical limit while being faster than arithmetic coding【583281890738763†L108-L112】.

During decoding, the steps are reversed: the bitstream is decoded, symbols are dequantised, inverse wavelet and Hadamard transforms are applied, and the VAE decoder reconstructs the image.

### Zero‑copy memory and batching

Traditional network I/O involves copying data between user‑space and kernel‑space buffers; zero‑copy techniques avoid these intermediate copies by transferring data directly between buffers within the kernel, reducing CPU involvement and latency【580669371589968†L22-L33】.  The SDK uses an arena—a large pre‑allocated `bytearray`—as a backing store.  Each tensor is represented by a `TensorRef` pointing into the arena; no data is copied when passing between systems.  Batching is built on top of this: the arena can hold a contiguous batch of latent tensors, and systems operate on batch views.  ONNX Runtime runs inference by loading a model into an `InferenceSession` and then calling `session.run(outputs, inputs)`【581473975824958†L44-L60】.  Grouping entities by shape allows a single `InferenceSession.run()` call per batch.

### Quality metrics

To evaluate reconstruction quality, the SDK wraps standard metrics from scikit‑image.  The `peak_signal_noise_ratio` function computes PSNR (higher is better) and returns the PSNR value【623168233359239†L489-L512】.  The `structural_similarity` function computes the mean structural similarity index (SSIM) between two images【623168233359239†L558-L612】; SSIM values closer to 1 indicate higher perceptual similarity.  A multi‑scale SSIM (MS‑SSIM) can be implemented by averaging SSIM over a pyramid of down‑sampled images.

## Layered Architecture

The SDK follows a clean layered architecture that separates concerns and promotes extensibility.  Each layer depends only on layers beneath it and exposes a stable interface to layers above.

![Layered architecture showing Application/API at the top, Pipeline & Scheduling, Systems & Transforms, ECS Components & Memory, and Arena & Hardware]({{file:file-Rq7nk8ivfShWs7xAB9HusU}})

1. **Application & API.**  Exposes `compress()`, `decompress()`, `World`, and the fluent `Pipe`.  This layer contains no compression logic; it orchestrates systems, schedules batches and collects traces.
2. **Pipeline & Scheduling.**  Maintains the sequence of systems to run for a given job.  Handles branching (`select`, `fork`), batching, and context configuration (e.g. device, data type).  It logs per‑step trace metadata (shapes, dtypes, entropy estimates, timing).
3. **Systems & Transforms.**  Houses implementations of individual steps: VAE encode/decode (ONNX), transforms (Hadamard, Wavelet, Blur), quantisers, entropy coders (ANS), and evaluation metrics.  Systems declare required input components and produced output components; the scheduler resolves dependencies.
4. **ECS Components & Memory.**  Defines lightweight data containers (RGB, Latent4, YUVW4, WaveletPyr, Symbols, Bitstream, Residual, etc.) that wrap `TensorRef`s into the arena.  The `World` manages entities (jobs) and stores per‑component dictionaries mapping entity IDs to components.
5. **Arena & Hardware.**  Implements the zero‑copy memory model.  A single arena holds all tensor data.  Hardware‑specific providers (CPU, CUDA) are selected when creating ONNX sessions【581473975824958†L44-L60】.

## Component Architecture

The following diagram summarises the major components and their relationships:

![Component and system relationships showing Arena, World, Entities, Components and Systems]({{file:file-WezKMzZ5q6NbRFJhdacV4b}})

### Arena & TensorRef

* **Arena** – A bump allocator allocating a large contiguous byte buffer.  It exposes `alloc_tensor(shape, dtype)` to carve out typed views with proper alignment.  When an arena is constructed, its size (e.g. 512 MB) determines the maximum number of jobs and intermediate buffers that can be held simultaneously.
* **TensorRef** – An immutable handle (`offset`, `shape`, `dtype`, `strides`) pointing into the arena.  `view(ref)` returns a NumPy view into the same memory without copying.  `subref_axis0(ref, i)` returns a slice view along the first dimension; this is used to slice batch buffers into per‑entity tensors.

### World & ECS store

* **World** – Maintains a monotonically increasing entity id and component stores.  Each component type has its own dictionary mapping `entity → component`.  The world provides helper methods to spawn batches of RGB images into contiguous buffers (`spawn_batch_rgb()`), allocate scratch buffers for intermediate batched operations and attach new components to entities.
* **Entity** – A simple integer identifier.  Entities represent compression jobs.  At any point an entity may have a subset of components attached (e.g. `RGB`, `Latent4`, `WaveletPyr`, `Symbols`, `Bitstream`, `ReconRGB`, `Residual`).

### Components

Components wrap `TensorRef`s and attach metadata:

- **RGB** – Holds an `(H,W,3)` uint8 tensor and a colours‑space string.
- **Latent4** – Holds a `(4,h,w)` float tensor representing the four channels output by the VAE encoder.
- **YUVW4** – Holds a Hadamard‑rotated latent.
- **WaveletPyr** – Contains packed wavelet coefficients and an index table referencing sub‑bands and levels.
- **Symbols/QuantParams** – Integer symbols representing quantised coefficients; `QuantParams` stores per‑channel scale and zero points.
- **Bitstream** – The ANS‑encoded sequence of words; also stores the model’s probability table and number of symbols.
- **Residual** – A signed tensor storing the pixel‑wise difference between source and reconstruction.
- **ReconRGB/BlurRGB** – Reconstructed and optionally blurred images.

### Systems

Systems implement transformations.  They declare the components they require and the components they produce.  Examples include:

- **OnnxVAEEncode / OnnxVAEDecode** – Loads ONNX models into an `InferenceSession`.  `session.run()` executes the encoder or decoder【581473975824958†L44-L60】.  The batched encoder groups entities by image shape and fills a batched `(N,3,H,W)` tensor before a single call to `run()`.
- **Hadamard4** – Performs an orthonormal 4×4 Hadamard transform on the latent channels.  It can operate in‑place or write to a separate component.
- **WaveletCDF53 / WaveletHaar** – Decomposes a tensor into multi‑level wavelet bands.  Output is normalised into a packed layout to be ECS‑friendly.
- **QuantizeU8 / Dequantize** – Scales and shifts floats to `uint8` (or other bit depths) and stores the scaling parameters.  Dequantisation inverts the mapping.
- **ANSEncode / ANSDecode** – Uses constriction’s `AnsCoder` and `Categorical` model.  It computes symbol histograms, builds a probability table, encodes all symbols into a sequence of 32‑bit words and stores them in the arena.  Decoding reconstructs the symbol tensor from the bitstream.
- **GaussianBlur** – Optionally blurs an image using a convolution kernel, implemented with a separable filter.  Useful for visualising reconstruction quality.
- **Residual / AddResidual** – Computes the residual between source and reconstruction and adds it back during decoding.
- **Metrics** – Compute PSNR, SSIM and MS‑SSIM using scikit‑image functions【623168233359239†L489-L512】【623168233359239†L558-L612】.  Metrics write scalar results into per‑entity metadata or a report structure.

### Pipeline & API

The `Pipe` object allows chaining systems using `|` or `.to()` and switching components with `.select()`/`.use()`.  Pipelines can branch: you can run a side pipeline on the same entity by calling `.fork()` and continue the main path.  Systems are run in the order added; the scheduler groups entities with identical shapes and runs batch operations whenever possible.

## Class Overview

Below is a textual class overview.  Relationships are shown using arrow notation (→ composition, ◇ aggregation, ▷ inheritance):

```
Arena
  + alloc_tensor(shape, dtype) → TensorRef
  + view(ref) → numpy.ndarray

TensorRef (offset, shape, dtype, strides)

World
  ◇ Arena
  + new_entity() → eid
  + spawn_batch_rgb(images) → eids
  + scratch(name, shape, dtype) → TensorRef
  + component stores: rgb[eid], latent[eid], wavelet[eid], symbols[eid], bitstream[eid], ...

Entity (alias: int)

Component classes wrap TensorRefs:
  RGB {pix: TensorRef, colourspace: str}
  Latent4 {z: TensorRef}
  YUVW4 {t: TensorRef}
  WaveletPyr {packed: TensorRef, index: TensorRef, levels: int, wavelet: str}
  SymbolsU8 {q: TensorRef, qp: QuantParams}
  QuantParams {scale: np.ndarray, zero: np.ndarray}
  ANSBitstream {probs: np.ndarray, words_u32: TensorRef, n_symbols: int}
  Residual {r: TensorRef}

System (abstract)
  + run(world: World, eids: list[int]) → None

Derived systems (inherit System):
  OnnxVAEEncode, OnnxVAEDecode
  Hadamard4, Hadamard4Inverse
  WaveletCDF53, WaveletCDF53Inverse
  QuantizeU8, Dequantize
  ANSEncode, ANSDecode
  GaussianBlur
  Residual, AddResidual
  MetricPSNR, MetricSSIM, MetricMSSSIM

Pipeline
  ◇ World
  + systems: list[System]
  + run(eids) – runs all systems in order
  + | operator adds a system
  + .to(step) – alias to add system
  + .select(key) / .use(key) – switch current component
  + .out(key) – return component handle for user
```

## File Tree Overview

```
vaerans_ecs/
  __init__.py
  api.py                # compress(), decompress(), Pipe facade
  core/
    arena.py            # Arena and TensorRef
    world.py            # World, entity and component stores
    pipeline.py         # Pipe and scheduling
    context.py          # device/dtype policies and tracing
  components/
    image.py            # RGB, ReconRGB, BlurRGB
    latent.py           # Latent4, YUVW4
    wavelet.py          # WaveletPyr, packing utilities
    quant.py            # SymbolsU8, QuantParams
    entropy.py          # ANSBitstream, probability tables
    residual.py         # Residual tensors
  systems/
    vae.py              # OnnxVAEEncode/Decode systems and model loader
    hadamard.py         # Hadamard4 forward/inverse
    wavelet.py          # WaveletCDF53 forward/inverse
    quantize.py         # QuantizeU8 / Dequantize
    ans.py              # ANSEncode / ANSDecode wrappers over constriction
    blur.py             # GaussianBlur
    residual.py         # Residual / AddResidual
    metrics.py          # PSNR, SSIM, MS‑SSIM systems
  eval/
    report.py           # HTML/JSON report generation from traces and metrics
  viz/
    plots.py            # RD curves, histograms, residual maps
```

Additional directories (e.g. `vae/backends`) can be added if multiple VAE back‑ends (ONNX, PyTorch, ONNXRuntime with I/O binding, etc.) are supported.

## Extensibility & Modularity

- **Adding new transforms or coders:** implement a new `System` subclass specifying required input component(s) and output component(s).  For example, adding a discrete cosine transform (DCT) would involve creating `DCTEncodeSys` and `DCTDecodeSys` classes and adding them to the registry.
- **Batching and grouping:** the scheduler groups entities by identical tensor shapes and runs a single system invocation (e.g. one `session.run()` for all images in a group).  Entities with different image sizes can still be processed in the same world; they are batched separately.
- **Branching pipelines:** `.select(key)` returns to a previously produced component, and `.fork()` duplicates the pipeline cursor, enabling multi‑stream codecs (e.g. latent stream + residual stream).
- **Pluggable backends:** VAE systems can use different execution providers (CPU, CUDA).  The ONNX Runtime API supports specifying providers when constructing an `InferenceSession`【581473975824958†L44-L60】 and supports I/O binding for device‑resident tensors【581473975824958†L120-L133】.
- **Instrumentation:** the context/tracing layer records per‑system metadata (shapes, dtypes, entropies, timings, seeds).  Reports summarise bitrates, PSNR, SSIM and MS‑SSIM across pipelines.

## Design Rationale

1. **ECS & zero‑copy.**  Zero‑copy transfer avoids unnecessary data copies and reduces latency; by storing all tensors in a contiguous arena and passing only lightweight references, the SDK avoids user‑space ↔ kernel‑space copies【580669371589968†L22-L33】.  The ECS pattern separates data (components) from behaviour (systems) and allows dynamic composition of pipelines, batch processing and optional operations.
2. **Variational autoencoder.**  A VAE is a generative model that learns a continuous, probabilistic latent space and reconstructs inputs via a decoder【326889321455777†L228-L244】.  Using a pretrained VAE compresses high‑dimensional image data into a small number of latent channels; using a probabilistic latent space allows for controlled lossy compression.
3. **Optimised entropy coding.**  The constriction library provides highly optimised range‑ANSCoders with near‑optimal bit rate and high speed【583281890738763†L108-L112】.  Using ANS instead of Huffman or arithmetic coding yields better compression at similar or faster speeds.
4. **ONNX Runtime.**  ONNX models can be loaded and run with an `InferenceSession`【581473975824958†L44-L60】.  Execution providers (CPU/GPU) and session options can be configured; on GPU, I/O binding avoids extra device ↔ host copies【581473975824958†L120-L133】.
5. **Standardised metrics.**  PSNR and SSIM from scikit‑image provide objective measures of reconstruction fidelity【623168233359239†L489-L512】【623168233359239†L558-L612】.  Exposing these metrics allows users to evaluate trade‑offs between compression rate and quality.

## Conclusion

This document presents a complete model‑driven architecture for a DX‑oriented image compression SDK.  By combining deep latent models (VAE), classical transforms (Hadamard, wavelets), optimised entropy coding (rANS) and a zero‑copy ECS execution model, the SDK offers efficient, modular and extensible compression pipelines.  Developers can compose transforms with a fluent API, exploit batching for high throughput and extend the system with new components, coders and metrics without modifying core code.  The layered design, clear component boundaries and strong separation of concerns enable maintainability and scalability.
