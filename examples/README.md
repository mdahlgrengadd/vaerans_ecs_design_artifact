# Examples

These examples demonstrate common workflows using the VAE+ANS ECS SDK. The default
configuration in `vaerans_ecs.toml` points to the real SDXL VAE ONNX models in
`models/`.

## 📓 Interactive Tutorial (Google Colab)

**File:** `tutorial.ipynb` ⭐ **RECOMMENDED FOR BEGINNERS**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/vaerans_ecs_design_artifact/blob/main/examples/tutorial.ipynb)

Interactive Jupyter notebook that runs on Google Colab with zero setup required!

**Features:**
- 🚀 **No installation needed**: Runs directly in your browser
- 📦 **Automatic setup**: Clones repo and installs dependencies
- 🎓 **Learn by doing**: Step-by-step guide with live code
- 📊 **Visual results**: See compression results with matplotlib
- ⚡ **Five tutorials in one**:
  1. High-level `compress()`/`decompress()` API
  2. Low-level ECS pipeline construction
  3. Batch processing
  4. Fluent pipeline API
  5. Quality metrics and visualization

**Quick Start:**
1. Click the "Open in Colab" badge above
2. Click "Runtime" → "Run all"
3. Watch the magic happen! ✨

*Note: Update `YOUR_USERNAME` in the badge link with your GitHub username. See `COLAB_SETUP.md` for details.*

---

## Quick Start Examples

### 1. High-Level API (Recommended for Beginners)
**File:** `quickstart_api.py`

Uses the Phase 13 high-level `compress()` and `decompress()` APIs for simple compression.

```bash
python examples/quickstart_api.py --input examples/23.png --output examples/reconstruction_api.png --quality 80
```

**Features:**
- One-line compress/decompress
- Quality parameter (1-100)
- Automatic compression ratio reporting
- Built-in PSNR metrics

---

### 2. Full Compression Pipeline
**File:** `full_pipeline_example.py` ⭐ **NEW**

Demonstrates the complete compression pipeline with all stages:
- VAE Encode → Hadamard → Wavelet → Quantize → ANS → Serialization
- Reverse decompression pipeline
- Quality metrics (PSNR, SSIM)
- Compression statistics

```bash
python examples/full_pipeline_example.py --input examples/23.png --quality 80 --wavelet-levels 4
```

**Features:**
- Complete Phases 1-12 integration
- Serialization/deserialization (Phase 9)
- Quality metrics (Phase 12)
- Detailed progress reporting
- Compression ratio and bits-per-pixel stats

---

### 3. Quality Metrics Demo
**File:** `metrics_example.py` ⭐ **NEW**

Focused demonstration of quality evaluation systems (Phase 12):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- MS-SSIM (Multi-Scale SSIM)

```bash
python examples/metrics_example.py --input examples/23.png
```

**Features:**
- All quality metrics in one place
- Interpretation guide
- Quality assessment thresholds
- Implementation notes

---

## Advanced Examples

### 4. Fluent Pipeline API - Complete
**File:** `fluent_api_complete.py` ⭐ **NEW**

Complete demonstration of the fluent pipeline API (Phase 11) with all features:
- Method chaining (`.to()`)
- Pipe operator (`|`)
- Component selection (`.select()`, `.use()`)
- Output extraction (`.out()`)
- Integration with all systems

```bash
python examples/fluent_api_complete.py --input examples/23.png --quality 80
```

**Features:**
- 6 progressive examples showing API capabilities
- Type-safe pipeline construction
- IDE-friendly syntax
- All systems integration

---

### 5. Fluent Pipeline API - Basic
**File:** `pipeline_example.py`

Demonstrates the fluent pipe API with a lightweight dummy encoder and Hadamard transforms.

```bash
python examples/pipeline_example.py
```

**Features:**
- Basic `.to()` chaining
- Pipe operator `|`
- Component selection
- No ONNX models required (uses dummy VAE)

---

### 6. Batch Processing
**File:** `batch_pipeline.py`

Shows how to batch-encode/decode multiple images efficiently with `World` + systems.

```bash
python examples/batch_pipeline.py --count 5 --size 256 --model sdxl-vae
```

**Features:**
- Batch image processing
- Quality metrics for all images (Phase 12)
- Summary statistics (average PSNR/SSIM)
- Optional Hadamard transform

---

### 7. Real VAE Round Trip
**File:** `test_real_vae.py`

Runs a single VAE encode/decode pass and computes quality metrics.

```bash
python examples/test_real_vae.py --output examples/reconstruction.png --size 512
```

**Features:**
- VAE encode/decode validation
- Quality metrics (PSNR, SSIM) via Phase 12 systems
- Detailed latent inspection
- Image export

---

## Example Comparison

| Example | Difficulty | Use Case | Key Features |
|---------|-----------|----------|--------------|
| `quickstart_api.py` | ⭐ Beginner | Simple compress/decompress | Phase 13 API, one-liners |
| `full_pipeline_example.py` | ⭐⭐ Intermediate | Complete pipeline | All phases, serialization |
| `metrics_example.py` | ⭐⭐ Intermediate | Quality evaluation | All metrics, interpretation |
| `fluent_api_complete.py` | ⭐⭐⭐ Advanced | Pipeline construction | Fluent API mastery |
| `pipeline_example.py` | ⭐⭐ Intermediate | API basics | Simple chaining |
| `batch_pipeline.py` | ⭐⭐ Intermediate | Multiple images | Batch processing |
| `test_real_vae.py` | ⭐ Beginner | VAE validation | Basic encode/decode |

---

## Implementation Phases Demonstrated

| Phase | Feature | Examples |
|-------|---------|----------|
| Phase 1 | Arena & TensorRef | All |
| Phase 2 | World & Components | All |
| Phase 3 | VAE Systems | All (except `pipeline_example.py`) |
| Phase 4 | Hadamard Transform | `full_pipeline_example.py`, `batch_pipeline.py`, `quickstart_api.py` |
| Phase 6 | Wavelet Transform | `full_pipeline_example.py` |
| Phase 7 | Quantization | `full_pipeline_example.py` |
| Phase 8 | ANS Entropy Coding | `full_pipeline_example.py` |
| Phase 9 | Serialization | `full_pipeline_example.py` |
| Phase 11 | Fluent Pipeline API | `pipeline_example.py`, `fluent_api_complete.py` |
| Phase 12 | Quality Metrics | `metrics_example.py`, all updated examples |
| Phase 13 | High-Level API | `quickstart_api.py` |

---

## Common Options

All examples support these common options:

- `--config PATH`: Path to `vaerans_ecs.toml` configuration file
- `--model NAME`: Model name from config (default: `sdxl-vae`)
- `--input PATH`: Input image path (defaults to `examples/23.png`)
- `--output PATH`: Output image path for reconstruction
- `--size N`: Image size for random generation if no input

Additional options vary by example (use `--help` to see all options).

---

## Notes

- If you keep configs outside the repo root, pass `--config` or set
  `VAERANS_CONFIG` to your `vaerans_ecs.toml` path.
- Image save/load uses Pillow when available (install with `pip install Pillow`).
- The `full_pipeline_example.py` demonstrates all implemented compression stages.
- Metrics examples require `scikit-image` (included in dev dependencies).
- All examples use the ECS architecture with zero-copy Arena memory management.
