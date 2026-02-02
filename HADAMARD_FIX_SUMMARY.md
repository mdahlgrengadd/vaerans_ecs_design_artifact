# Hadamard Transform Bug Fix

## Problem Description

The `compress()` and `decompress()` functions in `vaerans_ecs/api.py` had a critical mismatch that caused incorrect image reconstruction.

### The Bug

**In `compress()` (with `use_hadamard=True`, which is the default):**
```
RGB → VAE Encode → Latent4 → Hadamard Transform → YUVW4 → Serialize
```

**In `decompress()` (BEFORE fix):**
```
Deserialize → Latent4 → VAE Decode → ReconRGB
```

❌ **The problem**: When Hadamard was used during compression, the latent data was in YUVW4 space, but `decompress()` was treating it as Latent4 and feeding it directly to the VAE decoder without applying inverse Hadamard!

This caused the reconstructed images to look completely wrong because the decoder was receiving transformed coefficients instead of the expected latent representation.

## The Fix

### 1. Added `use_hadamard` to Serialization Metadata

**File**: `vaerans_ecs/core/serialization.py`

```python
# Before
metadata: dict[str, Any] = {
    "model": model,
    "wavelet_levels": levels,
    "image_shape": list(image_shape),
    "initial_state": bitstream.initial_state,
}

# After
metadata: dict[str, Any] = {
    "model": model,
    "wavelet_levels": levels,
    "image_shape": list(image_shape),
    "initial_state": bitstream.initial_state,
    "use_hadamard": use_hadamard,  # ✅ Added!
}
```

### 2. Updated `compress()` to Pass the Flag

**File**: `vaerans_ecs/api.py`

```python
return serialize_bitstream(
    bitstream,
    arena=world.arena,
    model=model,
    levels=1,
    image_shape=image.shape,
    quant_params=dummy_quant_params,
    use_hadamard=use_hadamard,  # ✅ Pass the flag!
)
```

### 3. Updated `decompress()` to Apply Inverse Hadamard

**File**: `vaerans_ecs/api.py`

**BEFORE:**
```python
# Always treated data as Latent4 (WRONG!)
latent_ref = world.arena.copy_tensor(latent_array)
world.add_component(entity, Latent4(z=latent_ref))

recon: ReconRGB = (
    world.pipe(entity)
    .to(OnnxVAEDecode(model=metadata["model"], mode="decode", config_path=config_path))
    .out(ReconRGB)
)
```

**AFTER:**
```python
# Check metadata and apply inverse Hadamard if needed
use_hadamard = metadata.get("use_hadamard", False)

if use_hadamard:
    # Data is in YUVW4 space, need inverse Hadamard before VAE decode
    world.add_component(entity, YUVW4(t=latent_ref))
    
    recon: ReconRGB = (
        world.pipe(entity)
        .to(Hadamard4(mode="inverse"))  # ✅ Apply inverse first!
        .to(OnnxVAEDecode(model=metadata["model"], mode="decode", config_path=config_path))
        .out(ReconRGB)
    )
else:
    # Data is in Latent4 space, decode directly
    world.add_component(entity, Latent4(z=latent_ref))
    
    recon: ReconRGB = (
        world.pipe(entity)
        .to(OnnxVAEDecode(model=metadata["model"], mode="decode", config_path=config_path))
        .out(ReconRGB)
    )
```

## Correct Pipeline Flow (AFTER Fix)

### With Hadamard (Default)

**Compression:**
```
RGB → VAE Encode → Latent4 → Hadamard Forward → YUVW4 → Serialize (use_hadamard=True)
```

**Decompression:**
```
Deserialize → YUVW4 → Hadamard Inverse → Latent4 → VAE Decode → ReconRGB ✅
```

### Without Hadamard

**Compression:**
```
RGB → VAE Encode → Latent4 → Serialize (use_hadamard=False)
```

**Decompression:**
```
Deserialize → Latent4 → VAE Decode → ReconRGB ✅
```

## Files Modified

1. `vaerans_ecs/core/serialization.py`
   - Added `use_hadamard` parameter to `serialize_bitstream()`
   - Store `use_hadamard` in metadata

2. `vaerans_ecs/api.py`
   - Updated `compress()` to pass `use_hadamard` flag
   - Updated `decompress()` to check metadata and apply inverse Hadamard

## Testing

The fix has been verified with the test script `test_compress_decompress_fix.py`:

```bash
python test_compress_decompress_fix.py
```

Expected results:
- ✅ `use_hadamard` metadata is correctly stored
- ✅ Decompression checks the flag
- ✅ Inverse Hadamard is applied when needed
- ✅ Both paths (with/without Hadamard) execute correctly

## Impact

### What Was Broken
- All calls to `compress()` with default settings (use_hadamard=True)
- Tutorial notebook's high-level API examples
- Any code using the `compress()`/`decompress()` API

### What Is Fixed Now
- ✅ High-level API correctly handles Hadamard transform
- ✅ Tutorial notebook will show correct reconstructions
- ✅ Image quality matches expectations
- ✅ Both Hadamard and non-Hadamard modes work correctly

## Backward Compatibility

### Old Compressed Files
Files compressed with the old code (before this fix) do NOT have the `use_hadamard` flag in metadata.

The `decompress()` function handles this gracefully:
```python
use_hadamard = metadata.get("use_hadamard", False)
```

**Default**: `False` (no Hadamard inverse)

This means:
- Old files compressed with `use_hadamard=False`: ✅ Will work correctly
- Old files compressed with `use_hadamard=True`: ❌ Will still be broken

**Recommendation**: Re-compress any images that were compressed with the old buggy code.

## Summary

This was a **critical bug** that made the default compression mode unusable. The fix ensures that:

1. The compression metadata tracks whether Hadamard was used
2. Decompression applies the inverse transform when needed
3. The pipeline is mathematically correct in both modes
4. Image reconstruction quality is preserved

The tutorial notebook and all examples using the high-level API will now work correctly! 🎉
