# Examples

These scripts demonstrate common workflows using the VAE+ANS SDK. The default
configuration in `vaerans_ecs.toml` points to the real SDXL VAE ONNX models in
`models/`.

## Quickstart API
- `quickstart_api.py` uses the high-level `compress()` and `decompress()` APIs.
- Example:
  - `python examples/quickstart_api.py --input examples/23.png --output examples/reconstruction_api.png`

## Batch Pipeline
- `batch_pipeline.py` shows how to batch-encode/decode multiple images with
  `World` + `OnnxVAEEncode/OnnxVAEDecode`.
- Example:
  - `python examples/batch_pipeline.py --count 3 --size 256`

## Real VAE Round Trip
- `test_real_vae.py` runs a single encode/decode pass and prints metrics.
- Example:
  - `python examples/test_real_vae.py --output examples/reconstruction.png`

## Fluent Pipeline API (Dummy VAE)
- `pipeline_example.py` demonstrates the fluent pipe API with a lightweight
  dummy encoder and Hadamard transforms.
- Example:
  - `python examples/pipeline_example.py`

## Notes
- If you keep configs outside the repo root, pass `--config` or set
  `VAERANS_CONFIG` to your `vaerans_ecs.toml` path.
- Image save/load uses Pillow when available.
