"""ONNX-based VAE encoder/decoder systems.

Systems for running Variational Autoencoder models via ONNX Runtime.
Supports batching multiple entities with the same shape in a single inference call.
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

import numpy as np

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None  # type: ignore[assignment, unused-ignore]

# TOML loading for Python 3.11+ and older
try:
    import tomllib  # type: ignore[import-not-found, unused-ignore]
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef, unused-ignore]

from vaerans_ecs.components.image import RGB, ReconRGB
from vaerans_ecs.components.latent import Latent4
from vaerans_ecs.core.system import System
from vaerans_ecs.core.world import World


def _resolve_config_path(config_path: str | None) -> str:
    """Resolve configuration path from env, explicit path, or defaults."""
    env_config = os.environ.get("VAERANS_CONFIG")
    if env_config:
        return env_config
    if config_path:
        return config_path
    # Look for vaerans_ecs.toml in current directory or home
    candidates = [
        "vaerans_ecs.toml",
        os.path.expanduser("~/vaerans_ecs.toml"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Config file not found. Set VAERANS_CONFIG or create vaerans_ecs.toml"
    )


def _load_model_config(model_name: str, config_path: str | None) -> tuple[dict[str, Any], str]:
    """Load model-specific configuration and return it with resolved config path."""
    resolved_path = _resolve_config_path(config_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Config file not found at {resolved_path}. Set VAERANS_CONFIG or create vaerans_ecs.toml"
        )
    with open(resolved_path, "rb") as f:
        config = cast(dict[str, Any], tomllib.load(f))
    try:
        model_cfg = cast(dict[str, Any], config["models"][model_name])
    except KeyError as e:
        raise ValueError(
            f"Model '{model_name}' not configured in {resolved_path}"
        ) from e
    return model_cfg, resolved_path


def _parse_range(value: Any, field_name: str) -> str | None:
    """Normalize range config values."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "")
        if normalized in {"0_1", "0-1", "0to1", "zero_one", "unit", "0..1"}:
            return "0_1"
        if normalized in {
            "-1_1",
            "-1-1",
            "-1to1",
            "-1..1",
            "minus1_1",
            "neg1_1",
            "minus1to1",
            "neg1to1",
        }:
            return "-1_1"
    raise ValueError(f"Unsupported {field_name} range value: {value!r}")


class OnnxVAEEncode(System):
    """VAE encoder system using ONNX Runtime.

    Transforms RGB images to latent representations.
    - Input: RGB component with shape (H, W, 3) uint8 or float32
    - Output: Latent4 component with shape (4, H/8, W/8) float32

    Attributes:
        model: Model name (must be configured in vaerans_ecs.toml)
        session: ONNX Runtime session
        input_name: Input tensor name in ONNX model
        output_name: Output tensor name in ONNX model
    """

    def __init__(
        self,
        model: str = "sdxl-vae",
        mode: Literal["encode", "forward"] = "encode",
        config_path: str | None = None,
    ) -> None:
        """Initialize VAE encoder.

        Args:
            model: Model name (from config)
            mode: Transformation mode (encode or forward)
            config_path: Path to vaerans_ecs.toml (auto-detected if None)
        """
        super().__init__(mode=mode)

        if not HAS_ORT:
            raise ImportError(
                "onnxruntime is required for VAE systems. "
                "Install with: pip install onnxruntime"
            )

        self.model_name = model
        self.config_path = config_path
        self.session: Any = None
        self.input_name = "input"
        self.output_name = "output"

        # Load model on first use
        self._session = None
        self._model_config: dict[str, Any] | None = None
        self._resolved_config_path: str | None = None

        # Optional preprocessing/scaling hints
        self._input_range: str | None = None
        self._latent_scale: float | None = None

    @property
    def session_lazy(self) -> Any:
        """Lazy-load ONNX session."""
        if self._session is None:
            self._session = self._load_session()
        return self._session

    def _get_model_config(self) -> dict[str, Any]:
        """Load and cache model config."""
        if self._model_config is None:
            self._model_config, self._resolved_config_path = _load_model_config(
                self.model_name, self.config_path
            )
        return self._model_config

    def _get_input_range(self) -> str | None:
        if self._input_range is None:
            model_cfg = self._get_model_config()
            self._input_range = _parse_range(model_cfg.get("input_range"), "input_range")
        return self._input_range

    def _get_latent_scale(self) -> float | None:
        if self._latent_scale is None:
            model_cfg = self._get_model_config()
            value = model_cfg.get("latent_scale", model_cfg.get("scaling_factor"))
            self._latent_scale = float(value) if value is not None else None
        return self._latent_scale

    def _load_session(self) -> Any:
        """Load ONNX model from config path."""
        model_path = self._get_model_path()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Create ONNX Runtime session
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Auto-detect input/output names from model
        if session.get_inputs():
            self.input_name = session.get_inputs()[0].name
        if session.get_outputs():
            self.output_name = session.get_outputs()[0].name

        return session

    def _get_model_path(self) -> str:
        """Get model path from config."""
        model_cfg = self._get_model_config()
        config_path = self._resolved_config_path
        if config_path is None:
            raise FileNotFoundError(
                "Config file not found. Set VAERANS_CONFIG or create vaerans_ecs.toml"
            )

        try:
            encoder_path = cast(str, model_cfg["encoder"])
        except KeyError as e:
            raise ValueError(
                f"Model '{self.model_name}' missing encoder path in {config_path}"
            ) from e

        # Expand relative paths relative to config file
        if not os.path.isabs(encoder_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            encoder_path = os.path.join(config_dir, encoder_path)

        return encoder_path

    def required_components(self) -> list[type]:
        """Return required input components."""
        return [RGB]

    def produced_components(self) -> list[type]:
        """Return produced output components."""
        return [Latent4]

    def run(self, world: World, eids: list[int]) -> None:
        """Apply VAE encoder to entities.

        Batches entities with the same image shape for efficient inference.
        """
        if not eids:
            return

        # Group by image shape for batching
        by_shape: dict[tuple[int, int, int], list[int]] = {}
        for eid in eids:
            rgb = world.get_component(eid, RGB)
            rgb_view = world.arena.view(rgb.pix)
            shape = rgb_view.shape
            if shape not in by_shape:
                by_shape[shape] = []
            by_shape[shape].append(eid)

        # Process each shape group
        for shape, batch_eids in by_shape.items():
            self._encode_batch(world, batch_eids)

    def _encode_batch(self, world: World, eids: list[int]) -> None:
        """Encode a batch of entities with the same shape."""
        if not eids:
            return

        # Get first entity to determine shape
        rgb = world.get_component(eids[0], RGB)
        rgb_view = world.arena.view(rgb.pix)

        # Handle uint8 to float32 conversion if needed
        if rgb_view.dtype == np.uint8:
            # Normalize to [0, 1]
            rgb_normalized = rgb_view.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_view.astype(np.float32)

        # Normalize to model expected range if configured
        input_range = self._get_input_range()
        if input_range == "-1_1":
            rgb_normalized = rgb_normalized * 2.0 - 1.0

        # Stack batch (add batch dimension if needed)
        if len(eids) == 1:
            batch_input = np.expand_dims(rgb_normalized, 0)  # (1, H, W, 3)
        else:
            # Multiple images with same shape - stack them
            images = []
            for eid in eids:
                view = world.arena.view(world.get_component(eid, RGB).pix)
                if view.dtype == np.uint8:
                    img = view.astype(np.float32) / 255.0
                else:
                    img = view.astype(np.float32)
                if input_range == "-1_1":
                    img = img * 2.0 - 1.0
                images.append(img)
            batch_input = np.stack(images, axis=0)  # (N, H, W, 3)

        # Convert CHW format if needed (most models expect this)
        if batch_input.shape[-1] == 3:  # HWC format
            batch_input = np.transpose(batch_input, (0, 3, 1, 2))  # NCHW

        # Run inference
        outputs = self.session_lazy.run(None, {self.input_name: batch_input})
        latents = outputs[0]  # (N, 4, h, w)

        latent_scale = self._get_latent_scale()
        if latent_scale is not None:
            latents = latents * latent_scale

        # Attach latent component to each entity
        for i, eid in enumerate(eids):
            latent_data = latents[i]  # (4, h, w)
            latent_ref = world.arena.copy_tensor(latent_data)
            world.add_component(eid, Latent4(z=latent_ref))


class OnnxVAEDecode(System):
    """VAE decoder system using ONNX Runtime.

    Transforms latent representations to RGB images.
    - Input: Latent4 component with shape (4, h, w) float32
    - Output: ReconRGB component with shape (H, W, 3) float32

    Attributes:
        model: Model name (must be configured in vaerans_ecs.toml)
        session: ONNX Runtime session
        input_name: Input tensor name in ONNX model
        output_name: Output tensor name in ONNX model
    """

    def __init__(
        self,
        model: str = "sdxl-vae",
        mode: Literal["decode", "inverse"] = "decode",
        config_path: str | None = None,
    ) -> None:
        """Initialize VAE decoder.

        Args:
            model: Model name (from config)
            mode: Transformation mode (decode or inverse)
            config_path: Path to vaerans_ecs.toml (auto-detected if None)
        """
        super().__init__(mode=mode)

        if not HAS_ORT:
            raise ImportError(
                "onnxruntime is required for VAE systems. "
                "Install with: pip install onnxruntime"
            )

        self.model_name = model
        self.config_path = config_path
        self._session = None
        self.input_name = "input"
        self.output_name = "output"
        self._model_config: dict[str, Any] | None = None
        self._resolved_config_path: str | None = None

        self._output_range: str | None = None
        self._latent_scale: float | None = None
        self._decoder_expects_scaled: bool | None = None

    @property
    def session_lazy(self) -> Any:
        """Lazy-load ONNX session."""
        if self._session is None:
            self._session = self._load_session()
        return self._session

    def _get_model_config(self) -> dict[str, Any]:
        """Load and cache model config."""
        if self._model_config is None:
            self._model_config, self._resolved_config_path = _load_model_config(
                self.model_name, self.config_path
            )
        return self._model_config

    def _get_output_range(self) -> str | None:
        if self._output_range is None:
            model_cfg = self._get_model_config()
            self._output_range = _parse_range(model_cfg.get("output_range"), "output_range")
        return self._output_range

    def _get_latent_scale(self) -> float | None:
        if self._latent_scale is None:
            model_cfg = self._get_model_config()
            value = model_cfg.get("latent_scale", model_cfg.get("scaling_factor"))
            self._latent_scale = float(value) if value is not None else None
        return self._latent_scale

    def _decoder_expects_scaled_latent(self) -> bool:
        if self._decoder_expects_scaled is None:
            model_cfg = self._get_model_config()
            value = model_cfg.get("decoder_expects_scaled_latent")
            if value is None:
                # Default to True when a latent scale is provided.
                self._decoder_expects_scaled = self._get_latent_scale() is not None
            else:
                self._decoder_expects_scaled = bool(value)
        return self._decoder_expects_scaled

    def _load_session(self) -> Any:
        """Load ONNX model from config path."""
        model_path = self._get_model_path()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Create ONNX Runtime session
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Auto-detect input/output names from model
        if session.get_inputs():
            self.input_name = session.get_inputs()[0].name
        if session.get_outputs():
            self.output_name = session.get_outputs()[0].name

        return session

    def _get_model_path(self) -> str:
        """Get model path from config."""
        model_cfg = self._get_model_config()
        config_path = self._resolved_config_path
        if config_path is None:
            raise FileNotFoundError(
                "Config file not found. Set VAERANS_CONFIG or create vaerans_ecs.toml"
            )

        try:
            decoder_path = cast(str, model_cfg["decoder"])
        except KeyError as e:
            raise ValueError(
                f"Model '{self.model_name}' missing decoder path in {config_path}"
            ) from e

        # Expand relative paths relative to config file
        if not os.path.isabs(decoder_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            decoder_path = os.path.join(config_dir, decoder_path)

        return decoder_path

    def required_components(self) -> list[type]:
        """Return required input components."""
        return [Latent4]

    def produced_components(self) -> list[type]:
        """Return produced output components."""
        return [ReconRGB]

    def run(self, world: World, eids: list[int]) -> None:
        """Apply VAE decoder to entities.

        Batches entities with the same latent shape for efficient inference.
        """
        if not eids:
            return

        # Group by latent shape for batching
        by_shape: dict[tuple[int, int, int], list[int]] = {}
        for eid in eids:
            latent = world.get_component(eid, Latent4)
            latent_view = world.arena.view(latent.z)
            shape = latent_view.shape
            if shape not in by_shape:
                by_shape[shape] = []
            by_shape[shape].append(eid)

        # Process each shape group
        for shape, batch_eids in by_shape.items():
            self._decode_batch(world, batch_eids)

    def _decode_batch(self, world: World, eids: list[int]) -> None:
        """Decode a batch of entities with the same shape."""
        if not eids:
            return

        # Get latents and stack
        latents = [
            world.arena.view(world.get_component(eid, Latent4).z) for eid in eids
        ]

        if len(latents) == 1:
            batch_input = np.expand_dims(latents[0], 0)  # (1, 4, h, w)
        else:
            batch_input = np.stack(latents, axis=0)  # (N, 4, h, w)

        latent_scale = self._get_latent_scale()
        if latent_scale is not None and not self._decoder_expects_scaled_latent():
            batch_input = batch_input / latent_scale

        # Run inference
        outputs = self.session_lazy.run(None, {self.input_name: batch_input})
        images = outputs[0]  # (N, 3, H, W) or (N, H, W, 3)

        # Convert to HWC if needed
        if images.shape[1] == 3:  # CHW format
            images = np.transpose(images, (0, 2, 3, 1))  # NHWC

        output_range = self._get_output_range()
        if output_range == "-1_1":
            images = (images + 1.0) / 2.0

        # Clip to [0, 1] range
        images = np.clip(images, 0.0, 1.0)

        # Attach reconstruction component to each entity
        for i, eid in enumerate(eids):
            image_data = images[i]  # (H, W, 3)
            recon_ref = world.arena.copy_tensor(image_data)
            world.add_component(eid, ReconRGB(pix=recon_ref, colorspace="sRGB"))
