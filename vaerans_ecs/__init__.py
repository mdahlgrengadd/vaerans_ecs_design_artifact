"""VAE+ANS Image Compression SDK with ECS Architecture.

This package provides a developer-friendly image compression SDK using:
- Variational Autoencoder (VAE) for learned image representation
- Asymmetric Numeral Systems (ANS) for entropy coding
- Entity-Component-System (ECS) architecture for flexibility
- Zero-copy memory management via Arena allocation

Quick Start:
    >>> from vaerans_ecs import compress, decompress
    >>> import numpy as np
    >>>
    >>> # Create test image
    >>> img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    >>>
    >>> # Compress and decompress
    >>> compressed = compress(img, model='sdxl-vae', quality=50)
    >>> reconstructed = decompress(compressed)

For more control, use the fluent pipeline API:
    >>> from vaerans_ecs import World
    >>> from vaerans_ecs.systems import OnnxVAEEncode, Hadamard4
    >>>
    >>> world = World()
    >>> entity = world.spawn_image(img)
    >>>
    >>> # Build custom pipeline
    >>> result = (
    ...     world.pipe(entity)
    ...     .to(OnnxVAEEncode(model='sdxl-vae', mode='encode'))
    ...     .to(Hadamard4(mode='forward'))
    ...     .out(YUVW4)
    ... )
"""

__version__ = "0.1.0"

# High-level API will be exposed after Phase 13
# from vaerans_ecs.api import compress, decompress

# Core classes will be exposed as implementation progresses
# from vaerans_ecs.core.world import World
# from vaerans_ecs.core.arena import Arena, TensorRef

__all__ = [
    "__version__",
    # "compress",
    # "decompress",
    # "World",
    # "Arena",
    # "TensorRef",
]
