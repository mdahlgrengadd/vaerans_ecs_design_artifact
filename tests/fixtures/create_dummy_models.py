"""Create dummy ONNX models for testing VAE systems.

This module generates minimal but valid ONNX models that can be used for testing
the OnnxVAEEncode and OnnxVAEDecode systems without requiring real VAE models.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_dummy_vae_encoder(output_path: str) -> None:
    """Create minimal ONNX encoder for testing.

    Input: (1, 3, 256, 256) RGB image
    Output: (1, 4, 32, 32) latent tensor (8x downsampling)

    Args:
        output_path: Path to save the ONNX model
    """
    # Input: RGB image
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 256, 256])

    # Output: latent vector
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 32, 32])

    # Create a simple identity-like operation by reshaping
    # (in reality, this would be a full VAE encoder, but for testing we just pool)

    # Add initializers for constants
    squeeze_val = helper.make_tensor(
        name="squeeze_val",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[1.0],
    )

    # Create nodes for a simple downsampling operation
    # Conv with stride to downsample from 256x256 to 32x32 (8x reduction)

    # Weights: (4, 3, 3, 3) for 4 output channels, 3 input channels, 3x3 kernel
    conv_w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    conv_weights = helper.make_tensor(
        name="conv_weights",
        data_type=TensorProto.FLOAT,
        dims=list(conv_w.shape),
        vals=conv_w.tobytes(),
        raw=True,
    )

    # Bias: (4,) for 4 output channels
    conv_b = np.zeros(4, dtype=np.float32)
    conv_bias = helper.make_tensor(
        name="conv_bias",
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=conv_b.tobytes(),
        raw=True,
    )

    # Create Conv node with stride 8 to achieve 8x downsampling
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_weights", "conv_bias"],
        outputs=["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],  # Same padding
        strides=[8, 8],  # 8x stride = 8x downsampling
    )

    # Create graph
    graph_def = helper.make_graph(
        [conv_node],
        "vae_encoder",
        [X],
        [Y],
        [conv_weights, conv_bias],
    )

    # Create model with IR version 6 for compatibility
    model_def = helper.make_model(
        graph_def,
        producer_name="test_generator",
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model_def.ir_version = 6  # IR version 6 for ONNX opset 10

    # Save model
    onnx.save(model_def, output_path)


def create_dummy_vae_decoder(output_path: str) -> None:
    """Create minimal ONNX decoder for testing.

    Input: (1, 4, 32, 32) latent tensor
    Output: (1, 3, 256, 256) RGB image (8x upsampling)

    Args:
        output_path: Path to save the ONNX model
    """
    # Input: latent vector
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 32, 32])

    # Output: RGB image
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 256, 256])

    # Create a simple upsampling operation using ConvTranspose

    # Weights: (3, 4, 3, 3) for 3 output channels, 4 input channels, 3x3 kernel
    conv_t_w = np.random.randn(3, 4, 3, 3).astype(np.float32)
    conv_t_weights = helper.make_tensor(
        name="conv_t_weights",
        data_type=TensorProto.FLOAT,
        dims=list(conv_t_w.shape),
        vals=conv_t_w.tobytes(),
        raw=True,
    )

    # Bias: (3,) for 3 output channels
    conv_t_b = np.zeros(3, dtype=np.float32)
    conv_t_bias = helper.make_tensor(
        name="conv_t_bias",
        data_type=TensorProto.FLOAT,
        dims=[3],
        vals=conv_t_b.tobytes(),
        raw=True,
    )

    # Create ConvTranspose node with stride 8 to achieve 8x upsampling
    conv_t_node = helper.make_node(
        "ConvTranspose",
        inputs=["input", "conv_t_weights", "conv_t_bias"],
        outputs=["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[8, 8],  # 8x stride = 8x upsampling
        output_padding=[7, 7],  # Needed for exact size match
    )

    # Create graph
    graph_def = helper.make_graph(
        [conv_t_node],
        "vae_decoder",
        [X],
        [Y],
        [conv_t_weights, conv_t_bias],
    )

    # Create model with IR version 6 for compatibility
    model_def = helper.make_model(
        graph_def,
        producer_name="test_generator",
        opset_imports=[helper.make_opsetid("", 10)],
    )
    model_def.ir_version = 6  # IR version 6 for ONNX opset 10

    # Save model
    onnx.save(model_def, output_path)


if __name__ == "__main__":
    import os

    # Get directory of this file
    fixture_dir = os.path.dirname(__file__)

    # Create models
    encoder_path = os.path.join(fixture_dir, "dummy_vae_encoder.onnx")
    decoder_path = os.path.join(fixture_dir, "dummy_vae_decoder.onnx")

    print(f"Creating {encoder_path}...")
    create_dummy_vae_encoder(encoder_path)
    print(f"Created {encoder_path}")

    print(f"Creating {decoder_path}...")
    create_dummy_vae_decoder(decoder_path)
    print(f"Created {decoder_path}")
