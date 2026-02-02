"""ANS entropy coding systems using constriction.

Implements range Asymmetric Numeral Systems (rANS) encoding for
lossless compression of quantized symbols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import constriction
import numpy as np

from vaerans_ecs.components.entropy import ANSBitstream
from vaerans_ecs.components.quant import SymbolsU8
from vaerans_ecs.core.system import System

if TYPE_CHECKING:
    from vaerans_ecs.core.world import World


class ANSEncode(System):
    """Encode quantized symbols using rANS.

    Uses constriction library for high-performance entropy coding.
    Builds a categorical probability model from symbol histogram.

    Forward mode: SymbolsU8 → ANSBitstream
    """

    def __init__(self, mode: Literal["forward"] = "forward"):
        """Initialize ANS encoder.

        Args:
            mode: Must be 'forward' (encoding only)
        """
        super().__init__(mode=mode)
        if mode != "forward":
            raise ValueError("ANSEncode only supports forward mode")

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [SymbolsU8]

    def produced_components(self) -> list[type]:
        """Return produced component types."""
        return [ANSBitstream]

    def run(self, world: World, eids: list[int]) -> None:
        """Encode symbols using ANS.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        for eid in eids:
            symbols_comp = world.get_component(eid, SymbolsU8)
            symbols = world.arena.view(symbols_comp.data).ravel()

            # Build histogram
            hist = np.bincount(symbols.astype(np.int32), minlength=256)
            
            # Add-1 smoothing to avoid zero probabilities
            hist = hist + 1
            total = hist.sum()
            probs = hist.astype(np.float32) / total

            # Encode using constriction
            # ANS expects symbols as int32 for compatibility
            symbols_int = symbols.astype(np.int32)

            # Create probability model
            # constriction expects probabilities that sum to 1
            encoder = constriction.stream.stack.AnsCoder()
            model = constriction.stream.model.Categorical(probs, perfect=False)

            # Encode in reverse order (ANS requirement)
            encoder.encode_reverse(symbols_int, model)

            # Get compressed data
            compressed = encoder.get_compressed()

            # Convert to uint8 for storage
            # constriction returns uint32 words, we store as bytes
            compressed_bytes = np.array(compressed, dtype=np.uint32).tobytes()
            compressed_u8 = np.frombuffer(compressed_bytes, dtype=np.uint8)

            # Store in arena
            data_ref = world.arena.copy_tensor(compressed_u8)
            probs_ref = world.arena.copy_tensor(probs)

            # Create ANSBitstream component
            bitstream = ANSBitstream(
                data=data_ref,
                probs=probs_ref,
                initial_state=0,  # Start state for decoder
            )
            world.add_component(eid, bitstream)


class ANSDecode(System):
    """Decode rANS bitstream back to symbols.

    Uses constriction library for high-performance entropy decoding.

    Forward mode: ANSBitstream → SymbolsU8
    """

    def __init__(self, mode: Literal["forward"] = "forward"):
        """Initialize ANS decoder.

        Args:
            mode: Must be 'forward' (decoding only)
        """
        super().__init__(mode=mode)
        if mode != "forward":
            raise ValueError("ANSDecode only supports forward mode")

    def required_components(self) -> list[type]:
        """Return required component types."""
        return [ANSBitstream]

    def produced_components(self) -> list[type]:
        """Return produced component types."""
        return [SymbolsU8]

    def run(self, world: World, eids: list[int]) -> None:
        """Decode ANS bitstream.

        Args:
            world: World containing entities
            eids: List of entity IDs to process
        """
        for eid in eids:
            bitstream = world.get_component(eid, ANSBitstream)
            
            # Get compressed data
            compressed_u8 = world.arena.view(bitstream.data)
            
            # Convert bytes back to uint32 words
            compressed_bytes = compressed_u8.tobytes()
            compressed = np.frombuffer(compressed_bytes, dtype=np.uint32).tolist()

            # Get probability model
            probs = world.arena.view(bitstream.probs)

            # We need to know the number of symbols to decode
            # This should be stored with SymbolsU8 or in metadata
            # For now, we'll need to access the original SymbolsU8 to get the size
            # In a real implementation, we'd store this in the bitstream metadata
            
            if world.has_component(eid, SymbolsU8):
                # If original symbols exist, use their size
                orig_symbols = world.get_component(eid, SymbolsU8)
                orig_data = world.arena.view(orig_symbols.data)
                n_symbols = len(orig_data)
                
                # Also preserve the quantization parameters
                params = orig_symbols.params
            else:
                # This shouldn't happen in normal flow
                raise ValueError(
                    "Cannot decode without knowing number of symbols. "
                    "Symbol count should be stored in ANSBitstream metadata."
                )

            # Create decoder (needs numpy array, not list)
            compressed_array = np.array(compressed, dtype=np.uint32)
            decoder = constriction.stream.stack.AnsCoder(compressed_array)
            model = constriction.stream.model.Categorical(probs, perfect=False)

            # Decode symbols
            decoded = decoder.decode(model, n_symbols)
            decoded_array = np.array(decoded, dtype=np.uint8)

            # Store in arena
            data_ref = world.arena.copy_tensor(decoded_array)

            # Create SymbolsU8 component
            symbols = SymbolsU8(data=data_ref, params=params)
            world.add_component(eid, symbols)
