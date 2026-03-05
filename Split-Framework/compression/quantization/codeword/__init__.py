"""Codebook/codeword-based quantization codecs."""

from .non_uniform import MuLawCodebookUInt8Codec
from .uniform import UniformCodebookUInt8Codec

__all__ = ["MuLawCodebookUInt8Codec", "UniformCodebookUInt8Codec"]
