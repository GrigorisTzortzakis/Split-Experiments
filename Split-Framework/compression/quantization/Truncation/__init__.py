"""Truncation-only (8-bit) compression codecs."""

from .truncation_int8 import TruncationInt8Codec
from compression.quantization.arithmetic_conversion.float8 import TruncationFloat8Codec

__all__ = ["TruncationInt8Codec", "TruncationFloat8Codec"]

