"""Truncation-only compression codecs."""

from .truncation_int import TruncationInt8Codec, TruncationIntCodec
from compression.quantization.arithmetic_conversion.float import TruncationFloatCodec

__all__ = ["TruncationIntCodec", "TruncationInt8Codec", "TruncationFloatCodec"]

