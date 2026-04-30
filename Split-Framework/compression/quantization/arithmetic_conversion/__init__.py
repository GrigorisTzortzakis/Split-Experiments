"""Arithmetic conversion codecs.

This package hosts codecs that convert between numeric formats (e.g., FP8 or
scaled int8) for communication/compression purposes.
"""

from .float8 import TruncationFloat8Codec
from .int8 import TruncationInt8Codec
from .int8_per_channel import PerChannelInt8Codec

__all__ = ["TruncationFloat8Codec", "TruncationInt8Codec", "PerChannelInt8Codec"]
