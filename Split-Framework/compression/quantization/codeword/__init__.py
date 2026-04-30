"""Codebook/codeword-based quantization codecs."""

from .non_uniform_loyd import LloydMaxCodebookUInt8Codec, NonUniformLoydCodebookUInt8Codec
from .non_uniform_loyd_per_channel import NonUniformLoydPerChannelCodebookUInt8Codec
from .non_uniform_mlaw import MuLawCodebookUInt8Codec
from .non_uniform_mlaw_per_channel import MuLawPerChannelCodebookUInt8Codec
from .uniform import UniformCodebookUInt8Codec
from .uniform_per_channel import UniformPerChannelCodebookUInt8Codec

__all__ = [
	"LloydMaxCodebookUInt8Codec",
	"NonUniformLoydCodebookUInt8Codec",
	"NonUniformLoydPerChannelCodebookUInt8Codec",
	"MuLawCodebookUInt8Codec",
	"MuLawPerChannelCodebookUInt8Codec",
	"UniformCodebookUInt8Codec",
	"UniformPerChannelCodebookUInt8Codec",
]
