"""Codebook/codeword-based quantization codecs."""

from .non_uniform_loyd import (
	LloydMaxCodebookUInt8Codec,
	NonUniformLoydCodebookCodec,
	NonUniformLoydCodebookUInt8Codec,
	NonUniformLoydPerChannelCodebookCodec,
	NonUniformLoydPerChannelCodebookUInt8Codec,
	NonUniformLoydPerGroupCodebookCodec,
)
from .non_uniform_mlaw import (
	MuLawCodebookCodec,
	MuLawCodebookUInt8Codec,
	MuLawPerChannelCodebookCodec,
	MuLawPerChannelCodebookUInt8Codec,
	MuLawPerGroupCodebookCodec,
)
from .uniform import (
	UniformCodebookCodec,
	UniformCodebookUInt8Codec,
	UniformPerChannelCodebookCodec,
	UniformPerChannelCodebookUInt8Codec,
	UniformPerGroupCodebookCodec,
)

__all__ = [
	"LloydMaxCodebookUInt8Codec",
	"NonUniformLoydCodebookCodec",
	"NonUniformLoydCodebookUInt8Codec",
	"NonUniformLoydPerChannelCodebookCodec",
	"NonUniformLoydPerChannelCodebookUInt8Codec",
	"NonUniformLoydPerGroupCodebookCodec",
	"MuLawCodebookCodec",
	"MuLawCodebookUInt8Codec",
	"MuLawPerChannelCodebookCodec",
	"MuLawPerChannelCodebookUInt8Codec",
	"MuLawPerGroupCodebookCodec",
	"UniformCodebookCodec",
	"UniformCodebookUInt8Codec",
	"UniformPerChannelCodebookCodec",
	"UniformPerChannelCodebookUInt8Codec",
	"UniformPerGroupCodebookCodec",
]
