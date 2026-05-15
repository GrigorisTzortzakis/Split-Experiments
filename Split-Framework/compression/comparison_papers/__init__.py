from importlib import import_module

__all__ = [
	"PaperAutoencoderClient",
	"PaperAutoencoderDecoder",
	"PaperAutoencoderEncoder",
	"PaperAutoencoderServer",
	"PaperAutoencoderSplitLearning",
	"PaperEntropyBackEnd",
	"PaperEntropyCodec",
	"PaperEntropyTransportCodec",
	"PaperEntropyFrontEnd",
	"PaperEntropyServerModel",
	"PaperEntropySplitLearning",
	"PaperTopKSparsityCodec",
	"SplitFCCodec",
]


_LAZY_EXPORTS = {
	"PaperAutoencoderClient": (".autoencoder_paper", "PaperAutoencoderClient"),
	"PaperAutoencoderDecoder": (".autoencoder_paper", "PaperAutoencoderDecoder"),
	"PaperAutoencoderEncoder": (".autoencoder_paper", "PaperAutoencoderEncoder"),
	"PaperAutoencoderServer": (".autoencoder_paper", "PaperAutoencoderServer"),
	"PaperAutoencoderSplitLearning": (".autoencoder_paper", "PaperAutoencoderSplitLearning"),
	"PaperEntropyBackEnd": (".entropy", "PaperEntropyBackEnd"),
	"PaperEntropyCodec": (".entropy", "PaperEntropyCodec"),
	"PaperEntropyTransportCodec": (".entropy", "PaperEntropyTransportCodec"),
	"PaperEntropyFrontEnd": (".entropy", "PaperEntropyFrontEnd"),
	"PaperEntropyServerModel": (".entropy", "PaperEntropyServerModel"),
	"PaperEntropySplitLearning": (".entropy", "PaperEntropySplitLearning"),
	"PaperTopKSparsityCodec": (".paper_top_k", "PaperTopKSparsityCodec"),
	"SplitFCCodec": (".split_fc", "SplitFCCodec"),
}


def __getattr__(name):
	if name not in _LAZY_EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module_name, attr_name = _LAZY_EXPORTS[name]
	module = import_module(module_name, __name__)
	value = getattr(module, attr_name)
	globals()[name] = value
	return value