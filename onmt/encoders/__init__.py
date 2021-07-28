"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
# node encoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder

# hier encoder
from onmt.encoders.hier_encoder import HierEncoder

# graph encoder

str2enc = {"hier": HierEncoder}

str2node_enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
                "transformer": TransformerEncoder, "mean": MeanEncoder}
str2graph_enc = {}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "HierEncoder", "str2enc", "str2node_enc", "str2graph_enc"]
