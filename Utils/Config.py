"""
This file is a configuration code of the official UCTransNet implementation:

    https://github.com/McGregorWwww/UCTransNet

Reference:
    H. Wang, P. Cao, J. Wang, and O. R. Zaiane,
    "UCTransNet: Rethinking the Skip Connections in U-Net from a
    Channel-wise Perspective with Transformer,"
    Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

"""
import ml_collections

n_channels = 1
n_labels = 1
epochs = 100
img_size = 256

learning_rate = 1e-4
batch_size = 1

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960   # Q1+Q2+Q3+Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.embeddings_dropout_rate3D = 0.1
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64
    config.n_classes = 1
    config.learning_rate=learning_rate
    return config
