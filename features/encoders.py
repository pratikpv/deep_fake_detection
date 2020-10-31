from functools import partial

from timm.models.efficientnet import tf_efficientnet_b0_ns, tf_efficientnet_b7_ns
import torch.nn as nn

encoder_params = {
    "tf_efficientnet_b0_ns": {
        "flat_features_dim": 1280,
        "imsize": 224,
        "init_op": partial(tf_efficientnet_b0_ns, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "flat_features_dim": 2560,
        "imsize": 600,
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2)
    }
}


def get_encoder(name=None):
    if name in encoder_params.keys():
        encoder = encoder_params[name]["init_op"]()
        return encoder
    else:
        raise Exception("Unknown encoder")


class DeepFakeEncoder(nn.Module):
    def __init__(self, encoder_name='tf_efficientnet_b7_ns'):
        super().__init__()
        self.encoder = get_encoder(encoder_name)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return x
