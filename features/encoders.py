from timm.models.efficientnet import tf_efficientnet_b0_ns, tf_efficientnet_b7_ns
import torch.nn as nn


def get_encoder(name='default'):
    if name == 'default':
        name = 'efficientnet_b0'
    if name == 'efficientnet_b7':
        encoder = tf_efficientnet_b7_ns(pretrained=True, drop_path_rate=0.2)
    elif name == 'efficientnet_b0':
        encoder = tf_efficientnet_b0_ns(pretrained=True, drop_path_rate=0.2)
    else:
        raise Exception("Unknown encoder")

    return encoder


class DeepFakeEncoder(nn.Module):
    def __init__(self, encoder_name='default'):
        super().__init__()
        self.encoder = get_encoder(encoder_name)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return x
