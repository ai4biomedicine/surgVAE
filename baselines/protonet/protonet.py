import torch.nn as nn
import torch.nn.functional as F

def mlp_block(in_features, out_features):

    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU()
    )

class ProtoNet(nn.Module):
    '''
    Protonet model adapted for 1-dimensional data using MLP as encoder.
    '''
    def __init__(self, x_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            mlp_block(x_dim, hid_dim),
            mlp_block(hid_dim, hid_dim),
            mlp_block(hid_dim, hid_dim),
            mlp_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)