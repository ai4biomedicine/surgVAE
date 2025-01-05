import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class feature_encoder_1d(nn.Module):
    def __init__(self, input_len, d_model):
        super(feature_encoder_1d, self).__init__()
        self.input_len = input_len
        self.d_model = d_model
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_len, self.input_len * self.d_model)
        )

    def forward(self, encoded_features):
        _b, _l = encoded_features.shape
        encoded_features = self.feature_encoder(encoded_features)  # (B x input_len) -> (B x input_len*d_model)
        encoded_features = encoded_features.view(_b, _l, self.d_model)  # (B x input_len*d_model) -> (Bxinput_lenxd_model)
        return encoded_features

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def forward(self, x):

        attn_output, _ = self.attention(x, x, x)
        x = attn_output + x
        return F.relu(self.norm(self.linear(x)))
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(z_dim//2, z_dim//2)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.attention1 = AttentionBlock(z_dim//2, z_dim//4, num_heads=4)
        self.layer2 = nn.Linear(z_dim//4, z_dim//8)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.attention2 = AttentionBlock(8, 4, num_heads=4)

        self.fc_mu_logvar = nn.Linear(input_dim* 4, 2 * z_dim)
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        
        x = self.attention1(x)
        
        x = F.relu(self.bn2(self.layer2(x)))
        
        x = self.attention2(x) # output of attention2 is (B x input_len x 128)
        
        x =x.flatten(1, 2) # (B x input_len x 128) -> (B x input_len*128)
        
        x = self.fc_mu_logvar(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
class VAE_attention(nn.Module):
    """Encoder and Decoder architecture for sequential data."""
    def __init__(self, z_dim=10, input_dim=664):
        super(VAE_attention, self).__init__()
        self.z_dim = z_dim
        # Encoder - attention
        self.encoder = Encoder(input_dim, z_dim)
        self.feature_encoder = feature_encoder_1d(input_dim, z_dim//2)
        # Decoder - DNN
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, input_dim),
        )


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, no_dec=False):
        encoded_features = self.feature_encoder(x)
        stats = self.encoder(encoded_features)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decoder(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

