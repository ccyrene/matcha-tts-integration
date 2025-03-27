import datetime as dt
import math
import random

import torch
import torch.nn as nn

from .components.flow_matching import CFM
from .components.text_encoder import TextEncoder
from converter.utils.model import (
    denormalize,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

from typing import Optional

class MatchaTTS(nn.Module):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats

        # if n_spks > 1:
        #     self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)
        
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def forward(self, x, x_lengths, nfe:int, temperature:float=1.0, length_scale:float=1.0, spks:Optional[torch.Tensor]=None):

        # if self.n_spks > 1:
        #     # Get speaker embedding
        #     spks = self.spk_emb(spks.long())

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        print(f"mu_x.shape: {mu_x.shape}")
        print(f"logw.shape: {logw.shape}")
        print(f"x_mask.shape: {x_mask.shape}")
        w = torch.exp(logw) * x_mask
        print(f"torch.ceil(w).shape: {torch.ceil(w).shape}")
        # print(f"length_scale.shape: {length_scale.shape}")
        w_ceil = torch.ceil(w) * length_scale.view(-1, 1, 1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, int(y_max_length_)).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        
        print(f"w_ceil.shape: {w_ceil.shape}")
        print(f"attn_mask.shape: {attn_mask.shape}")
        
        print(f"(0)w_ceil.squeeze(1).shape: {w_ceil.squeeze(1).shape}")
        print(f"(0)attn_mask.squeeze(1).shape: {attn_mask.squeeze(1).shape}")
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        print(f"attn.shape: {attn.shape}")

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        print(f"(0)mu_y.shape: {mu_y.shape}")
        mu_y = mu_y.transpose(1, 2)
        print(f"(1)mu_y.shape: {mu_y.shape}")
        encoder_outputs = mu_y[:, :, :y_max_length]
        
        print(f"encoder_outputs.shape: {encoder_outputs.shape}")

        # Generate sample tracing the probability flow
        
        decoder_outputs = self.decoder(mu_y, y_mask, nfe, temperature, spks)

        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return denormalize(decoder_outputs, self.mel_mean, self.mel_std)