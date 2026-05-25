"""
models/Mamba/model.py

Hugging Face implementation of Mamba adapted for multi-pollutant 
multi-station time-series forecasting.
"""

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

class HFMambaModel(nn.Module):
    """
    Wrapper for Hugging Face Mamba adapted for Time Series Forecasting.
    """
    def __init__(self, seq_len=48, pred_len=1, enc_in=37,
                 d_model=64, d_state=16, d_conv=4, e_layers=2,
                 use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm

        # Note: Mamba is optimized for CUDA. If you encounter issues on CPU
        # with the official `mamba_ssm`, the transformers fallback will be used.
        config = MambaConfig(
            d_model=d_model,
            n_layer=e_layers,
            vocab_size=1, # Not used since we pass inputs_embeds
            state_size=d_state,
            conv_kernel=d_conv
        )
        self.base_model = MambaModel(config)
        
        # Project our continuous features into d_model
        self.input_projection = nn.Linear(enc_in, d_model)
        
        # Output projection
        self.projector = nn.Linear(d_model * seq_len, pred_len * enc_in)

    def forward(self, x):
        """
        x: (B, seq_len, enc_in)
        Returns: (B, pred_len, enc_in)
        """
        if self.use_norm:
            # Reversible instance normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # Embed continuous input
        inputs_embeds = self.input_projection(x) # (B, seq_len, d_model)
        
        # Forward through Mamba
        outputs = self.base_model(inputs_embeds=inputs_embeds)
        hidden = outputs.last_hidden_state # (B, seq_len, d_model)
        
        # Flatten and project to output
        hidden_flat = hidden.reshape(hidden.size(0), -1)
        dec_out = self.projector(hidden_flat).view(hidden.size(0), self.pred_len, self.enc_in)

        if self.use_norm:
            # Inverse normalization
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
