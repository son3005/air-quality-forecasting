"""
models/Toto/model.py

Hugging Face implementation of Toto-313M wrapper adapted for multi-pollutant 
multi-station time-series forecasting.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class HFTotoModel(nn.Module):
    """
    Wrapper for Hugging Face Toto-2.0-313M adapted for Time Series Forecasting.
    """
    def __init__(self, seq_len=48, pred_len=1, enc_in=37,
                 model_name="DataDog/toto-313m", d_model=1024, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm

        try:
            # Try to load the base model from Hugging Face
            self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            # Try to infer hidden_size, fallback to passed d_model
            if hasattr(self.base_model.config, 'hidden_size'):
                d_model = self.base_model.config.hidden_size
            elif hasattr(self.base_model.config, 'd_model'):
                d_model = self.base_model.config.d_model
        except Exception as e:
            print(f"[Warning] Could not load {model_name} automatically: {e}")
            print(f"Fallback to a standard TransformerEncoder as placeholder.")
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
            self.base_model = nn.TransformerEncoder(encoder_layer, num_layers=4)

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
        
        # Forward through Toto
        if hasattr(self.base_model, 'last_hidden_state'):
            # If AutoModel returns a standard HF output object
            outputs = self.base_model(inputs_embeds=inputs_embeds)
            hidden = outputs.last_hidden_state # (B, seq_len, d_model)
        else:
            # Fallback for plain nn.Module
            hidden = self.base_model(inputs_embeds)
        
        # Flatten and project to output
        hidden_flat = hidden.reshape(hidden.size(0), -1)
        dec_out = self.projector(hidden_flat).view(hidden.size(0), self.pred_len, self.enc_in)

        if self.use_norm:
            # Inverse normalization
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
