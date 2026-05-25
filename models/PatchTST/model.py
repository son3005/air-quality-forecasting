"""
models/PatchTST/model.py

Hugging Face implementation of PatchTST (Patch Time Series Transformer)
adapted for multi-pollutant multi-station forecasting.
"""

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction


class HFPatchTSTModel(nn.Module):
    """
    Wrapper for Hugging Face PatchTSTForPrediction.
    """
    def __init__(self, seq_len=48, pred_len=1, enc_in=37,
                 patch_len=16, stride=8, d_model=64, n_heads=4,
                 e_layers=2, d_ff=128, dropout=0.1, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_norm = use_norm

        config = PatchTSTConfig(
            context_length=seq_len,
            prediction_length=pred_len,
            num_input_channels=enc_in,
            patch_length=patch_len,
            stride=stride,
            d_model=d_model,
            encoder_attention_heads=n_heads,
            encoder_layers=e_layers,
            encoder_ffn_dim=d_ff,
            dropout=dropout,
            use_cls_token=False,
            distribution_output="student_t", # Required but we use MSE loss on the pointwise prediction
            # Depending on version of transformers, output might be distributions or points.
            # PatchTSTForPrediction can output a loss if given labels, or we can just get the predictions.
        )
        
        # We will extract the exact point predictions manually if needed
        # Or we use a simpler custom wrapper to bypass distribution output.
        # Wait, HuggingFace's PatchTSTForPrediction outputs 'prediction_outputs' which is shape (batch_size, num_samples, prediction_length, input_size) if num_samples is set.
        # Let's bypass distribution to output raw predictions if possible.
        # Actually, if we just want MSE loss, we can use a custom head on top of PatchTSTModel (the base encoder).
        from transformers import PatchTSTModel as HFBasePatchTSTModel
        self.base_model = HFBasePatchTSTModel(config)
        
        # Calculate number of patches
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        
        # Linear head for each channel independently
        self.head = nn.Linear(d_model * self.num_patches, pred_len)

    def forward(self, x):
        """
        x: (B, seq_len, enc_in)
        Returns: (B, pred_len, enc_in)
        """
        B, L, N = x.shape
        
        if self.use_norm:
            # Reversible instance normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # HF base model expects: past_values of shape (batch_size, context_length, num_input_channels)
        outputs = self.base_model(past_values=x)
        
        # outputs.last_hidden_state: (batch_size, num_input_channels, num_patches, d_model)
        hidden = outputs.last_hidden_state
        
        # Project through flat linear head: (B * N, num_patches * d_model) -> (B * N, pred_len)
        hidden_flat = hidden.reshape(B * N, -1)
        out = self.head(hidden_flat)
        
        # Reshape channels back: (B * N, pred_len) -> (B, N, pred_len) -> (B, pred_len, N)
        dec_out = out.reshape(B, N, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            # Inverse normalization
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out
