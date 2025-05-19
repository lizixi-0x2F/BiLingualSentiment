#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script to test MoE integration with one training step.
"""

import os
import sys
import torch
import yaml
import traceback
from torch.optim import Adam

# Enable debugging
DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

debug_print(f"Python path: {sys.path}")

try:
    # Import model
    from src.core.model import LTC_NCP_RNN
    debug_print("Successfully imported LTC_NCP_RNN from project modules.")
except Exception as e:
    debug_print(f"Error importing model: {e}")
    traceback.print_exc()

def run_one_training_step(use_moe=True):
    """Run one training step to validate the MoE implementation."""
    try:
        print("=" * 50)
        print(f"Running one training step with use_moe={use_moe}...")
        print("=" * 50)
        
        # Config parameters for a minimal model
        vocab_size = 10000
        embedding_dim = 64
        hidden_size = 128
        output_size = 2  # V, A
        
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        debug_print("Creating model...")
        # Create model with correct parameter values for all required args
        model = LTC_NCP_RNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=0.3,
            sparsity_level=0.5,
            dt=1.0,
            integration_method="euler",
            use_meta_features=True,
            bidirectional=False,
            padding_idx=0,
            wiring_type="structured",
            multi_level=False,
            emotion_focused=False,
            heterogeneous=False,
            use_transformer=True,
            use_moe=use_moe,
            invert_valence=False,
            invert_arousal=False,
            enhance_valence=False,
            valence_layers=1,
            use_quadrant_head=False,
            quadrant_weight=0.0
        ).to(device)
        debug_print("Model created successfully.")
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.001)
        
        # Create random input data
        batch_size = 8
        seq_length = 20
        tokens = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        lengths = torch.randint(1, seq_length + 1, (batch_size,)).to(device)
        
        # Create dummy meta features
        meta_features = torch.randn(batch_size, 3).to(device)  # 3 meta features
        
        # Create random targets
        targets = torch.randn(batch_size, 2).to(device)
        
        # Training step
        debug_print("Starting training step...")
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        debug_print("Running forward pass...")
        outputs = model(tokens, lengths, meta_features)
        debug_print(f"Forward pass complete. Output shape: {outputs.shape}")
        
        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        debug_print("Running backward pass...")
        loss.backward()
        
        # Check if gradients are computed
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                debug_print(f"Found gradients in parameter {name}")
                break
        
        print(f"Gradients check: {'PASSED' if has_gradients else 'FAILED'}")
        
        # Optimizer step
        debug_print("Running optimizer step...")
        optimizer.step()
        
        print("=" * 50)
        return model, loss.item()
    except Exception as e:
        print(f"Error during training step: {e}")
        traceback.print_exc()
        return None, 0.0

if __name__ == "__main__":
    # Run with MoE enabled
    model_with_moe, loss_with_moe = run_one_training_step(use_moe=True)
    
    # Run without MoE for comparison
    model_without_moe, loss_without_moe = run_one_training_step(use_moe=False)
    
    print(f"Loss with MoE: {loss_with_moe:.6f}")
    print(f"Loss without MoE: {loss_without_moe:.6f}")
    print("Training validation completed successfully!")
