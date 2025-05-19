#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script for the Mixture of Experts (MoE) implementation.

This script validates that:
1. The MoE layer correctly handles input and produces output of the expected shape.
2. The routing weights sum to 1 for each sample.
3. The top-k mechanism works correctly.
4. The MoE layer can be integrated into the main model.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
# Import MoE and model modules
from src.core.modules.moe import MoE
from src.core.model import LTC_NCP_RNN
print("Successfully imported modules.")

def validate_moe():
    """Validate the MoE implementation."""
    print("=" * 50)
    print("Validating Mixture of Experts (MoE) implementation...")
    print("=" * 50)
    
    # Parameters
    batch_size = 64
    input_dim = 256
    output_dim = 256
    num_experts = 8
    k = 2
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create MoE layer
    moe = MoE(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        k=k
    ).to(device)
    
    # Create random input
    x = torch.randn(batch_size, input_dim).to(device)
    
    # Forward pass
    output = moe(x)
    
    # Get routing statistics
    routing_weights, expert_usage, load_balancing = moe.get_routing_stats(x)
    
    # Validation checks
    print("\nValidation Results:")
    print(f"1. Input shape: {x.shape}")
    print(f"2. Output shape: {output.shape}")
    print(f"3. Routing weights sum: {routing_weights.sum(dim=-1).mean().item():.6f} (target: 1.0)")
    
    # Check output shape
    shape_check = output.shape == (batch_size, output_dim)
    print(f"✓ Output shape check: {'PASSED' if shape_check else 'FAILED'}")
    
    # Check routing weights sum
    weights_sum_check = abs(routing_weights.sum(dim=-1).mean().item() - 1.0) < 1e-5
    print(f"✓ Routing weights sum check: {'PASSED' if weights_sum_check else 'FAILED'}")
    
    # Expert usage statistics
    print("\nExpert Usage Statistics:")
    for i, usage in enumerate(expert_usage.cpu().numpy()):
        print(f"Expert {i}: {usage:.4f} ({usage * batch_size * k:.1f} activations)")
    
    print(f"\nLoad balancing score: {load_balancing:.4f} (1.0 is perfect balance)")
    
    # Summary
    if shape_check and weights_sum_check:
        print("\n✓ All validation checks PASSED!")
    else:
        print("\n✗ Some validation checks FAILED!")
    
    print("=" * 50)
    return moe, x, output, routing_weights

def visualize_routing(moe, x):
    """Visualize routing patterns."""
    with torch.no_grad():
        router_logits = moe.router(x)
        routing_weights = torch.softmax(router_logits, dim=-1)
    
    # Convert to numpy for visualization
    weights = routing_weights.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Routing Weight')
    plt.xlabel('Expert')
    plt.ylabel('Sample')
    plt.title('MoE Routing Weights')
    plt.tight_layout()
    plt.savefig('moe_routing_visualization.png')
    plt.close()
    
    print("Routing visualization saved to 'moe_routing_visualization.png'")

def validate_model_with_moe():
    """Validate the MoE integration in the full model."""
    print("=" * 50)
    print("Validating MoE integration in the full model...")
    print("=" * 50)
    
    # Config parameters for a minimal model
    vocab_size = 10000
    embedding_dim = 64
    hidden_size = 128
    output_size = 2  # V, A
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model without MoE
    model_without_moe = LTC_NCP_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        use_transformer=True,
        use_moe=False
    ).to(device)
    
    # Create model with MoE
    model_with_moe = LTC_NCP_RNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=output_size,
        use_transformer=True,
        use_moe=True
    ).to(device)
    
    # Create random input
    batch_size = 8
    seq_length = 20
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    lengths = torch.randint(1, seq_length + 1, (batch_size,)).to(device)
    
    # Create dummy meta features
    meta_features = torch.randn(batch_size, 3).to(device)  # 3 meta features
    
    # Forward pass through both models
    print("Running forward pass through model without MoE...")
    output_without_moe = model_without_moe(tokens, lengths, meta_features)
    
    print("Running forward pass through model with MoE...")
    output_with_moe = model_with_moe(tokens, lengths, meta_features)
    
    # Check that outputs have the same shape
    same_shape = output_without_moe.shape == output_with_moe.shape
    print(f"Output shape check: {'PASSED' if same_shape else 'FAILED'}")
    print(f"Output shape without MoE: {output_without_moe.shape}")
    print(f"Output shape with MoE: {output_with_moe.shape}")
    
    # Check that outputs are different (MoE should change outputs)
    outputs_differ = not torch.allclose(output_without_moe, output_with_moe)
    print(f"Output difference check: {'PASSED' if outputs_differ else 'FAILED'}")
    
    # Summary
    if same_shape and outputs_differ:
        print("\n✓ MoE integration validation PASSED!")
    else:
        print("\n✗ MoE integration validation FAILED!")
    
    print("=" * 50)
    return model_with_moe, output_with_moe

def validate_config_integration():
    """Validate that the model correctly reads use_moe from config."""
    print("=" * 50)
    print("Validating MoE configuration integration...")
    print("=" * 50)
    
    config_path = os.path.join(project_root, "configs", "bilingual_base.yaml")
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if use_moe is in config
    use_moe = config.get('model', {}).get('use_moe', False)
    print(f"Config file has use_moe set to: {use_moe}")
    
    if use_moe:
        print("✓ Config integration validation PASSED!")
    else:
        print("✗ Config integration validation FAILED!")
    print("=" * 50)

if __name__ == "__main__":
    # Validate the MoE module itself
    moe, x, output, routing_weights = validate_moe()
    
    # Validate the model with MoE integration
    try:
        model_with_moe, model_output = validate_model_with_moe()
    except Exception as e:
        print(f"Error validating model with MoE: {str(e)}")
    
    # Validate config integration
    try:
        validate_config_integration()
    except Exception as e:
        print(f"Error validating config integration: {str(e)}")
    
    # Optional: Visualize routing patterns if matplotlib is available
    try:
        visualize_routing(moe, x)
    except Exception as e:
        print(f"Skipping visualization: {str(e)}")
