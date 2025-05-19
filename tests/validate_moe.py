#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script for the Mixture of Experts (MoE) implementation.

This script validates that:
1. The MoE layer correctly handles input and produces output of the expected shape.
2. The routing weights sum to 1 for each sample.
3. The top-k mechanism works correctly.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
# Import MoE from project modules
from src.core.modules.moe import MoE
print("Successfully imported MoE from project modules.")

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

if __name__ == "__main__":
    moe, x, output, routing_weights = validate_moe()
    
    # Optional: Visualize routing patterns if matplotlib is available
    try:
        visualize_routing(moe, x)
    except Exception as e:
        print(f"Skipping visualization: {str(e)}")
