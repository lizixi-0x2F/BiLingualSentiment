#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mixture of Experts (MoE) implementation with top-k=2 routing.

MoE routes the input to multiple expert networks and combines their outputs
based on a learned routing strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer with top-k=2 routing.
    
    This implementation uses a learned router to distribute input among
    multiple expert networks and then combines their outputs based on routing weights.
    
    Args:
        input_dim (int): Dimension of the input features
        output_dim (int, optional): Dimension of the output features. If None, uses input_dim
        num_experts (int, optional): Number of expert networks. Default: 4
        k (int, optional): Number of experts to route each input to. Default: 2
        expert_hidden_dim (int, optional): Hidden dimension in expert networks. Default: None (uses 4*input_dim)
        noise_factor (float, optional): Factor for adding noise to router logits. Default: 0.1
        dropout (float, optional): Dropout probability for expert networks. Default: 0.1
    """
    
    def __init__(
        self,
        input_dim,
        output_dim=None,
        num_experts=4,
        k=2,
        expert_hidden_dim=None,
        noise_factor=0.1,
        dropout=0.1
    ):
        super(MoE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # Can't route to more experts than we have
        self.noise_factor = noise_factor
        self.expert_hidden_dim = expert_hidden_dim if expert_hidden_dim is not None else 4 * input_dim
        
        # Create router (gate) network
        self.router = nn.Linear(input_dim, num_experts)
        
        # Create expert networks - each is a simple feedforward network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.expert_hidden_dim, self.output_dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Get router logits and add noise for exploration
        if self.training and self.noise_factor > 0:
            router_logits = self.router(x) + torch.randn_like(self.router(x)) * self.noise_factor
        else:
            router_logits = self.router(x)
        
        # Get routing weights with softmax
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts for each sample
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        
        # Normalize the top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Initialize the output tensor
        combined_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Compute the output from each selected expert and combine
        for i in range(self.k):
            # Get the indices of the i-th expert for each sample
            expert_indices = top_k_indices[:, i]
            
            # Get the weight for the i-th expert for each sample
            expert_weights = top_k_weights[:, i].unsqueeze(-1)
            
            # Initialize per-expert outputs
            expert_outputs = torch.zeros_like(combined_output)
            
            # Process each expert separately
            for expert_idx in range(self.num_experts):
                # Create a mask for samples that use this expert
                mask = (expert_indices == expert_idx)
                if mask.sum() > 0:
                    # Process only the samples that use this expert
                    expert_inputs = x[mask]
                    expert_output = self.experts[expert_idx](expert_inputs)
                    expert_outputs[mask] = expert_output
            
            # Combine outputs weighted by the router weights
            combined_output += expert_outputs * expert_weights
        
        return combined_output
    
    def get_routing_stats(self, x):
        """
        Get routing statistics for the given input.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            tuple: Contains:
                - routing_weights (torch.Tensor): Full routing weights
                - expert_usage (torch.Tensor): How many times each expert was used
                - load_balancing (float): How balanced the expert usage is (1.0 is perfect)
        """
        with torch.no_grad():
            router_logits = self.router(x)
            routing_weights = F.softmax(router_logits, dim=-1)
            
            # Get top-k experts
            _, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
            
            # Count how many times each expert is used
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.k):
                for expert_idx in range(self.num_experts):
                    expert_usage[expert_idx] += (top_k_indices[:, i] == expert_idx).sum().item()
            
            # Calculate load balancing - 1.0 is perfect balance
            expert_usage = expert_usage / (x.shape[0] * self.k)  # Normalize
            load_balancing = (self.num_experts * torch.min(expert_usage) / torch.sum(expert_usage)).item()
            
            return routing_weights, expert_usage, load_balancing


def test_moe():
    """Simple test for the MoE layer."""
    import numpy as np
    
    # Parameters
    batch_size = 32
    input_dim = 128
    output_dim = 128
    num_experts = 8
    
    # Create MoE layer
    moe = MoE(input_dim, output_dim, num_experts=num_experts)
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = moe(x)
    
    # Get routing statistics
    routing_weights, expert_usage, load_balancing = moe.get_routing_stats(x)
    
    # Print results
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights sum check: {routing_weights.sum(dim=-1).mean().item():.6f} (should be close to 1.0)")
    print(f"Expert usage: {expert_usage.numpy()}")
    print(f"Load balancing: {load_balancing:.4f} (1.0 is perfect balance)")
    
    # Check that output has the same shape as input
    assert output.shape == (batch_size, output_dim), "Output shape mismatch"
    
    # Check that routing weights sum to 1
    assert abs(routing_weights.sum(dim=-1).mean().item() - 1.0) < 1e-5, "Routing weights don't sum to 1"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_moe()
