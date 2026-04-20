import torch
import tvm_ffi
import os
import sys
import numpy as np

# Ensure we can import deepSeekV3_moe and integrated_moe_kernel
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import deepSeekV3_moe
from integrated_moe_kernel import integrated_moe, get_token_indices, gather_merged_weights

# Load compiled FFI library
lib_path = os.path.join(script_dir, "librouter_ffi.so")
if not os.path.exists(lib_path):
    print(f"Error: {lib_path} not found. Please run ./build.sh first.")
    # Try looking in the current directory as well
    lib_path = os.path.abspath("librouter_ffi.so")
    if not os.path.exists(lib_path):
        sys.exit(1)
tvm_ffi.load_module(lib_path)


def dispatch_fp8(hidden_states, token_expert_indices, token_expert_slots, expert_token_offsets, T, TOP_K, E_LOCAL, local_offset):
    """
    Simple Python implementation of FP8 token dispatch (permutation) for verification.
    Moves tokens to their assigned expert slots on this rank.
    """
    device = hidden_states.device
    H = hidden_states.shape[1]
    total_assigned = expert_token_offsets[E_LOCAL].item()
    
    permuted = torch.zeros(total_assigned, H, dtype=hidden_states.dtype, device=device)
    
    indices_cpu = token_expert_indices.cpu()
    slots_cpu = token_expert_slots.cpu()
    offsets_cpu = expert_token_offsets.cpu()
    
    for t in range(T):
        for k in range(TOP_K):
            ge = indices_cpu[t, k].item()
            if local_offset <= ge < local_offset + E_LOCAL:
                le = ge - local_offset
                slot = slots_cpu[t, k].item()
                dest = offsets_cpu[le].item() + slot
                permuted[dest] = hidden_states[t]
                
    return permuted



@torch.no_grad()
def verify_moe(T=16, routed_scaling_factor=1.0, local_expert_offset=0):
    print(f"\n--- Verifying MoE Integrated Kernel (T={T}, scaling={routed_scaling_factor}, offset={local_expert_offset}) ---")
    
    # 1. Geometry
    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    TOP_K = 8
    device = 'cuda'
    
    # 2. Random Inputs
    torch.manual_seed(42)
    
    routing_logits = torch.randn(T, E_GLOBAL, device=device, dtype=torch.float32) * 0.1
    routing_bias = torch.randn(E_GLOBAL, device=device, dtype=torch.bfloat16) * 0.1
    
    hidden_states = (torch.randn(T, H, device=device, dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand(H // 128, T, device=device, dtype=torch.float32)
    
    gemm1_weights = (torch.randn(E_LOCAL, 2*I, H, device=device, dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand(E_LOCAL, (2*I)//128, H // 128, device=device, dtype=torch.float32)
    
    gemm2_weights = (torch.randn(E_LOCAL, H, I, device=device, dtype=torch.float32) * 0.1).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand(E_LOCAL, H // 128, I // 128, device=device, dtype=torch.float32)


    # 3. Call PyTorch Reference
    print("Running PyTorch Reference...")
    ref_out = deepSeekV3_moe.run(
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor
    )

    # 4. Call Integrated MoE Kernel (TVM FFI)
    print("Running Integrated MoE Kernel...")
    output = integrated_moe(
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor
    )
    torch.cuda.synchronize()

    # --- Step 6: COMPARISON ---
    mse = torch.mean((ref_out.to(torch.float32) - output.to(torch.float32))**2).item()
    max_diff = torch.max(torch.abs(ref_out.to(torch.float32) - output.to(torch.float32))).item()
    
    print("Results Comparison:")
    print(f"  MSE:      {mse:.2e}")
    print(f"  Max Diff: {max_diff:.2e}")
    
    # Tolerances (BF16/FP8 can be noisy with random randn inputs)
    if mse < 1e-4:
        print("  VERIFICATION SUCCESSFUL!")
    else:
        print("  VERIFICATION FAILED! Excessive divergence.")
        if mse < 1e-1:
            print("  (Note: Small divergence might be due to BF16/FP8 quantization noise)")

    
    # Cleanup memory
    del ref_out, output, hidden_states, gemm1_weights, gemm2_weights
    torch.cuda.empty_cache()

if __name__ == "__main__":
    #verify_moe(T=1, routed_scaling_factor=1.0, local_expert_offset=96)
    verify_moe(T=32, routed_scaling_factor=1.0, local_expert_offset=96)
    #verify_moe(T=128, routed_scaling_factor=1.0, local_expert_offset=96)
