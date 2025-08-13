import csv
from pathlib import Path
import math

def estimate_gflops(model_type, N, D, K=128):
    """
    A simplified, illustrative calculation of GFLOPs for different model types.
    This is not a real performance measurement, but a conceptual estimate
    based on the dominant terms in the complexity analysis.
    The results are normalized for comparison.
    """
    if model_type == 'SSM':
        # A simplified model for Mamba-like SSMs, focusing on the linear dependency on N.
        # O(N * (D^2 + D*logD)) is complex, so we use a simpler O(N*D*D_inner) proxy.
        # Let's use an effective inner dimension for illustration.
        D_inner = 16
        ops = N * D * D_inner
    elif model_type == 'Attention':
        # Based on O(N^2 * D) for standard self-attention's main matmul.
        ops = (N**2) * D
    else:
        ops = 0

    # Return a raw operations count
    return ops

def main():
    # --- Parameters for the illustrative analysis ---
    N = 1024  # Sequence Length
    D = 768   # Model Dimension

    # --- Illustrative Accuracy values ---
    # These are not from a real experiment, but are chosen to reflect
    # the expected trade-offs discussed in the paper.
    accuracy_ssm = 0.85
    accuracy_attention = 0.90

    # --- Calculate illustrative GFLOPs ---
    ops_ssm = estimate_gflops('SSM', N, D)
    ops_attention = estimate_gflops('Attention', N, D)

    # Normalize GFLOPs for the table to make the comparison clearer.
    # We'll set the smaller value (SSM) to 1.0 for the table.
    norm_factor = ops_ssm
    gflops_ssm_norm = ops_ssm / norm_factor
    gflops_attention_norm = ops_attention / norm_factor

    # The data for the table
    data = [
        ('SSM', f"{accuracy_ssm:.2f}", f"{gflops_ssm_norm:.1f}"),
        ('Attention', f"{accuracy_attention:.2f}", f"{gflops_attention_norm:.1f}")
    ]

    # --- Output results ---
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / 'ssm_vs_attention.csv'

    print("--- Illustrative Model Comparison ---")
    print(f"Parameters: N={N}, D={D}")
    print(f"{'Model':<10} | {'Accuracy':<10} | {'Normalized GFLOPs':<20}")
    print("-" * 45)
    for name, acc, flops in data:
        print(f"{name:<10} | {acc:<10} | {flops:<20}")
    print("-" * 45)
    print("\nNote: GFLOPs are normalized for illustration, with SSM as the baseline (1.0).")
    print(f"This is a conceptual calculation, not a real benchmark.")

    with out_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy', 'gflops_normalized'])
        writer.writerows(data)

    print(f"\nWrote illustrative comparison to {out_file}")


if __name__ == '__main__':
    main()
