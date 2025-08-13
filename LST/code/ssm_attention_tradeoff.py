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

def generate_associative_recall_data():
    """
    Generates illustrative accuracy data for the Associative Recall task.
    These values are chosen to reflect the theoretical strengths of each architecture.
    """
    data = [
        ('Transformer', 100.0, 100.0, 100.0, 99.9),
        ('Mamba', 100.0, 98.5, 96.2, 94.1),
        ('LST (K=128)', 100.0, 100.0, 99.9, 99.8)
    ]
    return data

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

    # The data for the first table
    tradeoff_data = [
        ('SSM', f"{accuracy_ssm:.2f}", f"{gflops_ssm_norm:.1f}"),
        ('Attention', f"{accuracy_attention:.2f}", f"{gflops_attention_norm:.1f}")
    ]

    # The data for the second table
    recall_data = generate_associative_recall_data()

    # --- Output results ---
    out_dir = Path(__file__).resolve().parent
    tradeoff_file = out_dir / 'ssm_vs_attention.csv'
    recall_file = out_dir / 'associative_recall.csv'

    print("--- Illustrative Model Comparison ---")
    print(f"Parameters: N={N}, D={D}")
    print(f"{'Model':<10} | {'Accuracy':<10} | {'Normalized GFLOPs':<20}")
    print("-" * 45)
    for name, acc, flops in tradeoff_data:
        print(f"{name:<10} | {acc:<10} | {flops:<20}")
    print("-" * 45)
    print("\nNote: GFLOPs are normalized for illustration, with SSM as the baseline (1.0).")
    print(f"This is a conceptual calculation, not a real benchmark.")

    with tradeoff_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy', 'gflops_normalized'])
        writer.writerows(tradeoff_data)

    print(f"\nWrote illustrative comparison to {tradeoff_file}")

    print("\n--- Illustrative Associative Recall Accuracy (%) ---")
    header = ['Model'] + [f'N={n}' for n in [1000, 8000, 16000, 32000]]
    print(f"{header[0]:<15} | {header[1]:<10} | {header[2]:<10} | {header[3]:<10} | {header[4]:<10}")
    print("-" * 70)
    for row in recall_data:
        print(f"{row[0]:<15} | {row[1]:<10.1f} | {row[2]:<10.1f} | {row[3]:<10.1f} | {row[4]:<10.1f}")
    print("-" * 70)


    with recall_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'N=1K', 'N=8K', 'N=16K', 'N=32K'])
        writer.writerows(recall_data)

    print(f"\nWrote illustrative recall data to {recall_file}")


if __name__ == '__main__':
    main()
