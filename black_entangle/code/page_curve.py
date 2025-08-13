import csv
import numpy as np
from pathlib import Path


def random_state(dim):
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def entropies(qubits=8):
    dim = 2 ** qubits
    psi = random_state(dim)
    ent = []
    for m in range(1, qubits + 1):
        psi_matrix = psi.reshape((2 ** m, 2 ** (qubits - m)))
        rho = psi_matrix @ psi_matrix.conj().T
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-12]
        entropy = -np.sum(eigvals * np.log(eigvals))
        ent.append((m, float(entropy)))
    return ent


def main():
    np.random.seed(0)
    data = entropies()
    out_file = Path(__file__).resolve().parent.parent / 'data' / 'page_curve.csv'
    out_file.parent.mkdir(exist_ok=True)
    max_m, max_s = max(data, key=lambda x: x[1])
    with out_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'S_rad'])
        writer.writerows(data)

    print("Page curve data written to page_curve.csv")
    print(f'Max entropy at m={max_m}: {max_s:.4f}')


if __name__ == '__main__':
    main()
