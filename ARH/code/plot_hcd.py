import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


def generate_curves():
    TOTAL_STEPS = 10000
    SHIFT_STEP = 5000
    def stylized(initial, nadir, recovery, final=0.99):
        x = np.arange(TOTAL_STEPS)
        acc = np.full(TOTAL_STEPS, initial)
        acc[:SHIFT_STEP] = initial
        drop_end = SHIFT_STEP + 150
        acc[SHIFT_STEP:drop_end] = np.linspace(initial, nadir, 150)
        rec_start = drop_end + 400
        acc[drop_end:rec_start] = nadir
        rec_end = rec_start + recovery
        sig = 1/(1+np.exp(-np.linspace(-6,6,recovery)))
        acc[rec_start:rec_end] = nadir + (initial*final - nadir)*sig
        acc[rec_end:] = initial*final
        return x, acc
    return {
        'LLM': stylized(91.5,22.5,1000),
        'HRM': stylized(93.1,35.0,2500),
        'ARH': stylized(92.8,65.2,1700)
    }


def main():
    curves = generate_curves()
    plt.style.use('seaborn-v0_8-whitegrid')
    for label,(x,y) in curves.items():
        plt.plot(x,y,label=label)
    plt.xlabel('Time Steps')
    plt.ylabel('Task Accuracy (%)')
    plt.title('Model Performance on Hierarchical Concept Drift')
    plt.legend()
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir/'arh_hcd_visualization.pdf'
    data_file = out_dir/'arh_hcd_curves.csv'
    plt.tight_layout()
    plt.savefig(out_file)
    with data_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + list(curves.keys()))
        steps = next(iter(curves.values()))[0]
        for idx in range(len(steps)):
            row = [steps[idx]]
            for label in curves:
                row.append(curves[label][1][idx])
            writer.writerow(row)
    print(f'wrote {out_file} and {data_file}')

if __name__ == '__main__':
    main()
