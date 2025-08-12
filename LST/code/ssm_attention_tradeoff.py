import csv
from pathlib import Path


def main():
    data = [
        ('SSM', 0.85, 1.0),
        ('Attention', 0.90, 3.5)
    ]
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / 'ssm_vs_attention.csv'
    with out_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'accuracy', 'gflops'])
        writer.writerows(data)
    for row in data:
        print(row)

if __name__ == '__main__':
    main()
