import json
import matplotlib.pyplot as plt
import os

def generate_tex_plot(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    tex_content = r"""
\begin{tikzpicture}
\begin{axis}[
    title={\textbf{Budget--Performance Frontiers}},
    xlabel={Compute Budget (arbitrary units)},
    ylabel={Solution Quality (\%)},
    xmin=10, xmax=50,
    ymin=0, ymax=100,
    xtick={10, 20, 30, 40, 50},
    ytick={0, 20, 40, 60, 80, 100},
    legend pos=south east,
    grid=major,
    width=0.9\textwidth,
    height=0.5\textwidth,
    legend style={font=\tiny},
]
"""

    styles = {
        "IGD": "color=blue,mark=square*",
        "UCB-IGD": "color=cyan,mark=triangle*",
        "RS-MCTT": "color=purple,mark=diamond*",
        "CSC": "color=orange,mark=pentagon*",
        "ADR": "color=teal,mark=otimes*",
        "Baselines": "color=red,mark=*"
    }

    # Map task names to labels
    task_map = {
        "GameOf24": "Math",
        "BitSearch": "Code"
    }

    for task_key, task_label in task_map.items():
        if task_key not in data: continue

        dashed = ", dashed" if task_label == "Code" else ""
        suffix = f"({task_label})"

        for algo in data[task_key]:
            name = algo['name']
            pts = algo['points']
            # Scale Y to percentage
            coords = "\n".join([f"    ({x}, {y*100:.1f})" for x, y in pts])

            style = styles.get(name, "color=black")

            tex_content += f"""
\\addplot[{style}{dashed}, thick] coordinates {{
{coords}
}};
\\addlegendentry{{{name} {suffix}}}
"""

    tex_content += r"""
\end{axis}
\end{tikzpicture}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(tex_content)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "results/simulated_results.json")
    output_path = os.path.join(script_dir, "../tex/fig/bpf_curves_generated.tex")

    generate_tex_plot(json_path, output_path)
