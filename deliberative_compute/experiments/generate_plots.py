import json
import matplotlib.pyplot as plt
import os

def generate_tex_plot(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # We want to generate the TikZ code for the plot
    # The paper uses pgfplots

    # We will generate a file that can be \input into the paper
    # or just copy-pasted. Let's make it a standalone tex snippet.

    tex_content = r"""
\begin{tikzpicture}
\begin{axis}[
    title={\textbf{Budget--Performance Frontiers}},
    xlabel={Compute Budget (arbitrary units)},
    ylabel={Solution Quality (\%)},
    xmin=10, xmax=120,
    ymin=20, ymax=100,
    xtick={10, 40, 80, 120},
    ytick={20, 40, 60, 80, 100},
    legend pos=south east,
    grid=major,
    width=0.9\textwidth,
    height=0.4\textwidth,
    legend style={font=\tiny},
]
"""

    # Colors and markers for different algorithms
    styles = {
        "IGD": "color=blue,mark=square*",
        "UCB-IGD": "color=cyan,mark=triangle*",
        "RS-MCTT": "color=purple,mark=diamond*",
        "CSC": "color=orange,mark=pentagon*",
        "ADR": "color=teal,mark=otimes*",
        "Baselines": "color=red,mark=*"
    }

    # Baseline surrogates (simplification for plotting)
    # We treat CSC (low budget) or UCB (low budget) as proxies for baselines if needed,
    # or just plot all our methods.
    # The previous paper had "Baselines (SMR)" and "Ours (SMR)".
    # Let's plot our actual algorithms.

    for task in ["SMR", "SCG"]:
        dashed = ", dashed" if task == "SCG" else ""
        suffix = "(SCG)" if task == "SCG" else "(SMR)"

        for algo in data[task]:
            name = algo['name']
            pts = algo['points']
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

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(tex_content)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "results/simulated_results.json")
    output_path = os.path.join(script_dir, "../tex/fig/bpf_curves_generated.tex")

    generate_tex_plot(json_path, output_path)
