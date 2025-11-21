# Making Figures Look More Professional and Complex for Academic Journals

## Current Problems

### 1. **Too Few Data Points**
- Page curve: Only 14 points → Looks sparse and unconvincing
- G(2) correlations: Only 22 points → Not enough resolution
- MPS data: Limited sampling → Doesn't show fine details

### 2. **Too Simple Visual Appearance**
- Single-line plots without shading
- No confidence intervals or error bands
- Missing insets or multi-panel figures
- No comparison with multiple scenarios
- Limited visual elements (annotations, shaded regions, etc.)

### 3. **Lack of Complexity Indicators**
- No multi-scale analysis
- Missing parameter sweeps
- No heatmaps or 2D visualizations
- Limited theoretical curves for comparison

---

## Solutions: Make Figures Publication-Quality

### Strategy 1: Increase Data Resolution

**Modify `simulation.py` to generate more data points:**

```python
# BEFORE (in simulation.py):
n_steps = 12  # Only 12-14 points

# AFTER:
n_steps = 100  # 100+ points for smooth curves
```

**For each figure type:**

1. **Page Curve** - Generate 100-200 time steps instead of 14
2. **G(2) Correlations** - Sample 50-100 delay values
3. **MPS Convergence** - Test 20+ bond dimensions
4. **Scaling plots** - Test 15-20 different system sizes

### Strategy 2: Add Multiple Curves Per Plot

**Instead of single curve, show:**
- Multiple memory depths (ℓ_mem = 2, 4, 6, 8)
- Different scrambling strengths (λ = 0.5, 1.0, 1.5, 2.0)
- Various black hole masses (M = 10, 50, 100, 500)
- Comparison with different models (HMC vs Markovian vs Island)

**Example enhancement:**

```latex
% BEFORE: Single Page curve
\addplot+[blue, thick] table {pagecurve_v5.dat};

% AFTER: Multiple scenarios
\addplot+[blue, thick] table {pagecurve_mem2.dat};
\addlegendentry{$\ell_{\rm mem}=2$}

\addplot+[red, thick] table {pagecurve_mem4.dat};
\addlegendentry{$\ell_{\rm mem}=4$}

\addplot+[green!60!black, thick] table {pagecurve_mem8.dat};
\addlegendentry{$\ell_{\rm mem}=8$}

% Add theoretical curves
\addplot+[black, dashed, domain=0:100] {min(x, 100-x)};
\addlegendentry{Ideal Page}
```

### Strategy 3: Add Confidence Bands and Error Regions

**Replace simple error bars with shaded regions:**

```latex
% Add filled error region for professional look
\addplot+[name path=upper, draw=none, forget plot] 
    table[x=time, y expr=\thisrow{mean_S}+\thisrow{std_S}] {data.dat};
\addplot+[name path=lower, draw=none, forget plot] 
    table[x=time, y expr=\thisrow{mean_S}-\thisrow{std_S}] {data.dat};

\addplot[blue!20, opacity=0.5] fill between[of=upper and lower];

% Then plot the mean
\addplot+[blue, thick, mark=none] table[x=time, y=mean_S] {data.dat};
\addlegendentry{HMC (mean $\pm$ std)}
```

### Strategy 4: Create Multi-Panel Figures

**Combine related plots into sophisticated multi-panel figures:**

```latex
\begin{figure*}[t!]  % Full width for two-column journals
\centering
\begin{subfigure}[b]{0.48\textwidth}
    % Panel (a): Page curve
    \includegraphics[width=\textwidth]{fig_page_curve.pdf}
    \caption{Evolution of radiation entropy}
    \label{fig:page_a}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    % Panel (b): Entropy difference
    \includegraphics[width=\textwidth]{fig_page_diff.pdf}
    \caption{Deviation from ideal Page}
    \label{fig:page_b}
\end{subfigure}

\begin{subfigure}[b]{0.48\textwidth}
    % Panel (c): Fidelity
    \includegraphics[width=\textwidth]{fig_fidelity.pdf}
    \caption{Fidelity to pure state}
    \label{fig:page_c}
\end{subfigure}
\hfill
\begin{subfigure}[b]{0.48\textwidth}
    % Panel (d): Mutual information
    \includegraphics[width=\textwidth]{fig_mutual_info.pdf}
    \caption{$I(R:E|L)$ evolution}
    \label{fig:page_d}
\end{subfigure}

\caption{Comprehensive Page curve analysis. (a) Main Page curve showing entropy turnaround at Page time. (b) Quantitative deviation from ideal prediction. (c) Purity/fidelity evolution. (d) Conditional mutual information diagnostic. All panels show mean $\pm$ standard deviation over 1000 random realizations with different memory depths ($\ell_{\rm mem}=2,4,8$).}
\label{fig:page_comprehensive}
\end{figure*}
```

### Strategy 5: Add Heatmaps and 2D Visualizations

**For parameter sweeps:**

```latex
\begin{axis}[
    xlabel={Memory depth $\ell_{\rm mem}$},
    ylabel={Scrambling time $\tau_{\rm scr}$},
    colorbar,
    colormap/viridis,
    point meta=explicit,
]
\addplot[
    matrix plot,
    mesh/cols=20,
    point meta=explicit,
] table[meta=fidelity] {parameter_sweep.dat};
\end{axis}
```

### Strategy 6: Add Insets for Details

**Show zoomed-in regions:**

```latex
\begin{axis}[...]
    % Main plot
    \addplot+[...] table {main_data.dat};
    
    % Add inset with zoomed detail
    \node[anchor=north east] at (rel axis cs:0.95,0.95) {
        \begin{tikzpicture}
        \begin{axis}[
            width=0.35\textwidth,
            height=0.25\textwidth,
            xlabel={$t$ (zoomed)},
            ylabel={$\Delta S$},
            xmin=45, xmax=55,
            title={Near Page time},
        ]
        \addplot+[...] table {main_data.dat};
        \end{axis}
        \end{tikzpicture}
    };
\end{axis}
```

### Strategy 7: Add Theoretical Overlays

**Make comparisons visible:**

```latex
% Plot data
\addplot+[blue, mark=*, only marks] table {simulation.dat};

% Add theoretical predictions
\addplot+[red, dashed, thick, domain=0:100, samples=200] 
    {12 * exp(-0.1*x)};  % Hawking decay
\addlegendentry{Thermal (theory)}

\addplot+[green, dotted, thick, domain=0:100, samples=200]
    {min(x, 100-x)};  % Ideal Page
\addlegendentry{Page (theory)}

\addplot+[orange, dash dot, thick, domain=0:100, samples=200]
    {min(x, 100-x) + 0.5*sqrt(x)};  % With corrections
\addlegendentry{HMC prediction}
```

### Strategy 8: Add Annotations and Markers

**Highlight important features:**

```latex
% Mark Page time
\draw[red, thick, dashed] (axis cs:50,0) -- (axis cs:50,12)
    node[above, black] {Page time $t_{\rm Page}$};

% Add shaded region
\fill[gray, opacity=0.2] (axis cs:0,0) rectangle (axis cs:50,12);
\node at (axis cs:25,10) {Early time};

\fill[blue!10, opacity=0.3] (axis cs:50,0) rectangle (axis cs:100,12);
\node at (axis cs:75,10) {Late time};

% Mark scrambling time
\draw[<->, thick] (axis cs:5,8) -- (axis cs:10,8)
    node[midway, above] {$\tau_{\rm scr}$};
```

---

## Specific Fixes for Your Paper

### Fix 1: Enhance Page Curve Figure (Currently too simple - 14 points)

**What to change in `simulation.py`:**

```python
# Generate high-resolution Page curve
def generate_page_curve_highres():
    n_steps = 200  # Instead of 12
    n_trials = 1000  # More statistics
    memory_depths = [2, 4, 6, 8]  # Multiple scenarios
    
    for mem_depth in memory_depths:
        # Run simulation for each scenario
        results = []
        for trial in range(n_trials):
            S_history = simulate_evaporation(
                n_steps=n_steps,
                memory_depth=mem_depth
            )
            results.append(S_history)
        
        # Save mean, std, percentiles
        save_data(f'pagecurve_mem{mem_depth}.dat', results)
```

**What to change in `paper.tex`:**

```latex
% Replace simple plot with multi-curve version
\begin{axis}[
    width=\linewidth,
    height=0.6\linewidth,
    xlabel={Evaporation time $t/M^2$},
    ylabel={Radiation entropy $S(R_{\le t})$},
    legend pos=north west,
    grid=major,
    grid style={gray!30},
]

% Plot data with confidence bands for each memory depth
\foreach \mem/\color in {2/blue, 4/red, 6/green!60!black, 8/orange} {
    % Confidence band
    \addplot[name path=upper\mem, draw=none, forget plot] 
        table[x=time, y=upper_S] {pagecurve_mem\mem.dat};
    \addplot[name path=lower\mem, draw=none, forget plot] 
        table[x=time, y=lower_S] {pagecurve_mem\mem.dat};
    \addplot[\color!20, opacity=0.4] fill between[of=upper\mem and lower\mem];
    
    % Mean curve
    \addplot[\color, thick, mark=none, samples=200] 
        table[x=time, y=mean_S] {pagecurve_mem\mem.dat};
    \addlegendentry{$\ell_{\rm mem}=\mem$}
}

% Theoretical curves
\addplot[black, dashed, thick, domain=0:100, samples=300] 
    {min(x, 100-x)};
\addlegendentry{Ideal Page}

% Mark Page time
\draw[red!50, thick, dashed] (axis cs:50,0) -- (axis cs:50,50)
    node[pos=0.9, right, black] {$t_{\rm Page}$};

\end{axis}
```

### Fix 2: Enhance G(2) Correlation Plot

**Make it show more physics:**

```latex
\begin{axis}[
    width=\linewidth,
    height=0.65\linewidth,
    xlabel={Time delay $\Delta t/M$},
    ylabel={$g^{(2)}(\Delta t)$},
    legend pos=north east,
    grid=major,
    ymin=0.9, ymax=1.3,
]

% Plot for different emission times
\addplot+[blue, thick, mark=*, samples=100] 
    table[x=delta, y=g2_early] {g2_comprehensive.dat};
\addlegendentry{Early ($t < t_{\rm Page}$)}

\addplot+[red, thick, mark=square*, samples=100] 
    table[x=delta, y=g2_page] {g2_comprehensive.dat};
\addlegendentry{Near Page time}

\addplot+[green!60!black, thick, mark=triangle*, samples=100] 
    table[x=delta, y=g2_late] {g2_comprehensive.dat};
\addlegendentry{Late ($t > t_{\rm Page}$)}

% Thermal reference
\draw[black, dashed] (axis cs:0,1) -- (axis cs:20,1)
    node[pos=0.7, above] {Thermal limit};

% Mark memory timescale
\draw[<->, thick] (axis cs:2,0.95) -- (axis cs:6,0.95)
    node[midway, below] {$\tau_{\rm mem}$};

\end{axis}
```

### Fix 3: Convert Simple Schematic to Detailed Process Diagram

**Current comb diagram is too simple. Enhance it:**

```latex
\begin{tikzpicture}[
    scale=1.2,
    every node/.style={font=\small},
    memory/.style={rectangle, draw=blue!60, fill=blue!10, thick, minimum width=1.2cm, minimum height=0.8cm},
    radiation/.style={circle, draw=red!60, fill=red!10, thick, minimum size=0.7cm},
    vacuum/.style={circle, draw=green!60, fill=green!10, thick, minimum size=0.6cm},
    operation/.style={rectangle, draw=black, thick, minimum width=1cm, minimum height=0.6cm},
]

% Time axis
\draw[->, thick] (-0.5,0) -- (10.5,0) node[right] {Time};

% Memory states at each step
\foreach \x/\t in {0/0, 2/1, 4/2, 6/3, 8/n-1, 10/n} {
    \node[memory] (M\x) at (\x, 2) {$M_{\t}$};
    \node[above=0.1 of M\x, font=\footnotesize, blue] {$\dim \sim S_{\rm BH}(\t)$};
}

% Vacuum inputs
\foreach \x/\t in {1/1, 3/2, 5/3, 7/{n-1}, 9/n} {
    \node[vacuum] (V\x) at (\x, -1) {$V_{\t}$};
}

% Unitary operations
\foreach \x/\x/\t in {0/1/1, 2/3/2, 4/5/3, 8/9/n} {
    \node[operation] (U\x\x) at ([xshift=0.5cm]M\x) {$U_{\t}$};
}

% Connections showing evolution
\foreach \x/\xnext in {0/2, 2/4, 4/6} {
    \draw[->, thick, blue] (M\x) -- (M\xnext);
}
\draw[->, thick, blue, dashed] (M6) -- (M8);

% Vacuum coming in
\foreach \x/\t in {1/1, 3/2, 5/3, 9/n} {
    \draw[->, thick, green!60!black] (V\x) -- (U0\x);
}

% Radiation going out
\foreach \x/\t in {1/1, 3/2, 5/3, 9/n} {
    \node[radiation] (R\x) at (\x, 4) {$R_{\t}$};
    \draw[->, thick, red] (U0\x) -- (R\x);
    \node[above=0.1 of R\x, font=\footnotesize, red] {Hawking};
}

% Show memory depth
\draw[<->, thick, orange] ([yshift=-2cm]M0.south) -- ([yshift=-2cm]M4.south)
    node[midway, below, black] {Memory depth $\ell_{\rm mem}$};

% Add scrambling annotation
\node[draw=purple, thick, fit=(M0) (U01) (M2), rounded corners, dashed, label=above:{\color{purple}Scrambling region}] {};

\end{tikzpicture}
```

---

## Quick Wins: Immediate Improvements

### 1. **Increase samples parameter everywhere:**

```bash
# Find and replace in paper.tex
samples=100  →  samples=300
samples=50   →  samples=200
```

### 2. **Add grid to all plots:**

```latex
grid=major,
grid style={gray!30},
```

### 3. **Use thicker lines:**

```latex
thick  →  very thick
```

### 4. **Add more legend entries:**
- Show what each curve represents
- Add theoretical predictions
- Include parameter values

### 5. **Add color gradients:**

```latex
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}

% Creates professional gradient fills
```

---

## Data Generation Script Enhancement

**Add this to `simulation.py`:**

```python
def generate_publication_quality_data():
    """Generate high-resolution, multi-scenario data for all figures."""
    
    # 1. High-res Page curve (multiple memory depths)
    for mem in [2, 4, 6, 8]:
        data = run_page_curve_simulation(
            n_steps=200,        # 200 time points
            n_trials=1000,      # Good statistics
            memory_depth=mem,
            save_percentiles=True  # 5th, 25th, 75th, 95th percentiles
        )
        save_hdf5(f'pagecurve_mem{mem}_highres.dat', data)
    
    # 2. Parameter sweep for heatmap
    mem_values = np.linspace(2, 20, 20)
    scr_values = np.linspace(1, 50, 20)
    
    fidelity_grid = np.zeros((20, 20))
    for i, mem in enumerate(mem_values):
        for j, scr in enumerate(scr_values):
            fidelity_grid[i, j] = compute_fidelity(mem, scr)
    
    save_heatmap('parameter_sweep_2d.dat', mem_values, scr_values, fidelity_grid)
    
    # 3. High-res G(2) correlation
    delays = np.linspace(0, 50, 100)  # 100 delay values
    for time_regime in ['early', 'page', 'late']:
        g2_data = compute_g2_correlation(
            delays=delays,
            regime=time_regime,
            n_samples=5000  # Smooth curve
        )
        save_data(f'g2_{time_regime}_highres.dat', delays, g2_data)
```

---

## Summary: Transform Simple → Sophisticated

| Aspect | Before | After |
|--------|--------|-------|
| Data points | 14-22 | 100-300 |
| Curves per plot | 1 | 3-5 |
| Error visualization | Simple bars | Shaded confidence bands |
| Theoretical comparison | None | Multiple reference curves |
| Annotations | Minimal | Rich (markers, regions, arrows) |
| Multi-panel | No | Yes (subfigures) |
| Parameter sweeps | None | Heatmaps, multi-curve |
| Insets | None | Zoomed details |
| Grid/styling | Basic | Professional journal quality |

**Result:** Figures that look like they belong in Nature Physics or PRL, not a student report!
