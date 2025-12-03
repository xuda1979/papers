# Developmental Mechanistic Interpretability: Tracking Circuit Evolution

## 2.1 The Black Box of Training Dynamics
Mechanistic interpretability aims to reverse-engineer neural networks into understandable algorithms, decomposing them into functional components. However, a major open problem is understanding how these mechanisms form. Most current work analyzes fully trained models, treating the training process as a static artifact. "Developmental Interpretability" proposes tracking the emergence of specific circuits (e.g., induction heads, modular arithmetic circuits) throughout the training trajectory.

There is a profound lack of understanding regarding the phase transitions—often called "grokking"—where a model abruptly shifts from memorization to generalization. Current research highlights "Open Problems in Mechanistic Interpretability," specifically calling for methods to predict capabilities that arise during training. We do not yet know if knowledge circuits evolve linearly or if they undergo distinct topological shifts, effectively "rewiring" themselves at critical thresholds. The "Fractured Entangled Representation" hypothesis suggests that in large models, representations may not be clean and factored but rather entangled in ways that degrade continual learning, yet this remains a hypothesis ripe for rigorous testing.

## 2.2 Proposed Research: The Embryology of Intelligence
This proposal leverages "small models" (e.g., 1-layer or 2-layer transformers) trained on algorithmic tasks (like modular addition or synthetic logic gates) which require minimal compute but exhibit complex emergent dynamics.

### Methodology: Circuit Tracking and Topological Analysis
*   **High-Frequency Checkpointing**: Train a small transformer on a modular arithmetic task (e.g., $a + b \pmod p$). Save checkpoints every few gradient steps, specifically focusing on the "grokking" window where validation loss plummets.
*   **Dynamic Circuit Discovery**: Apply "Transcoders" or Sparse Autoencoders (SAEs) to identify feature circuits at each step. Unlike static analysis, this method tracks the trajectory of features. Do features strictly specialize, or do they undergo "polysemantic" phases before settling?
*   **Network Centrality Metrics**: Model the attention heads as a graph and measure the "network centrality" of specific neurons over time. The hypothesis is that the graph structure densifies during memorization and then sparsifies (prunes itself) during the generalization phase.

### Implications
By characterizing the dynamics of circuit formation, this research could lead to "early warning systems" for dangerous capabilities in large models. If we can identify the "precursor circuits" that lead to deception or power-seeking before they fully manifest, we can intervene during training. Furthermore, understanding the "knowledge entropy" of these circuits could provide "efficiency diagnostics" that indicate when a model has effectively learned a task, allowing for early stopping and saving compute.
