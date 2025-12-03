# Cognitive Architectures: Bridging Literal and Functional Theory of Mind

## 7.1 The Cognitive Gap in LLMs
Theory of Mind (ToM)—the ability to impute mental states to others—is a critical capability for AI interaction. However, current benchmarks are deeply flawed. They measure "Literal ToM" (answering questions about a story) but fail to measure "Functional ToM" (using that understanding to act strategically). LLMs often pass the former and fail the latter, revealing a "Cognitive Gap."

Furthermore, standard benchmarks often overlook foundational visual reasoning capabilities. The "VisFactor" benchmark reveals that while Multimodal LLMs (MLLMs) excel at high-level tasks, they struggle with basic visual cognition tasks like mental rotation or spatial relation inference, which are trivial for humans. This suggests that current evaluation metrics are masking fundamental deficits in the model's world model.

## 7.2 Proposed Research: The BDI Wrapper and VisFactor Analysis
This research focuses on evaluation and architecture, requiring only inference access.

### Methodology: Functional Benchmarking and Wrapper Design
*   **Functional ToM Benchmarking**: Design a "Strategic Communication" benchmark where the LLM must deceive or cooperate with a user based on hidden information. This tests functional ToM. The benchmark should include simple matrix games where success depends on correctly modeling the opponent's belief state.
*   **Epistemic Logic Wrapper**: Design a prompt-based or lightweight code-based wrapper that forces the LLM to explicitly represent the "Belief State" of the user (e.g., BDI architecture: Belief-Desire-Intention). Before the LLM answers, it must update a "State of Mind" graph (e.g., "User believes X," "User does not know Y"). Compare raw LLM performance vs. "BDI-Wrapped" LLM performance.
*   **Psychometric Factor Analysis**: Apply the VisFactor methodology to text-only reasoning. Can we identify specific "cognitive factors" (e.g., inductive reasoning, spatial scanning) that are surprisingly absent in SOTA models?

### Implications
This moves beyond "chatbots" to "cognitive agents." It addresses the "sycophancy" problem (where LLMs agree with user errors) by giving the model a stable representation of the user's misconceptions versus reality. It is a crucial step toward "Personalized AI" that is actually helpful rather than just agreeable.
