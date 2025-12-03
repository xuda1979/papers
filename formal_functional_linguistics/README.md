# The Formal-Functional Competence Gap: A Linguistics-First Approach

## 3.1 The Intermediate Language Challenge
A polarizing debate in cognitive science concerns whether LLMs truly "understand" language or merely parrot statistical regularities. A potent framework for resolving this is the distinction between "Formal Linguistic Competence" (knowledge of rules/grammar) and "Functional Linguistic Competence" (real-world usage/reasoning). While LLMs excel at the former, their performance on the latter is often brittle and prone to catastrophic failure when the surface form of a problem changes.

The "Intermediate Language Challenge" posits that the specific formal language used to represent a problem (e.g., First-Order Logic vs. Python vs. natural language) drastically affects an LLM's reasoning ability. However, we lack a formal theory explaining why certain intermediate representations are more "neural-friendly" than others. Recent findings indicate that LLMs can form incorrect associations between syntactic templates (grammar structures) and specific domains. For example, a model might learn that the grammatical structure of a geography question is associated with the entity "France," leading it to answer "France" even when the question is nonsense but grammatically identical.

## 3.2 Proposed Research: Prompt Linguistics and Neurosymbolic Translation
This project involves pure linguistic analysis and lightweight probing of existing models to map the topology of this competence gap.

### Methodology: Syntactic Perturbation and Neurosymbolic Benchmarking
*   **Syntactic Template Stress-Testing**: Construct a dataset of reasoning problems where the syntax (grammatical structure) is preserved but the semantics (content) is shifted to nonsense or unrelated domains. This isolates the model's reliance on grammatical heuristics versus genuine semantic parsing.
*   **Neurosymbolic Translation**: Evaluate the performance of "neurosymbolic" reasoning chains where the LLM translates natural language into various formalisms (e.g., Lambda Calculus, SQL, Prolog, Answer Set Programming).
*   **The Translation Distance Metric**: Develop a theoretical "Translation Distance" metric. The hypothesis is that reasoning accuracy is inversely proportional to the distance between the pre-training corpus distribution and the syntactic structure of the intermediate language. This requires no training, only the analysis of inference logs from open models like Llama 3 or Mistral.

### Implications
This work would revolutionize "Prompt Engineering" by transforming it into "Prompt Linguistics." Instead of guessing which prompts work, researchers could derive the optimal intermediate representation for any given reasoning task based on formal linguistic principles. It bridges the gap between symbolic AI and connectionist LLMs, offering a path to robust neurosymbolic systems that leverage the strengths of both paradigms.
