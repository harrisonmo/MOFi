# Experimental QLoRA candidates

`rsi.llm` provides an optional Qwen3-8B QLoRA training and inference path.
Training produces a versioned candidate; the evaluator and promotion rules
remain independent. GNN training and LLM adaptation are separate variables.
Training is an Evolution Loop operation and is not a Science Hub tool.

Install the `llm` extra in a separate Linux/CUDA environment. It conflicts with
the GNN extra. Inspect `configs/llm-qwen3-8b-qlora.json` and
`python -m rsi.llm --help` before preparing a run. The configuration pins the
base revision and specifies quantization, LoRA parameters, sequence limits and
training budgets. Model weights and training data are not bundled.

Admit training records with source hashes, independent verification and
family/paper-level split groups. Prevent held-out labels and evaluation evidence
from entering the training set. Use fresh output directories, record compute
costs, and independently evaluate any proposed successor before promotion.
The `rsi.llm.transport` backend implements the structured decision contract
for admitted local candidates. Training success alone is not a scientific gain.
