# How To Guide

## Generate & Evaluate

Generate a dataset and run the `classify_and_count` method on it.
Does not require any human annotations.

1. Install prerequisits
   1. UV
   2. Ollama
2. Clone the repository
3. Run `uv sync` to ensure the dependneceis

4. Run the generation task

```bash
uv run generate --config experiments/config_evaluate_local_generated.yaml
```

5. Update the `dataset.name` property in the config file to match the generated dataset file.
6. Run the evaluation task

```bash
uv run evaluate --config experiments/config_evaluate_local_generated.yaml
```