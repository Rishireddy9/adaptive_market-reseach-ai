# Local LLM research setup

The generative analysis uses a local model served by Ollama. It is a research
assistant only: its recommendation does not place orders or bypass the strategy,
risk, or safety layers.

## Start a local model

Install Ollama, then from PowerShell:

```powershell
ollama pull qwen2.5:7b
ollama serve
```

Keep Ollama running in one terminal. In a second terminal, from the project
directory:

```powershell
$env:PYTHONPATH = "src"
python .\run_document_platform.py --llm-analyze
```

Choose a different model or endpoint with command-line options:

```powershell
python .\run_document_platform.py --llm-analyze --llm-model qwen2.5:7b --ollama-url http://localhost:11434
```

`OLLAMA_MODEL` and `OLLAMA_BASE_URL` can be set in the environment instead.
The program fails with an actionable error if Ollama is unavailable or its
output does not match the required JSON schema; it does not silently substitute
a fabricated LLM response.

## Historical model terminology

`--llm-train` is retained for compatibility, but it trains a scikit-learn text
classifier on generated historical samples. It is not a generative LLM and does
not fine-tune model weights. `--llm-analyze` is the actual local generative-model
path. Both are research-only; validate on real point-in-time market data before
using any result for research conclusions.
