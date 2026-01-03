# 🧭 Prompt Optimization for Product Relevance

This repository evaluates different ways to **optimize prompts** for product relevance ranking tasks, using the **WANDS dataset** from Wayfair.  
It includes both **manual**, **LLM-based**, and **DSPy-based** optimization pipelines — and a unified evaluation framework to compare them.

---

## 🚀 Project Overview

The repository compares three prompt-generation strategies:

| Prompt                 | Description | Optimizer |
|:--------------------|:--|:--|
| 🧠 **brain_prompt** | Manually written baseline | Human |
| 🤖 **llm_prompt**   | Rewritten automatically by a small LLM | `LLMOptimizer` |
| 🧬 **gepa_prompt**  | Evolved automatically via feedback and reflection | `DSPyOptimizer` (GEPA) |

---

## 🧪 Evaluation Protocol

Each test presents a **search query** and two **candidate products (LHS, RHS)** from the WANDS dataset.  
The model decides which product better matches the query - `"LHS"`, `"RHS"`, or `"Neither"`.

**Metrics**

| Metric | Meaning |
|:--|:--|
| **Overall Agreement** | Match rate with human labels |
| **Accuracy (human-pref)** | Accuracy when human label isn’t “Neither” |
| **Coverage** | Share of cases where model makes a decision |
| **Selective Precision / Recall** | Precision and recall when model commits |

**Setup**
- 100 sampled query–product pairs  
- Pydantic-based structured outputs (`RankingResponse`)  
- Evaluation handled by `eval.py`

---

## 🛠️ How to Use

### 1️⃣ Optimize with DSPy
```python
from optimizers.dspy_optimizer import DSPyOptimizer
DSPyOptimizer().optimize_and_save(save_as_strategy="gepa_prompt")
```

### 2️⃣ Optimize with LLM
```python
from optimizers.llm_optimizer import LLMOptimizer
LLMOptimizer(client).optimize_and_save(save_as_strategy="llm_prompt")
```

### 3️⃣ Evaluate
```python
from eval import main
main(strategy=["gepa_prompt", "llm_prompt", "brain_prompt"])
```

### 4️⃣ Quickstart (end-to-end)
```python
from optimizers.dspy_optimizer import DSPyOptimizer
from optimizers.llm_optimizer import LLMOptimizer
from eval import main

DSPyOptimizer().optimize_and_save(save_as_strategy="gepa_prompt")
LLMOptimizer(client).optimize_and_save(save_as_strategy="llm_prompt")
main(strategy=["brain_prompt", "llm_prompt", "gepa_prompt"])
```

Results are saved under:
```
data/eval/
 ├── <strategy>_results.csv
 ├── <strategy>_summary.txt
 └── strategies_summary.csv
```

---

# 🧩 Repository Structure

```
.
├── llm_client.py              # LiteLLM wrapper (chat + structured output)
├── ranker.py                  # Ranker using prompt template + LLM responses
├── wands_data.py              # Data preparation and pairwise sampling
├── eval.py                    # Evaluation logic and metrics
├── optimizers/                # Prompt optimizers
│   ├── dspy_optimizer.py      # DSPy-based automated prompt optimizer
│   └── llm_optimizer.py       # LLM-based prompt rewriting optimizer
├── prompts/                   # Stored prompt templates
└── data/                      # WANDS dataset and evaluation results
```

---

# ⚙️ Requirements

- Python 3.10
- Dependencies:
```bash
pip install dspy gepa litellm pandas pydantic
```
- OpenAI environment variables (LLMClient):
```
OPENAI_API_KEY
OPENAI_API_BASE
```

---

## 🧾 Glossary

- Strategy: the name of a prompt template stored in `prompts/` (e.g., `brain_prompt`)
- Prompt file: the `.txt` template used by the ranker during evaluation

---

## 💡 Tips

- `LLMClient` is model-agnostic; start with `gpt-5-nano` for quick runs and switch to a larger model if you want higher accuracy.
- If your provider requires a custom base URL, set `OPENAI_API_BASE` or pass `api_base` to `LLMClient`.

---

## 📦 WANDS Dataset Layout

You can download the dataset from [
WANDS](https://github.com/wayfair/WANDS/tree/main) repo.

Quick download (CSV files only):
```bash
mkdir -p data/WANDS/dataset
curl -L -o data/WANDS/dataset/product.csv https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv
curl -L -o data/WANDS/dataset/query.csv https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv
curl -L -o data/WANDS/dataset/label.csv https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv
```

Place the downloaded WANDS files here:
```
data/WANDS/dataset/
  product.csv
  query.csv
  label.csv
```
